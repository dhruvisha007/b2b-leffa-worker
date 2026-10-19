"""
B2B Leffa virtual try-on — RunPod serverless handler.

Higher-fidelity try-on than Qwen for garment structure (straps/trims): Leffa (Meta, MIT)
on SD-inpaint, with a garment-agnostic mask from human parsing (ONNX) + OpenPose + DensePose.

Robust for ANY outfit: segformer_b2_clothes ANALYSES the garment to decide what it covers
(top / bottom / dress / full co-ord) and routes it to the right Leffa pass(es) — a two-piece
co-ord (jacket + jeans) runs as TWO passes (lower then upper) so BOTH pieces land reliably
instead of Leffa latching onto one. segformer also isolates the garment to a clean packshot
(drops the garment photo's model + any shopping-app screenshot chrome). The result is then
composited back onto the ORIGINAL photo (keeps size/framing/background/brightness), with
length-aware masking, hand protection and a sharpen/vibrance pass.

Input  (event["input"]): { "person": <b64/dataURI>, "garment": <b64/dataURI>,
                           "garment_type"?: "upper_body"|"lower_body"|"dresses"  (optional; auto if absent),
                           "length"?: "short"|"knee"|"midi"|"long",
                           "clean_garment"?: bool, "composite"?: bool, "seed"?: int, ... }
Output: { "image": "data:image/png;base64,..." }  (or {"error","trace"})

Checkpoints download once from HF franciszzj/Leffa to LEFFA_CKPT; segformer caches in HF_HOME.
"""
import os
import io
import sys
import base64
import traceback

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import runpod

sys.path.insert(0, "/Leffa")  # cloned in the Dockerfile

CKPT = os.environ.get("LEFFA_CKPT", "/runpod-volume/leffa-ckpts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# segformer_b2_clothes clothing labels: 4 upper-clothes, 5 skirt, 6 pants, 7 dress, 8 belt.
CLOTH_LABELS = [4, 5, 6, 7, 8]

_vt = None
_parsing = None
_openpose = None
_densepose = None
_seg_proc = None
_seg_model = None


def _ensure_ckpts():
    """Download the whole Leffa ckpts tree to the volume once (cached thereafter)."""
    from huggingface_hub import snapshot_download
    marker = os.path.join(CKPT, "virtual_tryon.pth")
    if not os.path.exists(marker):
        # Only the pieces this worker uses (skip the ~7GB SDXL-inpainting variant we never load).
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir=CKPT,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "virtual_tryon.pth",
                "densepose/*",
                "humanparsing/*",
                "openpose/*",
                "stable-diffusion-inpainting/*",
            ],
        )


def _load_seg():
    """Load the segformer clothing parser (tiny) — used to analyse + isolate the garment."""
    global _seg_proc, _seg_model
    if _seg_model is None:
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        _seg_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _seg_model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        ).to(DEVICE).eval()


def _load():
    global _vt, _parsing, _openpose, _densepose
    if _vt is not None:
        _load_seg()
        return
    _ensure_ckpts()
    _load_seg()
    from leffa.model import LeffaModel
    from leffa.inference import LeffaInference
    from leffa_utils.densepose_predictor import DensePosePredictor
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    _parsing = Parsing(
        atr_path=os.path.join(CKPT, "humanparsing/parsing_atr.onnx"),
        lip_path=os.path.join(CKPT, "humanparsing/parsing_lip.onnx"),
    )
    _openpose = OpenPose(body_model_path=os.path.join(CKPT, "openpose/body_pose_model.pth"))
    _densepose = DensePosePredictor(
        config_path=os.path.join(CKPT, "densepose/densepose_rcnn_R_50_FPN_s1x.yaml"),
        weights_path=os.path.join(CKPT, "densepose/model_final_162be9.pkl"),
    )
    model = LeffaModel(
        pretrained_model_name_or_path=os.path.join(CKPT, "stable-diffusion-inpainting"),
        pretrained_model=os.path.join(CKPT, "virtual_tryon.pth"),
        dtype="float16",
    )
    _vt = LeffaInference(model=model)


def _b64_to_img(b):
    if isinstance(b, str) and "," in b and b.strip().startswith("data:"):
        b = b.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _segment(img):
    """Per-pixel segformer label map [H, W] for an image."""
    inputs = _seg_proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _seg_model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    return up.argmax(dim=1)[0].cpu().numpy()


def _isolate_crop(img, seg, pad=0.08):
    """Isolate the garment onto white + crop to it → a clean packshot. Drops the garment photo's
    model/background AND any shopping-app screenshot chrome (none of which is a clothing label)."""
    cloth = np.isin(seg, CLOTH_LABELS)
    if int(cloth.sum()) < 200:
        return img                      # couldn't segment → use as-is
    arr = np.array(img)
    out = np.full_like(arr, 255)
    out[cloth] = arr[cloth]
    ys, xs = np.where(cloth)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    H, W = arr.shape[:2]
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0 = max(0, y0 - py); y1 = min(H, y1 + py + 1)
    x0 = max(0, x0 - px); x1 = min(W, x1 + px + 1)
    return Image.fromarray(out[y0:y1, x0:x1])


def _region_length(seg, labels):
    """Coarse length of a garment region from its bounding-box aspect (tall = long)."""
    m = np.isin(seg, labels)
    if int(m.sum()) < 50:
        return None
    ys, xs = np.where(m)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    ar = h / float(max(1, w))
    if ar < 0.95:
        return "short"
    if ar < 1.5:
        return "knee"
    return None                         # long / maxi → no clip


def _classify_garment(seg):
    """Decide the Leffa pass(es) from the garment's segmentation. Returns a list of
    (garment_type, length) — one entry per pass; a two-piece co-ord returns TWO (lower then
    upper) so both pieces are applied reliably instead of Leffa dropping one."""
    up    = int((seg == 4).sum())
    skirt = int((seg == 5).sum())
    pants = int((seg == 6).sum())
    dress = int((seg == 7).sum())
    lower = skirt + pants
    total = up + lower + dress
    if total < 500:
        return [("dresses", None)]                      # couldn't read → safe default
    sig = 0.12 * total
    has_up, has_lower, has_dress = up > sig, lower > sig, dress > sig
    low_len = None if pants >= skirt else _region_length(seg, [5])  # trousers → full length

    if has_dress and dress >= max(up, lower):
        return [("dresses", _region_length(seg, [7]))]
    if has_up and has_lower:                            # full co-ord / two-piece
        return [("lower_body", low_len), ("upper_body", None)]
    if has_up:
        return [("upper_body", None)]
    if has_lower:
        return [("lower_body", low_len)]
    return [("dresses", None)]


def _clip_mask_by_length(mask, keypoints, length):
    """Garment-length-aware masking: clip the agnostic mask below a body-keypoint-anchored line so a
    SHORT garment can't be stretched to the model's full-length-pants mask. Keypoints are COCO-18 in
    the mask's own (384x512) space: hips=8,11 knees=9,12 ankles=10,13. None/long keep the full mask."""
    if not length or length in ("long", "maxi", "full", "pants", "lower_body"):
        return mask
    try:
        kp = np.array(keypoints["pose_keypoints_2d"], dtype=float).reshape(-1, 2)
    except Exception:
        return mask
    H = mask.size[1]  # 512

    def ymean(idxs):
        ys = [kp[i][1] for i in idxs if i < len(kp) and not (kp[i][0] == 0 and kp[i][1] == 0)]
        return float(np.mean(ys)) if ys else None

    hip = ymean([8, 11]) or 0.55 * H
    knee = ymean([9, 12]) or 0.75 * H
    ank = ymean([10, 13]) or 0.95 * H
    if length in ("short", "mini"):
        cut = hip + 0.45 * (knee - hip)      # mid-thigh
    elif length == "knee":
        cut = knee
    elif length == "midi":
        cut = knee + 0.5 * (ank - knee)      # mid-calf
    else:
        return mask
    cut = int(max(0, min(H, cut)))
    arr = np.array(mask)
    arr[cut:, :] = 0                         # unmask everything below the hemline
    return Image.fromarray(arr)


def _protect_hands(mask, keypoints, radius_scale=0.28):
    """Keep the model's ORIGINAL hands: zero the mask in a disk around each hand (just beyond the
    wrist) so Leffa doesn't regenerate them — SD mangles fingers. COCO-18 384x512: RElbow=3 RWrist=4,
    LElbow=6 LWrist=7."""
    try:
        kp = np.array(keypoints["pose_keypoints_2d"], dtype=float).reshape(-1, 2)
    except Exception:
        return mask
    arr = np.array(mask)
    H, W = arr.shape[:2]
    yy, xx = np.ogrid[:H, :W]

    def ok(i):
        return i < len(kp) and not (kp[i][0] == 0 and kp[i][1] == 0)

    for elb, wr in ((3, 4), (6, 7)):
        if not (ok(elb) and ok(wr)):
            continue
        ex, ey = kp[elb]
        wx, wy = kp[wr]
        fore = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
        if fore < 1:
            continue
        cx = wx + 0.30 * (wx - ex)
        cy = wy + 0.30 * (wy - ey)
        r = max(12.0, radius_scale * fore)
        arr[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 0
    return Image.fromarray(arr)


def _composite_back(orig, gen, mask, enhance=True,
                    sharpen=90, color=1.16, contrast=1.08, bright=1.0):
    """Paste Leffa's garment region back onto the ORIGINAL full-res photo so the output keeps the
    input's exact size/framing/background/brightness — only the masked clothing changes. The garment
    region (from Leffa's softer 768x1024 SD output) is sharpened + nudged in colour/contrast so it
    matches the crisp original. resize_and_center is fit+white-pad, so the centred box inverts cleanly."""
    W, H = orig.size
    scale = min(1024.0 / H, 768.0 / W)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    top = (1024 - new_h) // 2
    left = (768 - new_w) // 2
    box = (left, top, left + new_w, top + new_h)
    gen_c = gen.crop(box).resize((W, H), Image.LANCZOS)
    if enhance:
        if sharpen:
            gen_c = gen_c.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen), threshold=2))
        if color and color != 1.0:
            gen_c = ImageEnhance.Color(gen_c).enhance(float(color))
        if contrast and contrast != 1.0:
            gen_c = ImageEnhance.Contrast(gen_c).enhance(float(contrast))
        if bright and bright != 1.0:
            gen_c = ImageEnhance.Brightness(gen_c).enhance(float(bright))
    m = mask.convert("L").crop(box).resize((W, H), Image.LANCZOS)
    m = m.filter(ImageFilter.GaussianBlur(radius=max(1, W // 300)))  # feather the seam
    return Image.composite(gen_c, orig.convert("RGB"), m)


def _tryon_once(person, garment_ref, gtype, length, steps, scale, seed, inp):
    """One Leffa pass for a single region (gtype) and composite it back onto `person`.
    Returns the composited PIL (same size as `person`), ready to be fed into the next pass."""
    from leffa.transform import LeffaTransform
    from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd

    src = resize_and_center(person, 768, 1024)
    ref = resize_and_center(garment_ref, 768, 1024)

    model_parse, _ = _parsing(src.resize((384, 512)))
    keypoints = _openpose(src.resize((384, 512)))
    mask = get_agnostic_mask_hd(model_parse, keypoints, gtype)
    mask = _clip_mask_by_length(mask, keypoints, length)
    if inp.get("protect_hands", True):
        mask = _protect_hands(mask, keypoints, float(inp.get("hand_radius", 0.28)))
    mask = mask.resize((768, 1024))

    seg = _densepose.predict_seg(np.array(src))[:, :, ::-1]
    densepose = Image.fromarray(seg)

    data = {"src_image": [src], "ref_image": [ref], "mask": [mask], "densepose": [densepose]}
    data = LeffaTransform()(data)
    out = _vt(data, num_inference_steps=steps, guidance_scale=scale, seed=seed, repaint=False)
    gen = out["generated_image"][0]

    if inp.get("composite", True):
        gen = _composite_back(
            person, gen, mask,
            enhance=bool(inp.get("enhance", True)),
            sharpen=inp.get("q_sharpen", 90),
            color=float(inp.get("q_color", 1.16)),
            contrast=float(inp.get("q_contrast", 1.08)),
            bright=float(inp.get("q_bright", 1.0)),
        )
    return gen


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if not inp.get("person") or not inp.get("garment"):
            return {"error": "need both 'person' and 'garment' (base64/dataURI)"}

        person = _b64_to_img(inp["person"])
        garment = _b64_to_img(inp["garment"])
        steps = int(inp.get("steps", 30))
        scale = float(inp.get("guidance_scale", 2.5))
        seed = int(inp.get("seed", 42))

        _load()

        # Analyse + (optionally) isolate the garment once.
        try:
            gseg = _segment(garment)
        except Exception:
            gseg = None
        garment_ref = garment
        if gseg is not None and inp.get("clean_garment", True):
            garment_ref = _isolate_crop(garment, gseg)

        # Decide the pass(es): explicit garment_type wins; else auto-route from the garment.
        if inp.get("garment_type"):
            passes = [(inp["garment_type"], inp.get("length"))]
        elif gseg is not None:
            passes = _classify_garment(gseg)
        else:
            passes = [("dresses", inp.get("length"))]

        # Run each pass, feeding the result forward (two-piece co-ord = lower then upper).
        running = person
        for gtype, length in passes:
            running = _tryon_once(running, garment_ref, gtype, length, steps, scale, seed, inp)

        return {"image": _img_to_b64(running), "passes": [p[0] for p in passes]}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-1800:]}


runpod.serverless.start({"handler": handler})

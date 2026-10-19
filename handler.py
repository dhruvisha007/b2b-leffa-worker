"""
B2B Leffa virtual try-on — RunPod serverless handler.

Higher-fidelity try-on than Qwen for garment structure (straps/trims): Leffa (Meta, MIT)
on SD-inpaint, with a garment-agnostic mask from human parsing (ONNX) + OpenPose + DensePose.

Input  (event["input"]): { "person": <b64/dataURI>, "garment": <b64/dataURI>,
                           "garment_type"?: "upper_body"|"lower_body"|"dresses",
                           "steps"?: int, "guidance_scale"?: float, "seed"?: int }
Output: { "image": "data:image/png;base64,..." }  (or {"error","trace"})

The Leffa checkpoints (SD-inpaint + virtual_tryon.pth + densepose + humanparsing + openpose)
download once from HF franciszzj/Leffa to the network volume (LEFFA_CKPT), cached thereafter.
"""
import os
import io
import sys
import base64
import traceback

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import runpod

sys.path.insert(0, "/Leffa")  # cloned in the Dockerfile

CKPT = os.environ.get("LEFFA_CKPT", "/runpod-volume/leffa-ckpts")

_vt = None
_parsing = None
_openpose = None
_densepose = None


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


def _load():
    global _vt, _parsing, _openpose, _densepose
    if _vt is not None:
        return
    _ensure_ckpts()
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


def _clip_mask_by_length(mask, keypoints, length):
    """Garment-length-aware masking: clip the agnostic mask below a body-keypoint-anchored line so a
    SHORT garment can't be stretched to the model's full-length-pants mask. `length` is the garment's
    intended length. Keypoints are COCO-18 in the mask's own (384x512) space: hips=8,11 knees=9,12
    ankles=10,13. 'long'/'maxi'/'full'/'pants'/None keep the mask as-is (down to the ankles)."""
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


def _composite_back(orig, gen, mask, enhance=True,
                    sharpen=90, color=1.14, contrast=1.06, bright=1.06):
    """Paste Leffa's garment region back onto the ORIGINAL full-res photo so the output keeps the
    input's exact size, framing, background and brightness — only the masked clothing changes.
    resize_and_center() does fit+white-pad (no crop), so the person occupies a known centered box
    in the 768x1024 space; invert that box to map the result + mask back to the original.

    The garment region comes from Leffa's 768x1024 SD output, which is softer + slightly duller
    than a real photo, so (when enhance) it is sharpened (unsharp) and nudged in colour/contrast/
    brightness to match the crisp original it sits in — fixes the "dull / blurry garment" look."""
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


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if not inp.get("person") or not inp.get("garment"):
            return {"error": "need both 'person' and 'garment' (base64/dataURI)"}

        person = _b64_to_img(inp["person"])
        garment = _b64_to_img(inp["garment"])
        gtype = inp.get("garment_type", "upper_body")
        length = inp.get("length")  # top|short|mini|knee|midi|long|maxi|full — clips the mask hemline
        steps = int(inp.get("steps", 30))
        scale = float(inp.get("guidance_scale", 2.5))
        seed = int(inp.get("seed", 42))

        _load()
        from leffa.transform import LeffaTransform
        from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd

        src = resize_and_center(person, 768, 1024)
        ref = resize_and_center(garment, 768, 1024)

        model_parse, _ = _parsing(src.resize((384, 512)))
        keypoints = _openpose(src.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, gtype)
        mask = _clip_mask_by_length(mask, keypoints, length)  # garment-length-aware hemline
        mask = mask.resize((768, 1024))

        seg = _densepose.predict_seg(np.array(src))[:, :, ::-1]
        densepose = Image.fromarray(seg)

        data = {"src_image": [src], "ref_image": [ref], "mask": [mask], "densepose": [densepose]}
        data = LeffaTransform()(data)

        # LeffaInference exposes __call__ (no .inference method) — call the instance directly.
        out = _vt(data, num_inference_steps=steps, guidance_scale=scale, seed=seed, repaint=False)
        gen = out["generated_image"][0]
        # Composite the garment region back onto the original photo (keeps original size/framing/
        # brightness/background). Set composite:false to get the raw 768x1024 generation instead.
        if inp.get("composite", True):
            gen = _composite_back(
                person, gen, mask,
                enhance=bool(inp.get("enhance", True)),
                sharpen=inp.get("q_sharpen", 90),
                color=float(inp.get("q_color", 1.14)),
                contrast=float(inp.get("q_contrast", 1.06)),
                bright=float(inp.get("q_bright", 1.06)),
            )
        return {"image": _img_to_b64(gen)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-1800:]}


runpod.serverless.start({"handler": handler})

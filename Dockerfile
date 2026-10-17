# syntax=docker/dockerfile:1
# B2B Leffa virtual try-on — RunPod serverless worker.
# Leffa (Meta, MIT) = higher-fidelity try-on than Qwen for garment structure. It needs human
# parsing (ONNX), OpenPose (.pth) AND DensePose. DensePose is NOT vendored and NOT in Leffa's
# requirements.txt: leffa_utils/densepose_predictor.py hard-imports `detectron2` and `densepose`,
# so we compile detectron2 (CUDA ops) + its DensePose project from source here. The devel base
# image ships nvcc; FORCE_CUDA + a bounded arch list build the ops without a GPU present in CI.
# Checkpoints (SD-inpaint + virtual_tryon.pth + densepose/humanparsing/openpose) download once to
# the network volume at runtime (LEFFA_CKPT), cached thereafter.
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    LEFFA_CKPT=/runpod-volume/leffa-ckpts \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9" \
    MAX_JOBS=2

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Leffa code: leffa/ (model, inference, transform), leffa_utils/ (densepose, masks), preprocess/.
RUN git clone --depth 1 https://github.com/franciszzj/Leffa.git /Leffa

# numpy MUST stay 1.x (torch 2.1.1 built against numpy 1; numpy 2 breaks torch.from_numpy).
RUN pip install --no-cache-dir \
      "numpy==1.26.4" \
      runpod hf_transfer "huggingface_hub>=0.23,<1.0" \
      "diffusers==0.25.1" "transformers==4.36.2" "accelerate==0.25.0" "peft==0.8.2" \
      safetensors einops omegaconf fvcore cloudpickle pycocotools scikit-image scipy timm \
      onnxruntime-gpu opencv-python-headless pillow matplotlib torchmetrics av regex==2024.5.15

# sanity 1: numpy stayed 1.x and torch<->numpy interop works
RUN python -c "import numpy,torch; assert numpy.__version__.startswith('1.'); assert int(torch.from_numpy(numpy.arange(3)).sum())==3; print('numpy',numpy.__version__,'torch',torch.__version__,'OK')"

# detectron2 core (compiles CUDA ops) + the DensePose project (pure-python on top of it).
# --no-deps on DensePose so pip does not try to re-resolve/replace detectron2 or numpy.
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"
RUN pip install --no-cache-dir --no-deps "git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose"

# sanity 2: report EVERY broken Leffa/densepose import in one pass (so one rebuild shows them all)
COPY sanity.py /sanity.py
RUN python /sanity.py

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]

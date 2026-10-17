# B2B Leffa virtual try-on — RunPod serverless worker.
# Leffa (Meta, MIT) = higher-fidelity try-on than Qwen for garment structure. Needs human
# parsing (ONNX), OpenPose (.pth) and DensePose. Leffa VENDORS densepose (requirements list
# detectron2's deps — fvcore/cloudpickle/omegaconf/pycocotools — but not detectron2 itself),
# so no from-source detectron2 compile. Checkpoints download once to the network volume.
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    LEFFA_CKPT=/runpod-volume/leffa-ckpts \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

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

# sanity 2: the Leffa modules import (catches a missing detectron2/densepose dep at build time)
RUN cd /Leffa && python -c "import sys; sys.path.insert(0,'/Leffa'); \
from leffa.model import LeffaModel; from leffa.inference import LeffaInference; from leffa.transform import LeffaTransform; \
from leffa_utils.densepose_predictor import DensePosePredictor; \
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd; \
from preprocess.humanparsing.run_parsing import Parsing; from preprocess.openpose.run_openpose import OpenPose; \
print('Leffa imports OK')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]

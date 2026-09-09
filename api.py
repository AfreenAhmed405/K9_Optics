import io
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

app = FastAPI(title="Dog Eye Pathology & Glaucoma Segmentation API")

# Hardware selection (MPS for M2, CUDA if available, otherwise CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_PATH = "best_resnet34.pth"
BACKBONE = "resnet34"
NUM_CLASSES = 5

CLASS_NAMES = ["Background", "Corneal Edema", "Episcleral Congestion", "Epiphora", "Cherry Eye"]
COLOR_PALETTE = {
    0: [0, 38, 255],     # Blue
    1: [0, 148, 255],    # Light blue (Corneal edema)
    2: [76, 255, 0],     # Green (Episcleral congestion)
    3: [255, 106, 0],    # Orange (Epiphora)
    4: [255, 0, 110]     # Magenta (Cherry Eye)
}

# Preprocessing
transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load model once at startup
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint '{MODEL_PATH}' successfully on {DEVICE}")
except Exception as e:
    print(f"Warning: Could not load '{MODEL_PATH}' ({e}). Running with uninitialized weights.")

model.to(DEVICE).eval()


def decode_mask_to_bgr(mask_2d: np.ndarray) -> np.ndarray:
    """Map class indices into RGB/BGR for overlay."""
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in COLOR_PALETTE.items():
        if cls_idx == 0:
            continue  # Leave background transparent
        color_mask[mask_2d == cls_idx] = [rgb[2], rgb[1], rgb[0]]  # BGR for OpenCV
    return color_mask


@app.post("/predict")
async def predict_eye(file: UploadFile = File(...)):
    # 1. Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocess & Tensor creation
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # 3. Model Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask_320 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Resize mask back to original image dimensions
    pred_mask_full = cv2.resize(pred_mask_320, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 5. Compute pathology metrics
    total_pixels = float(orig_h * orig_w)
    class_coverage = {}
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(pred_mask_full == idx))
        class_coverage[name] = {
            "pixel_count": count,
            "area_ratio": round(count / total_pixels, 4)
        }

    # Glaucoma risk criteria: Corneal Edema (Class 1) or Episcleral Congestion (Class 2)
    edema_ratio = class_coverage["Corneal Edema"]["area_ratio"]
    congestion_ratio = class_coverage["Episcleral Congestion"]["area_ratio"]
    is_glaucoma_suspect = bool(edema_ratio > 0.015 or congestion_ratio > 0.015)

    # 6. Generate Alpha Blend Visual Overlay
    color_mask = decode_mask_to_bgr(pred_mask_full)
    mask_regions = (pred_mask_full > 0)
    blended = img_bgr.copy()
    blended[mask_regions] = cv2.addWeighted(img_bgr, 0.45, color_mask, 0.55, 0)[mask_regions]

    # Encode overlay image to base64 JPEG
    _, buffer = cv2.imencode(".jpg", blended)
    base64_overlay = base64.b64encode(buffer).decode("utf-8")

    return {
        "glaucoma_risk": "HIGH_SUSPECT" if is_glaucoma_suspect else "LOW_NORMAL",
        "primary_indicators": {
            "corneal_edema_detected": bool(edema_ratio > 0.005),
            "episcleral_congestion_detected": bool(congestion_ratio > 0.005)
        },
        "class_breakdown": class_coverage,
        "overlay_base64": f"data:image/jpeg;base64,{base64_overlay}"
    }
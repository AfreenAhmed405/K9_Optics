import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import gradio as gr

# Hardware selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_PATH = "model_weights.pth"
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

# Preprocessing pipeline
transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load model
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
    print(f"Loaded '{MODEL_PATH}' successfully on {DEVICE}")
except Exception as e:
    print(f"Warning: Could not load '{MODEL_PATH}' ({e}). Running with uninitialized weights.")

model.to(DEVICE).eval()


def decode_mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    """Map class indices into RGB for overlay."""
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in COLOR_PALETTE.items():
        if cls_idx == 0:
            continue
        color_mask[mask_2d == cls_idx] = rgb
    return color_mask


def predict_eye(img_rgb: np.ndarray):
    if img_rgb is None:
        return None, "No image provided", {}

    orig_h, orig_w = img_rgb.shape[:2]

    # Preprocessing & Inference
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask_320 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask to original resolution
    pred_mask_full = cv2.resize(pred_mask_320, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Coverage metrics
    total_pixels = float(orig_h * orig_w)
    coverage = {}
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(pred_mask_full == idx))
        coverage[name] = round(count / total_pixels, 4)

    # Glaucoma risk logic
    edema_ratio = coverage["Corneal Edema"]
    congestion_ratio = coverage["Episcleral Congestion"]
    is_glaucoma_suspect = bool(edema_ratio > 0.015 or congestion_ratio > 0.015)
    risk_label = "HIGH RISK / SUSPECT" if is_glaucoma_suspect else "LOW RISK / NORMAL"

    # Alpha blend overlay in RGB space
    color_mask = decode_mask_to_rgb(pred_mask_full)
    mask_regions = (pred_mask_full > 0)
    blended = img_rgb.copy()
    blended[mask_regions] = cv2.addWeighted(img_rgb, 0.45, color_mask, 0.55, 0)[mask_regions]

    return blended, risk_label, coverage


# Generate an HTML badge legend (skipping background class 0)
legend_items_html = "".join([
    f'''<span style="display:inline-flex; align-items:center; gap:6px; font-size:13px; line-height:1;">
          <span style="flex-shrink:0; width:12px; height:12px; border-radius:3px; background-color:rgb({c[0]},{c[1]},{c[2]}); border:1px solid rgba(255,255,255,0.3);"></span>
          {CLASS_NAMES[i]}
        </span>'''
    for i, c in COLOR_PALETTE.items() if i != 0
])

legend_component = f"""
<div style="padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); width: 100%; box-sizing: border-box;">
    <div style="font-weight: 600; font-size: 11px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7;">Pathology Legend</div>
    <div style="display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;">
        {legend_items_html}
    </div>
</div>
"""

# Build UI
with gr.Blocks(title="Canine Eye Glaucoma & Pathology Detector") as demo:
    gr.Markdown("## Canine Eye Pathology & Glaucoma Segmentation")
    gr.Markdown("Upload an eye image to run semantic segmentation and estimate glaucoma risk.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Eye Image", sources=["upload"])
            btn = gr.Button("Analyze Eye", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="numpy", label="Segmentation Overlay")
            gr.HTML(legend_component)
            risk_box = gr.Textbox(label="Glaucoma Assessment")
            metrics_json = gr.JSON(label="Class Coverage Ratios")

    btn.click(
        fn=predict_eye,
        inputs=[input_img],
        outputs=[output_img, risk_box, metrics_json]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
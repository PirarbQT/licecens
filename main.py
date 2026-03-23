"""
YOLOv11 OCR Training Script — Thai License Plate Recognition
Dataset: LPR plate.v1i.yolov11 (124 classes: digits 0-9, Thai characters, provinces)
"""

from ultralytics import YOLO

# --- Configuration ---
DATA_YAML = "merged_lpr_motor/data.yaml"
MODEL = "yolo11n.pt"  # YOLOv11 nano (pretrained) — change to yolo11s/m/l/x for larger models
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # Adjust based on your GPU VRAM
DEVICE = 0  # GPU device index (0 = first GPU)
PROJECT = "runs/detect"
NAME = "lpr_plate_ocr"

# --- Learning Rate ---
LR0 = 0.01             # Initial learning rate (SGD=0.01, Adam=0.001)
LRF = 0.01             # Final learning rate factor (lr0 * lrf)
OPTIMIZER = "auto"      # Optimizer: SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto
WARMUP_EPOCHS = 3.0     # Warmup epochs
WARMUP_MOMENTUM = 0.8   # Warmup initial momentum
WARMUP_BIAS_LR = 0.1    # Warmup initial bias lr
COS_LR = False          # Use cosine LR scheduler (otherwise linear)

# --- Train ---
def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        lr0=LR0,
        lrf=LRF,
        optimizer=OPTIMIZER,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        cos_lr=COS_LR,
        patience=20,       # Early stopping patience
        save=True,          # Save checkpoints
        save_period=10,     # Save every N epochs
        plots=True,         # Generate training plots
        verbose=True,
    )
    print("✅ Training complete!")
    print(f"📁 Results saved to: {PROJECT}/{NAME}")

if __name__ == "__main__":
    main()

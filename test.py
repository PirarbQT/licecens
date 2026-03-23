"""
YOLOv11 OCR Model Testing — Thai License Plate Recognition
วิธีใช้:
  1. Evaluate บน test set:     python test.py
  2. Predict รูปเดียว:         python test.py --predict path/to/image.jpg
  3. Predict + โชว์ผลลัพธ์:     python test.py --predict path/to/image.jpg --show
  4. Predict ทั้งโฟลเดอร์:      python test.py --predict path/to/folder/
"""

import argparse
import csv
import os
import re
import cv2
from ultralytics import YOLO

# --- Config ---
# MODEL_PATH = "runs/detect/runs/detect/lpr_plate_ocr/weights/best.pt"
MODEL_PATH = "best.pt"  # โมเดลที่ดีที่สุด (project root)
DATA_YAML = "LPR plate.v1i.yolov11/data.yaml"
DEVICE = 0  # GPU
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_label_maps():
    """โหลด CSV mapping เพื่อแปลง label เป็นภาษาไทย"""
    label_map = {}

    # โหลด letter_map.csv (A1→ก, A2→ข, ...)
    letter_csv = os.path.join(BASE_DIR, "letter_map.csv")
    if os.path.exists(letter_csv):
        with open(letter_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()      # e.g. "A1"
                letter = row["letter"].strip()   # e.g. "ก"
                label_map[code] = letter
                # เพิ่ม zero-padded version: A1 → A01
                m = re.match(r"A(\d+)", code)
                if m:
                    padded = f"A{int(m.group(1)):02d}"
                    label_map[padded] = letter

    # โหลด province_map.csv (CMI→เชียงใหม่, BKK→กรุงเทพฯ, ...)
    province_csv = os.path.join(BASE_DIR, "province_map.csv")
    if os.path.exists(province_csv):
        with open(province_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                province = row["province"].strip()
                label_map[code] = province

    return label_map

def translate_label(cls_name, label_map):
    """แปลง label code → ภาษาไทย"""
    return label_map.get(cls_name, cls_name)  # ถ้าไม่เจอ ใช้ชื่อเดิม (เช่น ตัวเลข 0-9)

# โหลด mapping ตอนเริ่มรัน
LABEL_MAP = load_label_maps()

def evaluate(model):
    """Evaluate model on test set — ได้ mAP, precision, recall"""
    print("📊 Evaluating on test set...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=DEVICE,
        plots=True,
        verbose=True,
    )
    print("\n✅ Evaluation Results:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    return metrics

def predict(model, source, show=False):
    """Predict on image/folder — วาด bounding box + บันทึกผลลัพธ์"""
    print(f"🔍 Predicting on: {source}")
    results = model.predict(
        source=source,
        device=DEVICE,
        save=True,          # Save images with bounding boxes
        save_txt=True,      # Save detection labels
        conf=0.25,          # Confidence threshold
        iou=0.45,           # NMS IoU threshold
        show_labels=True,
        show_conf=True,
    )
    print(f"\n✅ Prediction complete! Results saved to: {results[0].save_dir}")
    
    # แสดงผลทุกรูป
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"\n📸 {r.path}")
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                thai_name = translate_label(cls_name, LABEL_MAP)
                conf = float(box.conf[0])
                if thai_name != cls_name:
                    print(f"   → {cls_name} → {thai_name} ({conf:.2f})")
                else:
                    print(f"   → {cls_name} ({conf:.2f})")
        else:
            print(f"\n📸 {r.path} — ไม่พบตัวอักษร")

    # โชว์ผลลัพธ์
    if show:
        for r in results:
            img = r.plot()  # วาด bounding box บนรูป
            cv2.imshow("YOLOv11 LPR Result", img)
        print("\n👀 กด key ใดก็ได้เพื่อปิดหน้าต่าง...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv11 LPR Model")
    parser.add_argument("--predict", type=str, help="Path to image or folder to predict")
    parser.add_argument("--show", action="store_true", help="แสดงผลลัพธ์เป็นหน้าต่างรูป")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)
    print(f"📦 Loaded model: {MODEL_PATH}")

    if args.predict:
        predict(model, args.predict, show=args.show)
    else:
        evaluate(model)

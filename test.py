"""
สคริปต์สำหรับทดสอบโมเดล YOLO OCR ป้ายทะเบียนไทย

วิธีใช้:
  1. ประเมินบน test set:
     python test.py
  2. ทำนายรูปเดี่ยว:
     python test.py --predict path/to/image.jpg
  3. ทำนายและเปิดหน้าต่างแสดงผล:
     python test.py --predict path/to/image.jpg --show
  4. ทำนายทั้งโฟลเดอร์:
     python test.py --predict path/to/folder/
"""

import argparse
import csv
import os
import re

import cv2
from ultralytics import YOLO

# ค่าตั้งต้นสำหรับการทดสอบโมเดล
MODEL_PATH = "best.pt"
DATA_YAML = "LPR plate.v1i.yolov11/data.yaml"
DEVICE = 0  # ใช้ GPU ตัวแรก
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# โหลด mapping จาก CSV เพื่อแปล label code เป็นตัวอักษรไทย/ชื่อจังหวัด
def load_label_maps():
    label_map = {}

    # โหลด letter_map.csv เช่น A1 -> ก
    letter_csv = os.path.join(BASE_DIR, "letter_map.csv")
    if os.path.exists(letter_csv):
        with open(letter_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                letter = row["letter"].strip()
                label_map[code] = letter

                # เพิ่มรูปแบบที่มี zero-padding เช่น A1 -> A01
                m = re.match(r"A(\d+)", code)
                if m:
                    padded = f"A{int(m.group(1)):02d}"
                    label_map[padded] = letter

    # โหลด province_map.csv เช่น BKK -> กรุงเทพมหานคร
    province_csv = os.path.join(BASE_DIR, "province_map.csv")
    if os.path.exists(province_csv):
        with open(province_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                province = row["province"].strip()
                label_map[code] = province

    return label_map


# แปลชื่อ class จากโค้ดภายในโมเดลให้เป็นข้อความที่อ่านเข้าใจง่ายขึ้น
def translate_label(cls_name, label_map):
    return label_map.get(cls_name, cls_name)


# โหลด mapping ครั้งเดียวตอนเริ่มรันไฟล์
LABEL_MAP = load_label_maps()


# ประเมินโมเดลบนชุด test แล้วพิมพ์ metric สำคัญออกมา
def evaluate(model):
    print("กำลังประเมินโมเดลบน test set...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=DEVICE,
        plots=True,
        verbose=True,
    )

    print("\nผลการประเมิน:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    return metrics


# ทำนายภาพหรือโฟลเดอร์ภาพ แล้วพิมพ์ผลการตรวจจับของแต่ละรูป
def predict(model, source, show=False):
    print(f"กำลังทำนายจาก: {source}")
    results = model.predict(
        source=source,
        device=DEVICE,
        save=True,      # บันทึกรูปที่วาดกรอบแล้ว
        save_txt=True,  # บันทึก label ที่ตรวจจับได้
        conf=0.25,      # ค่าความมั่นใจขั้นต่ำ
        iou=0.45,       # ค่า IoU สำหรับ NMS
        show_labels=True,
        show_conf=True,
    )
    print(f"\nทำนายเสร็จแล้ว ผลลัพธ์ถูกบันทึกไว้ที่: {results[0].save_dir}")

    # แสดงผลที่โมเดลตรวจจับได้ในแต่ละรูป
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"\nรูป: {r.path}")
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                thai_name = translate_label(cls_name, LABEL_MAP)
                conf = float(box.conf[0])

                if thai_name != cls_name:
                    print(f"   -> {cls_name} -> {thai_name} ({conf:.2f})")
                else:
                    print(f"   -> {cls_name} ({conf:.2f})")
        else:
            print(f"\nรูป: {r.path} - ไม่พบตัวอักษร")

    # ถ้าผู้ใช้เปิด --show ให้แสดงรูปในหน้าต่าง OpenCV
    if show:
        for r in results:
            img = r.plot()
            cv2.imshow("YOLOv11 LPR Result", img)

        print("\nกดปุ่มใดก็ได้เพื่อปิดหน้าต่างแสดงผล...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # รับ argument จาก command line เพื่อเลือกโหมด evaluate หรือ predict
    parser = argparse.ArgumentParser(description="Test YOLOv11 LPR Model")
    parser.add_argument("--predict", type=str, help="Path to image or folder to predict")
    parser.add_argument("--show", action="store_true", help="แสดงผลลัพธ์เป็นหน้าต่างรูป")
    args = parser.parse_args()

    # โหลดโมเดลก่อนเริ่มทำงาน
    model = YOLO(MODEL_PATH)
    print(f"โหลดโมเดลแล้ว: {MODEL_PATH}")

    # ถ้ามี --predict ให้ทำนายรูป ไม่เช่นนั้นจะ evaluate บน test set
    if args.predict:
        predict(model, args.predict, show=args.show)
    else:
        evaluate(model)

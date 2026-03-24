"""
สคริปต์สำหรับเทรนโมเดล YOLO OCR ป้ายทะเบียนไทย
ใช้กับ dataset ที่รวมรถยนต์และรถจักรยานยนต์ไว้แล้ว
"""

from ultralytics import YOLO

# ค่าตั้งต้นหลักของการเทรน
DATA_YAML = "merged_lpr_motor/data.yaml"
MODEL = "yolo11n.pt"  # โมเดลตั้งต้นขนาดเล็ก ถ้าต้องการใหญ่ขึ้นเปลี่ยนเป็น s/m/l/x
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # ปรับตาม VRAM ของ GPU
DEVICE = 0  # เลือก GPU ตัวที่ต้องการใช้
PROJECT = "runs/detect"
NAME = "lpr_plate_ocr"

# ค่าที่เกี่ยวกับ learning rate และ optimizer
LR0 = 0.01
LRF = 0.01
OPTIMIZER = "auto"
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1
COS_LR = False


# ฟังก์ชันหลักสำหรับเริ่มเทรนโมเดล
def main():
    # โหลดโมเดลตั้งต้นจากไฟล์ pretrained
    model = YOLO(MODEL)

    # สั่งเทรนด้วยค่าที่กำหนดด้านบน
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
        patience=20,    # ถ้า metric ไม่ดีขึ้นตามช่วงนี้จะหยุดเทรนก่อน
        save=True,      # บันทึก checkpoint ระหว่างเทรน
        save_period=10, # บันทึกทุก ๆ 10 epoch
        plots=True,     # สร้างกราฟสรุปผลการเทรน
        verbose=True,
    )

    print("เทรนเสร็จเรียบร้อย")
    print(f"ผลลัพธ์ถูกบันทึกไว้ที่: {PROJECT}/{NAME}")


if __name__ == "__main__":
    # เริ่มรันสคริปต์เทรนเมื่อเปิดไฟล์นี้ตรง ๆ
    main()

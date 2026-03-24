import argparse

from ultralytics import YOLO


# สคริปต์นี้ใช้เช็กว่าโมเดลมี class สำหรับแยกประเภทรถหรือยัง
# เช่น CAR_PLATE หรือ MOTO_PLATE
def main():
    parser = argparse.ArgumentParser(description="Check whether model has vehicle-type classes")
    parser.add_argument("--model", default="best.pt", help="Path to YOLO .pt model")
    parser.add_argument(
        "--expect",
        nargs="*",
        default=["CAR_PLATE", "MOTO_PLATE"],
        help="Expected vehicle classes",
    )
    args = parser.parse_args()

    # โหลดโมเดลแล้วดึงรายชื่อ class ทั้งหมดออกมา
    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    class_names = {str(v).upper() for v in names.values()}

    print(f"Model: {args.model}")
    print(f"Total classes: {len(class_names)}")

    # ตรวจว่า class ที่คาดหวังมีอยู่จริงในโมเดลหรือไม่
    missing = []
    for cls in args.expect:
        if cls.upper() in class_names:
            print(f"[OK] {cls}")
        else:
            print(f"[MISSING] {cls}")
            missing.append(cls)

    # สรุปผลว่าตอนนี้โมเดลยังเป็น OCR-only หรือเริ่มแยกประเภทรถได้แล้ว
    if missing:
        print("\nModel is still OCR-only for characters/provinces. Add vehicle classes to training dataset.")
    else:
        print("\nVehicle classes found. App can classify car vs motorcycle via vehicle_profiles.yaml")


if __name__ == "__main__":
    # เริ่มตรวจโมเดลเมื่อรันไฟล์นี้ตรง ๆ
    main()

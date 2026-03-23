import argparse

from ultralytics import YOLO


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

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    class_names = {str(v).upper() for v in names.values()}

    print(f"Model: {args.model}")
    print(f"Total classes: {len(class_names)}")

    missing = []
    for cls in args.expect:
        if cls.upper() in class_names:
            print(f"[OK] {cls}")
        else:
            print(f"[MISSING] {cls}")
            missing.append(cls)

    if missing:
        print("\nModel is still OCR-only for characters/provinces. Add vehicle classes to training dataset.")
    else:
        print("\nVehicle classes found. App can classify car vs motorcycle via vehicle_profiles.yaml")


if __name__ == "__main__":
    main()

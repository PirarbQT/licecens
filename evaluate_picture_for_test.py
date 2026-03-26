import csv
import json
import re
from pathlib import Path

from app import app

# ระบุตำแหน่งไฟล์และโฟลเดอร์ที่สคริปต์นี้ต้องใช้
BASE_DIR = Path(__file__).resolve().parent
PICTURE_DIR = BASE_DIR / "picture-for-test"
GROUND_TRUTH_CSV = BASE_DIR / "picture_for_test_ground_truth.csv"
REPORT_JSON = BASE_DIR / "picture_for_test_eval_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# กำหนด column ของไฟล์ ground truth ให้ตายตัว เพื่อให้แก้และเทียบผลได้ง่าย
CSV_FIELDS = [
    "relative_path",
    "expected_vehicle_family",
    "expected_plate_number",
    "expected_province",
    "expected_vehicle_type",
    "expected_plate_color",
    "expected_usage",
    "notes",
]


# ลิสต์ไฟล์รูปทั้งหมดใน picture-for-test แล้วเรียงลำดับให้คงที่
def list_picture_files():
    if not PICTURE_DIR.exists():
        return []
    return sorted(
        [p for p in PICTURE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p.relative_to(BASE_DIR)).lower(),
    )


# เดา vehicle family ขั้นต้นจากชื่อโฟลเดอร์ที่ผู้ใช้จัดไว้
def infer_vehicle_family(path: Path) -> str:
    parent_name = path.parent.name.strip().lower()
    if "มอไซ" in parent_name or "motor" in parent_name:
        return "รถจักรยานยนต์"
    if "รถยนต์" in parent_name or "car" in parent_name:
        return "รถยนต์"
    return ""


# ทำความสะอาดข้อความทั่วไปให้เทียบผลได้เสถียรขึ้น
def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ทำความสะอาดเลขทะเบียน โดยบีบช่องว่างและตัดอักขระแปลกที่ไม่จำเป็น
def normalize_plate_number(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^\u0E00-\u0E7F0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# แปลง vehicle_type ที่ระบบทำนายให้เป็น family ใหญ่ เพื่อเช็กว่าอย่างน้อยรถยนต์/มอไซค์ถูกหรือไม่
def predicted_vehicle_family(vehicle_type: str) -> str:
    text = normalize_text(vehicle_type)
    if "จักรยานยนต์" in text:
        return "รถจักรยานยนต์"
    if "รถยนต์" in text:
        return "รถยนต์"
    return ""


# โหลดไฟล์ ground truth ถ้ายังไม่มีจะสร้าง template ให้พร้อมใช้งาน
def ensure_ground_truth_file(image_paths):
    existing_rows = {}
    if GROUND_TRUTH_CSV.exists():
        with GROUND_TRUTH_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                relative_path = normalize_text(row.get("relative_path", ""))
                if not relative_path:
                    continue
                existing_rows[relative_path] = {field: row.get(field, "") for field in CSV_FIELDS}

    merged_rows = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(BASE_DIR).as_posix()
        existing = existing_rows.get(relative_path, {})
        merged_rows.append(
            {
                "relative_path": relative_path,
                "expected_vehicle_family": existing.get("expected_vehicle_family") or infer_vehicle_family(image_path),
                "expected_plate_number": existing.get("expected_plate_number", ""),
                "expected_province": existing.get("expected_province", ""),
                "expected_vehicle_type": existing.get("expected_vehicle_type", ""),
                "expected_plate_color": existing.get("expected_plate_color", ""),
                "expected_usage": existing.get("expected_usage", ""),
                "notes": existing.get("notes", ""),
            }
        )

    with GROUND_TRUTH_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    return merged_rows


# เรียก backend ปัจจุบันทีละรูป แล้วเก็บผลลัพธ์กลับมาเป็น list
def run_predictions(image_paths):
    client = app.test_client()
    predictions = {}

    for image_path in image_paths:
        relative_path = image_path.relative_to(BASE_DIR).as_posix()
        with image_path.open("rb") as f:
            response = client.post(
                "/api/detect",
                data={"image": (f, image_path.name)},
                content_type="multipart/form-data",
            )

        payload = response.get_json(silent=True) or {"error": response.get_data(as_text=True)}
        predictions[relative_path] = {
            "status_code": response.status_code,
            "payload": payload,
        }

    return predictions


# เปรียบเทียบค่าคาดหวังกับค่าที่ระบบทำนาย โดยใช้ normalizer ที่เหมาะกับ field นั้น
def compare_field(expected: str, predicted: str, normalizer) -> bool | None:
    expected_norm = normalizer(expected)
    if not expected_norm:
        return None
    predicted_norm = normalizer(predicted)
    return expected_norm == predicted_norm


# คำนวณสรุปผลและสร้าง report รายรูป
def build_report(ground_truth_rows, predictions):
    field_total = {
        "plate_number": 0,
        "province": 0,
        "vehicle_type": 0,
        "plate_color": 0,
        "usage": 0,
    }
    field_correct = {
        "plate_number": 0,
        "province": 0,
        "vehicle_type": 0,
        "plate_color": 0,
        "usage": 0,
    }
    family_total = 0
    family_correct = 0
    detailed_rows = 0
    full_match_rows = 0
    report_rows = []

    for truth in ground_truth_rows:
        relative_path = truth["relative_path"]
        prediction = predictions.get(relative_path, {})
        payload = prediction.get("payload", {})

        predicted_family = predicted_vehicle_family(payload.get("vehicle_type", ""))
        family_match = compare_field(
            truth.get("expected_vehicle_family", ""),
            predicted_family,
            normalize_text,
        )
        if family_match is not None:
            family_total += 1
            family_correct += int(family_match)

        field_matches = {
            "plate_number": compare_field(
                truth.get("expected_plate_number", ""),
                payload.get("plate_number", ""),
                normalize_plate_number,
            ),
            "province": compare_field(
                truth.get("expected_province", ""),
                payload.get("province", ""),
                normalize_text,
            ),
            "vehicle_type": compare_field(
                truth.get("expected_vehicle_type", ""),
                payload.get("vehicle_type", ""),
                normalize_text,
            ),
            "plate_color": compare_field(
                truth.get("expected_plate_color", ""),
                payload.get("plate_color", ""),
                normalize_text,
            ),
            "usage": compare_field(
                truth.get("expected_usage", ""),
                payload.get("usage", ""),
                normalize_text,
            ),
        }

        compared_fields = 0
        correct_fields = 0
        for key, value in field_matches.items():
            if value is None:
                continue
            field_total[key] += 1
            compared_fields += 1
            if value:
                field_correct[key] += 1
                correct_fields += 1

        if compared_fields > 0:
            detailed_rows += 1
            if correct_fields == compared_fields:
                full_match_rows += 1

        report_rows.append(
            {
                "relative_path": relative_path,
                "status_code": prediction.get("status_code", 0),
                "expected": truth,
                "predicted": payload,
                "predicted_vehicle_family": predicted_family,
                "family_match": family_match,
                "field_matches": field_matches,
            }
        )

    summary = {
        "total_images": len(ground_truth_rows),
        "family_total": family_total,
        "family_correct": family_correct,
        "detailed_rows": detailed_rows,
        "full_match_rows": full_match_rows,
        "field_total": field_total,
        "field_correct": field_correct,
    }
    return {"summary": summary, "rows": report_rows}


# พิมพ์ผลสรุปให้อ่านง่ายใน terminal
def print_summary(report):
    summary = report["summary"]
    print(f"จำนวนรูปทั้งหมด: {summary['total_images']}")
    print(f"Vehicle family accuracy: {summary['family_correct']}/{summary['family_total']}")
    print(f"Detailed rows ที่มี ground truth เต็มบางส่วน: {summary['detailed_rows']}")
    print(f"Full match rows: {summary['full_match_rows']}/{summary['detailed_rows']}")

    for field_key, label in (
        ("plate_number", "ทะเบียน"),
        ("province", "จังหวัด"),
        ("vehicle_type", "ประเภทรถ"),
        ("plate_color", "สีป้าย"),
        ("usage", "การใช้งาน"),
    ):
        total = summary["field_total"][field_key]
        correct = summary["field_correct"][field_key]
        print(f"{label}: {correct}/{total}")

    print(f"บันทึกรายงานไว้ที่: {REPORT_JSON.name}")
    print(f"แก้ ground truth ได้ที่: {GROUND_TRUTH_CSV.name}")


def main():
    image_paths = list_picture_files()
    if not image_paths:
        raise SystemExit("ไม่พบรูปในโฟลเดอร์ picture-for-test")

    ground_truth_rows = ensure_ground_truth_file(image_paths)
    predictions = run_predictions(image_paths)
    report = build_report(ground_truth_rows, predictions)

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)


if __name__ == "__main__":
    main()

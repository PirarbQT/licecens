import csv
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
VEHICLE_PROFILE_PATH = os.path.join(BASE_DIR, "vehicle_profiles.yaml")

app = Flask(__name__)


def load_label_maps():
    label_map = {}

    letter_csv = os.path.join(BASE_DIR, "letter_map.csv")
    if os.path.exists(letter_csv):
        with open(letter_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                letter = row["letter"].strip()
                label_map[code] = letter
                m = re.match(r"A(\d+)", code)
                if m:
                    label_map[f"A{int(m.group(1)):02d}"] = letter

    province_map = {}
    province_csv = os.path.join(BASE_DIR, "province_map.csv")
    if os.path.exists(province_csv):
        with open(province_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = row["code"].strip()
                province = row["province"].strip()
                label_map[code] = province
                province_map[code] = province

    return label_map, province_map


def load_vehicle_profiles():
    default_profiles = {
        "unknown": {
            "vehicle_type": "ไม่สามารถระบุได้",
            "plate_color": "ไม่สามารถระบุได้",
            "usage": "ไม่สามารถระบุได้",
            "aliases": [],
        },
        "car_private": {
            "vehicle_type": "รถยนต์ส่วนบุคคล",
            "plate_color": "ขาว (ส่วนบุคคล)",
            "usage": "ส่วนบุคคล",
            "aliases": ["CAR", "CAR_PLATE", "AUTO", "PRIVATE_CAR"],
        },
        "moto_private": {
            "vehicle_type": "รถจักรยานยนต์ส่วนบุคคล",
            "plate_color": "ขาว (รถจักรยานยนต์)",
            "usage": "ส่วนบุคคล",
            "aliases": ["MOTO", "MOTORCYCLE", "MOTO_PLATE", "BIKE"],
        },
    }

    if not os.path.exists(VEHICLE_PROFILE_PATH):
        return default_profiles

    with open(VEHICLE_PROFILE_PATH, encoding="utf-8-sig") as f:
        loaded = yaml.safe_load(f) or {}

    profiles = loaded.get("profiles", {})
    if not profiles:
        return default_profiles

    normalized = {}
    for key, value in profiles.items():
        normalized[key] = {
            "vehicle_type": value.get("vehicle_type", "ไม่สามารถระบุได้"),
            "plate_color": value.get("plate_color", "ไม่สามารถระบุได้"),
            "usage": value.get("usage", "ไม่สามารถระบุได้"),
            "aliases": [a.upper() for a in value.get("aliases", [])],
        }

    if "unknown" not in normalized:
        normalized["unknown"] = default_profiles["unknown"]
    return normalized


LABEL_MAP, PROVINCE_MAP = load_label_maps()
VEHICLE_PROFILES = load_vehicle_profiles()
VEHICLE_ALIAS_TO_CODE = {}
for code, profile in VEHICLE_PROFILES.items():
    for alias in profile.get("aliases", []):
        VEHICLE_ALIAS_TO_CODE[alias.upper()] = code

DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL = YOLO(MODEL_PATH)


def translate_label(cls_name: str) -> str:
    return LABEL_MAP.get(cls_name, cls_name)


def decode_image(file_storage):
    image_bytes = file_storage.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปได้")
    return image


def parse_plate_result(result):
    province_candidates = []
    confidence_values = []
    alnum_candidates = []
    vehicle_candidates = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        cls_upper = str(cls_name).upper()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        confidence_values.append(conf)

        if cls_upper in VEHICLE_ALIAS_TO_CODE:
            vehicle_candidates.append((conf, VEHICLE_ALIAS_TO_CODE[cls_upper]))
            continue

        if cls_name.isdigit():
            alnum_candidates.append(
                {
                    "type": "digit",
                    "label": cls_name,
                    "conf": conf,
                    "x": x_center,
                    "y": y_center,
                    "w": width,
                    "h": height,
                }
            )
            continue

        if cls_name.startswith("A"):
            thai_letter = translate_label(cls_name)
            alnum_candidates.append(
                {
                    "type": "letter",
                    "label": thai_letter,
                    "conf": conf,
                    "x": x_center,
                    "y": y_center,
                    "w": width,
                    "h": height,
                }
            )
            continue

        if cls_name in PROVINCE_MAP:
            province_candidates.append((conf, PROVINCE_MAP[cls_name]))

    def cluster_rows(cands, y_scale=0.8):
        if not cands:
            return []
        heights = np.array([c["h"] for c in cands], dtype=float)
        median_h = float(np.median(heights)) if len(heights) else 0.0
        y_threshold = max(10.0, median_h * y_scale)
        rows = []
        for c in sorted(cands, key=lambda x: x["y"]):
            placed = False
            for row in rows:
                row_y = float(np.mean([r["y"] for r in row]))
                if abs(c["y"] - row_y) <= y_threshold:
                    row.append(c)
                    placed = True
                    break
            if not placed:
                rows.append([c])
        return rows

    def select_main_row(cands):
        if not cands:
            return []

        heights = np.array([c["h"] for c in cands], dtype=float)
        median_h = float(np.median(heights)) if len(heights) else 0.0
        keep = [c for c in cands if c["h"] >= max(8.0, median_h * 0.55)]
        if not keep:
            keep = cands

        keep = sorted(keep, key=lambda c: c["y"])
        y_threshold = max(10.0, median_h * 0.6)
        clusters = []
        for c in keep:
            placed = False
            for cluster in clusters:
                cluster_y = float(np.mean([x["y"] for x in cluster]))
                if abs(c["y"] - cluster_y) <= y_threshold:
                    cluster.append(c)
                    placed = True
                    break
            if not placed:
                clusters.append([c])

        def cluster_score(cluster):
            conf_sum = sum(x["conf"] for x in cluster)
            h_avg = float(np.mean([x["h"] for x in cluster]))
            return conf_sum + (0.25 * len(cluster)) + (0.01 * h_avg)

        return max(clusters, key=cluster_score) if clusters else keep

    def dedupe_by_x(cands):
        if not cands:
            return []
        cands = sorted(cands, key=lambda c: c["x"])
        median_w = float(np.median([c["w"] for c in cands])) if cands else 18.0
        x_threshold = max(7.0, median_w * 0.45)

        slots = []
        for c in cands:
            if not slots:
                slots.append([c])
                continue
            last_slot_x = float(np.mean([x["x"] for x in slots[-1]]))
            if abs(c["x"] - last_slot_x) <= x_threshold:
                slots[-1].append(c)
            else:
                slots.append([c])

        collapsed = []
        for slot in slots:
            best = max(slot, key=lambda x: x["conf"])
            collapsed.append(best)
        return collapsed

    # Geometry from all detected alnum boxes helps infer plate style when no vehicle class exists.
    text_box_ratio = 0.0
    row_count = 0
    rows = []
    if alnum_candidates:
        min_x = min(c["x"] - (c["w"] / 2.0) for c in alnum_candidates)
        max_x = max(c["x"] + (c["w"] / 2.0) for c in alnum_candidates)
        min_y = min(c["y"] - (c["h"] / 2.0) for c in alnum_candidates)
        max_y = max(c["y"] + (c["h"] / 2.0) for c in alnum_candidates)
        text_w = max(1.0, max_x - min_x)
        text_h = max(1.0, max_y - min_y)
        text_box_ratio = float(text_w / text_h)
        rows = cluster_rows(alnum_candidates, y_scale=0.8)
        row_count = len(rows)

    letters = []
    digits = []
    used_tokens = []

    def normalize_moto_prefix(top_row_tokens):
        """Normalize motorcycle top row to common format: 1 digit + 2-3 Thai letters."""
        if not top_row_tokens:
            return ""

        leading_digit = ""
        thai_letters = []

        for idx, tok in enumerate(top_row_tokens):
            if tok["type"] == "digit" and idx == 0 and not leading_digit:
                leading_digit = tok["label"]
            elif tok["type"] == "letter":
                thai_letters.append(tok["label"])

        # Common confusion: leading '1' predicted as Thai glyph in motorcycle plates.
        # Example: 'รกง' should be '1กง' in some samples.
        if not leading_digit and len(thai_letters) >= 3 and thai_letters[0] in {"ร", "ว"}:
            leading_digit = "1"
            thai_letters = thai_letters[1:]

        if leading_digit:
            return f"{leading_digit}{''.join(thai_letters[:3])}"
        return "".join(thai_letters[:3])

    # Two-row plates (common in motorcycles): top row is prefix (can include leading digit),
    # bottom row is mostly numeric running number.
    if row_count >= 2:
        top_row = sorted(dedupe_by_x(rows[0]), key=lambda c: c["x"])
        bottom_row = sorted(dedupe_by_x(rows[-1]), key=lambda c: c["x"])
        used_tokens = top_row + bottom_row

        top_prefix = normalize_moto_prefix(top_row)
        letters = list(top_prefix)
        digits = [c["label"] for c in bottom_row if c["type"] == "digit"]

        if not digits:
            for row in rows[1:]:
                for c in sorted(dedupe_by_x(row), key=lambda x: x["x"]):
                    if c["type"] == "digit":
                        digits.append(c["label"])
    else:
        row_candidates = select_main_row(alnum_candidates)
        cleaned = dedupe_by_x(row_candidates)
        used_tokens = cleaned

        digit_started = False
        for item in cleaned:
            if item["type"] == "digit":
                digit_started = True
                digits.append(item["label"])
            elif not digit_started:
                letters.append(item["label"])

    # Caps for noise control
    letters = letters[:4]
    digits = digits[:4]

    if letters and digits:
        plate_number = f"{''.join(letters)} {''.join(digits)}"
    elif letters:
        plate_number = "".join(letters)
    elif digits:
        plate_number = "".join(digits)
    else:
        plate_number = "ไม่พบตัวอักษรบนป้าย"

    province = "-"
    province_conf = 0.0
    if province_candidates:
        province_conf, province = sorted(province_candidates, key=lambda x: x[0], reverse=True)[0]

    vehicle_code = "unknown"
    vehicle_conf = 0.0
    if vehicle_candidates:
        vehicle_conf, vehicle_code = sorted(vehicle_candidates, key=lambda x: x[0], reverse=True)[0]

    avg_conf = round(float(np.mean(confidence_values)) * 100, 2) if confidence_values else 0.0
    clean_conf = float(np.mean([x["conf"] for x in used_tokens])) if used_tokens else 0.0

    quality_score = 0.0
    quality_score += 3.0 * len(letters)
    quality_score += 2.2 * len(digits)
    quality_score += 4.5 * clean_conf
    if 1 <= len(letters) <= 3:
        quality_score += 2.0
    if 1 <= len(digits) <= 4:
        quality_score += 2.0
    if province != "-":
        quality_score += 0.8 + province_conf
    if vehicle_code != "unknown":
        quality_score += 0.6 + vehicle_conf

    return {
        "plate_number": plate_number,
        "letters": "".join(letters),
        "digits": "".join(digits),
        "province": province,
        "text_box_ratio": round(text_box_ratio, 4),
        "row_count": row_count,
        "vehicle_code": vehicle_code,
        "vehicle_conf": vehicle_conf,
        "vehicle_type": "ไม่สามารถระบุได้",
        "plate_color": "ไม่สามารถระบุได้",
        "usage": "ไม่สามารถระบุได้",
        "confidence": avg_conf,
        "province_conf": province_conf,
        "quality_score": round(quality_score, 4),
    }


def run_predict(image, conf=0.18, imgsz=1280):
    return MODEL.predict(
        source=image,
        device=DEVICE,
        conf=conf,
        iou=0.45,
        imgsz=imgsz,
        verbose=False,
        save=False,
    )[0]


def enhance_plate_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    enhanced = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


def build_fallback_views(image):
    h, w = image.shape[:2]
    x1, x2 = int(w * 0.18), int(w * 0.82)
    y1, y2 = int(h * 0.45), h
    lower_center = image[y1:y2, x1:x2]

    enhanced = enhance_plate_image(image)
    enhanced_crop = enhance_plate_image(lower_center) if lower_center.size > 0 else None

    views = [image, enhanced]
    if lower_center.size > 0:
        views.append(lower_center)
    if enhanced_crop is not None:
        views.append(enhanced_crop)

    views.append(cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    views.append(cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    if lower_center.size > 0:
        views.append(cv2.resize(lower_center, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    if enhanced_crop is not None:
        views.append(cv2.resize(enhanced_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    return views


def infer_vehicle_code_from_heuristic(parsed):
    # Heuristic only for OCR-only models that don't predict vehicle class.
    digits = parsed.get("digits", "")
    letters = parsed.get("letters", "")
    province = parsed.get("province", "-")
    ratio = float(parsed.get("text_box_ratio", 0.0) or 0.0)
    rows = int(parsed.get("row_count", 0) or 0)

    moto_score = 0.0
    car_score = 0.0

    if province != "-":
        moto_score += 0.4
        car_score += 0.3

    if len(digits) >= 4:
        moto_score += 1.0
    elif 1 <= len(digits) <= 3:
        car_score += 0.6

    if not letters and len(digits) >= 4 and province != "-":
        moto_score += 1.1
    elif letters:
        car_score += 0.7

    if rows >= 2:
        moto_score += 1.2
    else:
        car_score += 0.5

    # Motorcycle plates are often closer to square than car plates.
    if 0 < ratio < 2.0:
        moto_score += 1.0
    elif ratio >= 2.4:
        car_score += 1.0

    return "moto_private" if moto_score > car_score else "car_private"


def finalize_metadata(parsed):
    code = parsed.get("vehicle_code", "unknown")
    has_signal = parsed.get("province", "-") != "-" or parsed.get("letters") or parsed.get("digits")

    # If model explicitly predicts a vehicle class, use it directly.
    if code != "unknown" and code in VEHICLE_PROFILES:
        profile = VEHICLE_PROFILES[code]
        parsed["vehicle_type"] = profile["vehicle_type"]
        parsed["plate_color"] = profile["plate_color"]
        parsed["usage"] = profile["usage"]
        return parsed

    # For OCR-only models (no vehicle class), keep legacy behavior:
    # if plate/province signal exists, assume private car.
    if has_signal:
        inferred_code = infer_vehicle_code_from_heuristic(parsed)
        fallback = VEHICLE_PROFILES.get(inferred_code, VEHICLE_PROFILES.get("car_private", VEHICLE_PROFILES["unknown"]))
        parsed["vehicle_type"] = fallback["vehicle_type"]
        parsed["plate_color"] = fallback["plate_color"]
        parsed["usage"] = fallback["usage"]
    else:
        unknown = VEHICLE_PROFILES["unknown"]
        parsed["vehicle_type"] = unknown["vehicle_type"]
        parsed["plate_color"] = unknown["plate_color"]
        parsed["usage"] = unknown["usage"]
    return parsed


def predict_best(image):
    candidate_results = []
    for view in build_fallback_views(image):
        for conf, imgsz in ((0.18, 1280), (0.12, 1600)):
            result = run_predict(view, conf=conf, imgsz=imgsz)
            parsed = parse_plate_result(result)
            if parsed["letters"] or parsed["digits"] or parsed["vehicle_code"] != "unknown":
                candidate_results.append(parsed)

    if not candidate_results:
        return finalize_metadata(parse_plate_result(run_predict(image, conf=0.1, imgsz=1600)))

    plate_votes = {}
    for c in candidate_results:
        key = f"{c['letters']}|{c['digits']}"
        if key not in plate_votes:
            plate_votes[key] = {"score": 0.0, "best": c}
        plate_votes[key]["score"] += c["quality_score"]
        if c["quality_score"] > plate_votes[key]["best"]["quality_score"]:
            plate_votes[key]["best"] = c

    best_key = max(plate_votes.items(), key=lambda item: item[1]["score"])[0]
    chosen = dict(plate_votes[best_key]["best"])

    province_votes = defaultdict(float)
    for c in candidate_results:
        if c["province"] != "-":
            weight = c["quality_score"] + c["province_conf"]
            province_votes[c["province"]] += weight
    if province_votes:
        chosen["province"] = max(province_votes.items(), key=lambda item: item[1])[0]

    vehicle_votes = defaultdict(float)
    for c in candidate_results:
        if c["vehicle_code"] != "unknown":
            weight = c["quality_score"] + c["vehicle_conf"]
            vehicle_votes[c["vehicle_code"]] += weight
    if vehicle_votes:
        chosen["vehicle_code"] = max(vehicle_votes.items(), key=lambda item: item[1])[0]

    confs = [c["confidence"] for c in candidate_results]
    chosen["confidence"] = round(float(np.mean(confs)), 2) if confs else chosen["confidence"]
    return finalize_metadata(chosen)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/detect")
def detect_plate():
    if "image" not in request.files:
        return jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "ไม่พบไฟล์รูปภาพ"}), 400

    try:
        image = decode_image(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    parsed = predict_best(image)
    for key in (
        "letters",
        "digits",
        "province_conf",
        "quality_score",
        "vehicle_code",
        "vehicle_conf",
        "text_box_ratio",
        "row_count",
    ):
        parsed.pop(key, None)
    return jsonify(parsed)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

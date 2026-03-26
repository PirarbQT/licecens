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

# ระบุตำแหน่งโฟลเดอร์หลักของโปรเจกต์ และไฟล์สำคัญที่ backend ต้องใช้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "HurricaneOD_beta.pt")
VEHICLE_PROFILE_PATH = os.path.join(BASE_DIR, "vehicle_profiles.yaml")

app = Flask(__name__)


# โหลด mapping ของตัวอักษรไทยและจังหวัดจากไฟล์ CSV
# เพื่อแปลง class code จากโมเดลให้เป็นข้อความจริงที่หน้าเว็บใช้แสดงผล
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


# โหลด profile ของประเภทรถจาก YAML เพื่อใช้เติมคำอธิบายที่หน้าเว็บต้องการ
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
        "car_public": {
            "vehicle_type": "รถยนต์สาธารณะ",
            "plate_color": "เหลือง (สาธารณะ)",
            "usage": "สาธารณะ",
            "aliases": ["PUBLIC_CAR", "TAXI", "SERVICE_CAR", "COMMERCIAL_CAR"],
        },
        "car_auction": {
            "vehicle_type": "รถยนต์ส่วนบุคคล",
            "plate_color": "ป้ายประมูล",
            "usage": "ส่วนบุคคล",
            "aliases": ["AUCTION", "AUCTION_CAR", "RED_PLATE"],
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
    for key, fallback in default_profiles.items():
        value = profiles.get(key, {})
        normalized[key] = {
            "vehicle_type": value.get("vehicle_type", fallback["vehicle_type"]),
            "plate_color": value.get("plate_color", fallback["plate_color"]),
            "usage": value.get("usage", fallback["usage"]),
            "aliases": [a.upper() for a in value.get("aliases", fallback["aliases"])],
        }

    for key, value in profiles.items():
        if key not in normalized:
            normalized[key] = {
                "vehicle_type": value.get("vehicle_type", "ไม่สามารถระบุได้"),
                "plate_color": value.get("plate_color", "ไม่สามารถระบุได้"),
                "usage": value.get("usage", "ไม่สามารถระบุได้"),
                "aliases": [a.upper() for a in value.get("aliases", [])],
            }

    return normalized


LABEL_MAP, PROVINCE_MAP = load_label_maps()
VEHICLE_PROFILES = load_vehicle_profiles()
VEHICLE_ALIAS_TO_CODE = {}
for code, profile in VEHICLE_PROFILES.items():
    for alias in profile.get("aliases", []):
        VEHICLE_ALIAS_TO_CODE[alias.upper()] = code

# ถ้ามี GPU จะใช้ GPU อัตโนมัติ ถ้าไม่มีจะ fallback ไป CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL = YOLO(MODEL_PATH)
DETECTOR_MODEL = YOLO(DETECTOR_MODEL_PATH) if os.path.exists(DETECTOR_MODEL_PATH) else None


# แปลงชื่อ class จากโมเดลเป็นตัวอักษรไทยหรือชื่อจังหวัด
def translate_label(cls_name: str) -> str:
    return LABEL_MAP.get(cls_name, cls_name)


# แปลงไฟล์รูปจากหน้าเว็บให้เป็นภาพ OpenCV
def decode_image(file_storage):
    image_bytes = file_storage.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปได้")
    return image


# จำกัดพิกัดกรอบให้อยู่ในภาพจริงเสมอ
def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return x1, y1, x2, y2


# ครอปภาพตามกรอบที่คำนวณได้
def crop_image(image, bbox):
    if image is None or bbox is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


# เรียงจุด 4 มุมให้อยู่ในลำดับ ซ้ายบน ขวาบน ขวาล่าง ซ้ายล่าง
def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


# พยายามหา contour ของป้ายใน crop แล้ว warp ให้ป้ายตรง
def rectify_plate_crop(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, False

    h, w = plate_crop.shape[:2]
    if h < 24 or w < 48:
        return plate_crop, False

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, False

    img_area = float(h * w)
    best_quad = None
    best_score = -1.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < img_area * 0.12:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            quad = box

        ordered = order_quad_points(quad)
        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])
        warp_w = max(width_top, width_bottom)
        warp_h = max(height_left, height_right)
        if warp_w < 40 or warp_h < 16:
            continue

        ratio = warp_w / max(1.0, warp_h)
        if ratio < 1.4 or ratio > 8.5:
            continue

        area_ratio = area / img_area
        rectangularity = area / max(1.0, warp_w * warp_h)
        score = area_ratio + rectangularity + min(1.0, ratio / 6.0)
        if score > best_score:
            best_score = score
            best_quad = ordered

    if best_quad is None:
        return plate_crop, False

    width_a = np.linalg.norm(best_quad[2] - best_quad[3])
    width_b = np.linalg.norm(best_quad[1] - best_quad[0])
    height_a = np.linalg.norm(best_quad[1] - best_quad[2])
    height_b = np.linalg.norm(best_quad[0] - best_quad[3])
    max_width = int(round(max(width_a, width_b)))
    max_height = int(round(max(height_a, height_b)))

    if max_width < 40 or max_height < 16:
        return plate_crop, False

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(best_quad, destination)
    warped = cv2.warpPerspective(
        plate_crop,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if warped is None or warped.size == 0:
        return plate_crop, False
    return warped, True


# คำนวณความเอียงของข้อความจากตำแหน่งกล่องตัวอักษร
def estimate_text_angle(tokens):
    if len(tokens) < 2:
        return 0.0

    xs = np.array([t["x"] for t in tokens], dtype=np.float32)
    ys = np.array([t["y"] for t in tokens], dtype=np.float32)
    if float(xs.max() - xs.min()) < 12.0:
        return 0.0

    slope, _ = np.polyfit(xs, ys, deg=1)
    angle = float(np.degrees(np.arctan(slope)))
    return max(-18.0, min(18.0, angle))


# สร้างกรอบครอบป้ายจากกล่องตัวอักษรและจังหวัดที่ตรวจเจอ
def estimate_plate_bbox(tokens, province_boxes):
    if not tokens:
        return None

    left = min(t["x1"] for t in tokens)
    top = min(t["y1"] for t in tokens)
    right = max(t["x2"] for t in tokens)
    bottom = max(t["y2"] for t in tokens)

    token_widths = [max(1.0, t["x2"] - t["x1"]) for t in tokens]
    token_heights = [max(1.0, t["y2"] - t["y1"]) for t in tokens]
    median_w = float(np.median(token_widths)) if token_widths else 18.0
    median_h = float(np.median(token_heights)) if token_heights else 18.0

    for province in province_boxes:
        vertical_gap = province["y1"] - bottom
        horizontal_overlap = min(right, province["x2"]) - max(left, province["x1"])
        if vertical_gap <= median_h * 2.2 and horizontal_overlap >= -(median_w * 1.5):
            left = min(left, province["x1"])
            top = min(top, province["y1"])
            right = max(right, province["x2"])
            bottom = max(bottom, province["y2"])

    text_w = max(1.0, right - left)
    text_h = max(1.0, bottom - top)
    pad_x = max(median_w * 1.4, text_w * 0.20)
    pad_top = max(median_h * 1.0, text_h * 0.22)
    pad_bottom = max(median_h * 1.8, text_h * 0.35)

    return (
        left - pad_x,
        top - pad_top,
        right + pad_x,
        bottom + pad_bottom,
    )


# หมุนภาพครอปเล็กน้อยเพื่อแก้กรณีป้ายเอียง
def rotate_crop(image, angle):
    if image is None or abs(angle) < 1.2:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ปรับภาพให้คมและบาลานซ์แสงมากขึ้น เพื่อช่วย OCR ในรูปยาก
def enhance_plate_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    y = clahe.apply(y)
    enhanced = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


# วิเคราะห์สีพื้นหลังป้ายจาก crop ที่ครอบป้ายมาแล้ว
def analyze_plate_background(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return {"bg_hint": "unknown", "bg_conf": 0.0, "bg_white_ratio": 0.0, "bg_yellow_ratio": 0.0, "bg_red_ratio": 0.0}

    h, w = plate_crop.shape[:2]
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    inner = plate_crop[margin_y : max(margin_y + 1, h - margin_y), margin_x : max(margin_x + 1, w - margin_x)]
    if inner.size == 0:
        inner = plate_crop

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    valid = val > 55
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {"bg_hint": "unknown", "bg_conf": 0.0, "bg_white_ratio": 0.0, "bg_yellow_ratio": 0.0, "bg_red_ratio": 0.0}

    white_mask = valid & (sat < 55) & (val > 120)
    yellow_mask = valid & (hue >= 15) & (hue <= 42) & (sat > 70) & (val > 85)
    red_mask = valid & (((hue <= 10) | (hue >= 170)) & (sat > 75) & (val > 75))

    white_ratio = float(np.count_nonzero(white_mask) / valid_count)
    yellow_ratio = float(np.count_nonzero(yellow_mask) / valid_count)
    red_ratio = float(np.count_nonzero(red_mask) / valid_count)

    if red_ratio >= 0.16 and red_ratio > yellow_ratio * 1.1:
        return {
            "bg_hint": "car_auction",
            "bg_conf": round(min(1.0, red_ratio * 2.4), 4),
            "bg_white_ratio": round(white_ratio, 4),
            "bg_yellow_ratio": round(yellow_ratio, 4),
            "bg_red_ratio": round(red_ratio, 4),
        }

    if yellow_ratio >= 0.18 and yellow_ratio > white_ratio * 0.45:
        return {
            "bg_hint": "car_public",
            "bg_conf": round(min(1.0, yellow_ratio * 2.0), 4),
            "bg_white_ratio": round(white_ratio, 4),
            "bg_yellow_ratio": round(yellow_ratio, 4),
            "bg_red_ratio": round(red_ratio, 4),
        }

    if white_ratio >= 0.30:
        return {
            "bg_hint": "white_plate",
            "bg_conf": round(min(1.0, white_ratio * 1.6), 4),
            "bg_white_ratio": round(white_ratio, 4),
            "bg_yellow_ratio": round(yellow_ratio, 4),
            "bg_red_ratio": round(red_ratio, 4),
        }

    return {
        "bg_hint": "unknown",
        "bg_conf": 0.0,
        "bg_white_ratio": round(white_ratio, 4),
        "bg_yellow_ratio": round(yellow_ratio, 4),
        "bg_red_ratio": round(red_ratio, 4),
    }


# ให้คะแนนรูปแบบทะเบียนไทย เพื่อช่วยเลือกผลที่สมเหตุสมผลกว่า
def score_plate_pattern(letters, digits, row_count, text_box_ratio):
    score = 0.0

    if row_count >= 2:
        if digits:
            score += 2.2
        if 1 <= len(letters) <= 4:
            score += 1.6
        if text_box_ratio < 2.2:
            score += 1.4
    else:
        if 1 <= len(letters) <= 3:
            score += 2.0
        if 1 <= len(digits) <= 4:
            score += 2.0
        if letters and digits:
            score += 1.6
        if text_box_ratio >= 2.1:
            score += 1.0

    return score


# แปลงผลตรวจจับของ YOLO ให้เป็นทะเบียน จังหวัด และข้อมูล geometry ที่ใช้ต่อใน post-process
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
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        confidence_values.append(conf)

        box_info = {
            "label": cls_name,
            "conf": conf,
            "x": x_center,
            "y": y_center,
            "w": x2 - x1,
            "h": y2 - y1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        if cls_upper in VEHICLE_ALIAS_TO_CODE:
            vehicle_candidates.append((conf, VEHICLE_ALIAS_TO_CODE[cls_upper]))
            continue

        if cls_name.isdigit():
            alnum_candidates.append({"type": "digit", **box_info})
            continue

        if cls_name.startswith("A"):
            thai_letter = translate_label(cls_name)
            alnum_candidates.append({"type": "letter", **box_info, "label": thai_letter})
            continue

        if cls_name in PROVINCE_MAP:
            province_candidates.append({"conf": conf, "province": PROVINCE_MAP[cls_name], **box_info})

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

        return [max(slot, key=lambda x: x["conf"]) for slot in slots]

    # ใช้รูปทรงของกล่องตัวอักษรทั้งหมดช่วยเดาว่าป้ายเป็นทรงรถยนต์หรือมอเตอร์ไซค์
    text_box_ratio = 0.0
    row_count = 0
    rows = []
    if alnum_candidates:
        min_x = min(c["x1"] for c in alnum_candidates)
        max_x = max(c["x2"] for c in alnum_candidates)
        min_y = min(c["y1"] for c in alnum_candidates)
        max_y = max(c["y2"] for c in alnum_candidates)
        text_w = max(1.0, max_x - min_x)
        text_h = max(1.0, max_y - min_y)
        text_box_ratio = float(text_w / text_h)
        rows = cluster_rows(alnum_candidates, y_scale=0.8)
        row_count = len(rows)

    letters = []
    digits = []
    used_tokens = []

    def normalize_moto_prefix(top_row_tokens):
        """ปรับ prefix ป้ายมอเตอร์ไซค์ให้เข้าใกล้รูปแบบที่ใช้จริง"""
        if not top_row_tokens:
            return ""

        leading_digit = ""
        thai_letters = []

        for idx, tok in enumerate(top_row_tokens):
            if tok["type"] == "digit" and idx == 0 and not leading_digit:
                leading_digit = tok["label"]
            elif tok["type"] == "letter":
                thai_letters.append(tok["label"])

        if not leading_digit and len(thai_letters) >= 3 and thai_letters[0] in {"ร", "ว"}:
            leading_digit = "1"
            thai_letters = thai_letters[1:]

        if leading_digit:
            return f"{leading_digit}{''.join(thai_letters[:3])}"
        return "".join(thai_letters[:3])

    # ถ้ามี 2 แถว ให้มองเป็นป้ายมอเตอร์ไซค์ก่อน
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

    letters = letters[:4]
    digits = digits[:4]

    if letters and digits:
        plate_number = f"{''.join(letters)} {''.join(digits)}"
    elif letters:
        plate_number = "".join(letters)
    elif digits:
        plate_number = "".join(digits)
    else:
        plate_number = "-"

    province = "-"
    province_conf = 0.0
    province_boxes = []
    if province_candidates:
        sorted_provinces = sorted(province_candidates, key=lambda x: x["conf"], reverse=True)
        province_conf = float(sorted_provinces[0]["conf"])
        province = sorted_provinces[0]["province"]
        province_boxes = sorted_provinces[:2]

    vehicle_code = "unknown"
    vehicle_conf = 0.0
    if vehicle_candidates:
        vehicle_conf, vehicle_code = sorted(vehicle_candidates, key=lambda x: x[0], reverse=True)[0]

    avg_conf = round(float(np.mean(confidence_values)) * 100, 2) if confidence_values else 0.0
    clean_conf = float(np.mean([x["conf"] for x in used_tokens])) if used_tokens else 0.0
    plate_bbox = estimate_plate_bbox(used_tokens or alnum_candidates, province_boxes)
    text_angle = estimate_text_angle(used_tokens) if used_tokens else 0.0

    pattern_score = score_plate_pattern("".join(letters), "".join(digits), row_count, text_box_ratio)

    quality_score = 0.0
    quality_score += 3.0 * len(letters)
    quality_score += 2.4 * len(digits)
    quality_score += 4.8 * clean_conf
    quality_score += pattern_score
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
        "plate_bbox": plate_bbox,
        "text_angle": round(float(text_angle), 4),
        "plate_crop_ratio": 0.0,
        "bg_hint": "unknown",
        "bg_conf": 0.0,
        "bg_white_ratio": 0.0,
        "bg_yellow_ratio": 0.0,
        "bg_red_ratio": 0.0,
        "plate_crop": None,
    }


# รันโมเดล YOLO ด้วยค่า conf/imgsz ที่กำหนด
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


# รันโมเดล detector สำหรับหาตำแหน่งป้ายทะเบียนทั้งแผ่นก่อนส่งเข้า OCR
def run_plate_detector(image, conf=0.2, imgsz=1280):
    if DETECTOR_MODEL is None:
        return None

    return DETECTOR_MODEL.predict(
        source=image,
        device=DEVICE,
        conf=conf,
        iou=0.45,
        imgsz=imgsz,
        verbose=False,
        save=False,
    )[0]


# สร้างมุมมองหลายแบบจากภาพเต็ม เพื่อให้รอบแรกหาป้ายได้ดีขึ้น
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


# ใช้ detector ใหม่หา plate crop ที่น่าจะใช่ที่สุดจากภาพเต็ม
def detect_plate_crops(image):
    if DETECTOR_MODEL is None or image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    detector_views = [image, enhance_plate_image(image)]
    raw_hits = []

    for detector_view in detector_views:
        for conf, imgsz in ((0.18, 1280), (0.12, 1600)):
            result = run_plate_detector(detector_view, conf=conf, imgsz=imgsz)
            if result is None:
                continue
            for box in result.boxes:
                score = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                ratio = bw / bh

                # ป้ายรถไทยจริงมักไม่ผอมจนเกินไป และไม่เป็นกรอบใหญ่เกินภาพทั้งภาพ
                if ratio < 1.0 or ratio > 8.5:
                    continue
                if bw * bh > (w * h * 0.85):
                    continue

                pad_x = max(12.0, bw * 0.14)
                pad_y = max(10.0, bh * 0.30)
                bbox = clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)
                raw_hits.append({"bbox": bbox, "score": score, "ratio": ratio})

    if not raw_hits:
        return []

    deduped = []
    for hit in sorted(raw_hits, key=lambda item: item["score"], reverse=True):
        x1, y1, x2, y2 = hit["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        keep = True
        for chosen in deduped:
            a1, b1, a2, b2 = chosen["bbox"]
            ccx = (a1 + a2) / 2.0
            ccy = (b1 + b2) / 2.0
            if abs(cx - ccx) < 40 and abs(cy - ccy) < 30:
                keep = False
                break
        if keep:
            deduped.append(hit)
        if len(deduped) >= 4:
            break

    crops = []
    for hit in deduped:
        crop = crop_image(image, hit["bbox"])
        if crop is None or crop.size == 0:
            continue
        rectified_crop, rectified = rectify_plate_crop(crop)
        crops.append(
            {
                "crop": rectified_crop if rectified else crop,
                "raw_crop": crop,
                "bbox": hit["bbox"],
                "score": hit["score"],
                "rectified": rectified,
            }
        )

    return crops


# สร้างมุมมองแบบ zoom-in จากบริเวณป้าย เพื่อให้ OCR รอบสองแม่นขึ้น
def build_focus_views(plate_crop, text_angle):
    if plate_crop is None or plate_crop.size == 0:
        return []

    views = [plate_crop]
    enhanced = enhance_plate_image(plate_crop)
    views.append(enhanced)

    corrected = rotate_crop(plate_crop, -text_angle)
    if corrected is not None and corrected.size:
        views.append(corrected)
        views.append(enhance_plate_image(corrected))

    for base in list(views):
        h, w = base.shape[:2]
        if min(h, w) < 700:
            views.append(cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))

    unique = []
    seen = set()
    for view in views:
        if view is None or view.size == 0:
            continue
        key = (view.shape[0], view.shape[1], int(np.mean(view)))
        if key not in seen:
            unique.append(view)
            seen.add(key)
    return unique


# เติมข้อมูล crop ป้ายและสีพื้นหลังให้ candidate แต่ละตัว
def enrich_candidate(parsed, source_image):
    plate_crop = crop_image(source_image, parsed.get("plate_bbox"))
    parsed["plate_crop"] = plate_crop
    if plate_crop is not None and plate_crop.size:
        rectified_crop, rectified = rectify_plate_crop(plate_crop)
        if rectified:
            plate_crop = rectified_crop
            parsed["plate_crop"] = rectified_crop
        parsed["plate_crop_ratio"] = round(float(plate_crop.shape[1] / max(1, plate_crop.shape[0])), 4)
        parsed.update(analyze_plate_background(plate_crop))
        parsed["quality_score"] = round(parsed["quality_score"] + (parsed["bg_conf"] * 2.2), 4)
    return parsed


# ดึง candidate จากภาพหนึ่งมุมมอง โดยรันหลายค่า threshold แล้วเก็บผลที่พอมีสัญญาณ
def collect_candidates_from_view(view, quality_bonus=0.0):
    candidates = []
    for conf, imgsz in ((0.18, 1280), (0.12, 1600)):
        result = run_predict(view, conf=conf, imgsz=imgsz)
        parsed = parse_plate_result(result)
        if parsed["letters"] or parsed["digits"] or parsed["vehicle_code"] != "unknown":
            enriched = enrich_candidate(parsed, view)
            if quality_bonus:
                char_count = len(enriched.get("letters", "")) + len(enriched.get("digits", ""))
                bonus_scale = 1.0
                if char_count <= 2:
                    bonus_scale = 0.0
                elif char_count == 3:
                    bonus_scale = 0.25
                elif char_count == 4:
                    bonus_scale = 0.55
                enriched["quality_score"] = round(enriched["quality_score"] + (quality_bonus * bonus_scale), 4)
            candidates.append(enriched)
    return candidates


# เดาประเภทรถจากสีป้าย ทรงป้าย จำนวนแถว และสัดส่วน crop จริง
def infer_vehicle_code(parsed):
    explicit_code = parsed.get("vehicle_code", "unknown")
    if explicit_code != "unknown" and explicit_code in VEHICLE_PROFILES:
        return explicit_code

    bg_hint = parsed.get("bg_hint", "unknown")
    if bg_hint in {"car_public", "car_auction"}:
        return bg_hint

    row_count = int(parsed.get("row_count", 0) or 0)
    text_box_ratio = float(parsed.get("text_box_ratio", 0.0) or 0.0)
    plate_crop_ratio = float(parsed.get("plate_crop_ratio", 0.0) or 0.0)
    letters = parsed.get("letters", "")
    digits = parsed.get("digits", "")

    # กรณีที่ลักษณะป้ายเป็นรถยนต์ชัดมาก ให้ตัดสินเป็นรถยนต์ทันที
    if row_count <= 1 and text_box_ratio >= 2.6 and letters:
        return "car_private"
    if row_count <= 1 and text_box_ratio >= 3.2 and len(digits) >= 3:
        return "car_private"

    moto_score = 0.0
    car_score = 0.0

    if row_count >= 2:
        moto_score += 3.2
    else:
        car_score += 1.6

    if 0 < plate_crop_ratio < 1.05:
        moto_score += 2.0
    elif 1.05 <= plate_crop_ratio < 1.35:
        moto_score += 0.5
        car_score += 0.3
    elif plate_crop_ratio >= 1.35:
        car_score += 0.8

    if 0 < text_box_ratio < 1.8:
        moto_score += 2.2
    elif 1.8 <= text_box_ratio < 2.4:
        moto_score += 0.8
        car_score += 0.4
    elif text_box_ratio >= 2.4:
        car_score += 2.4

    if len(digits) >= 4:
        car_score += 0.5
        moto_score += 0.3
    elif 1 <= len(digits) <= 3:
        car_score += 0.7

    if not letters and len(digits) >= 4:
        moto_score += 0.4
        car_score += 0.2
    elif letters:
        car_score += 1.0

    if bg_hint == "white_plate":
        if row_count >= 2 and text_box_ratio < 2.2:
            moto_score += 0.8
        else:
            car_score += 0.9

    return "moto_private" if moto_score > car_score else "car_private"


# เติมข้อมูล vehicle_type / plate_color / usage ให้ครบก่อนส่งกลับหน้าเว็บ
def finalize_metadata(parsed):
    code = infer_vehicle_code(parsed)
    profile = VEHICLE_PROFILES.get(code, VEHICLE_PROFILES["unknown"])
    parsed["vehicle_code"] = code
    parsed["vehicle_type"] = profile["vehicle_type"]
    parsed["plate_color"] = profile["plate_color"]
    parsed["usage"] = profile["usage"]
    return parsed


# ยิงโมเดลบนภาพเต็มก่อน แล้วเอา crop ป้ายที่เจอไปวิ่ง OCR รอบสองแบบ zoom-in
def predict_best(image):
    candidate_results = []
    detected_plate_crops = detect_plate_crops(image)

    # ถ้า detector ใหม่เจอป้าย ให้เอาครอปนั้นไปเข้า OCR ก่อน เพราะแม่นกว่าการอ่านทั้งภาพ
    for plate_item in detected_plate_crops:
        for focus_view in build_focus_views(plate_item["crop"], 0.0):
            candidate_results.extend(
                collect_candidates_from_view(
                    focus_view,
                    quality_bonus=2.4 + min(1.2, plate_item["score"]),
                )
            )

    # ยังเก็บการวิ่งบนภาพเต็มไว้เป็น fallback สำหรับเคส detector พลาด
    for view in build_fallback_views(image):
        candidate_results.extend(collect_candidates_from_view(view))

    if not candidate_results:
        parsed = parse_plate_result(run_predict(image, conf=0.1, imgsz=1600))
        parsed = enrich_candidate(parsed, image)
        return finalize_metadata(parsed)

    seed_candidates = sorted(candidate_results, key=lambda c: c["quality_score"], reverse=True)[:4]
    focus_candidates = []
    for seed in seed_candidates:
        for focus_view in build_focus_views(seed.get("plate_crop"), seed.get("text_angle", 0.0)):
            focus_candidates.extend(collect_candidates_from_view(focus_view))

    all_candidates = candidate_results + focus_candidates

    plate_votes = {}
    for c in all_candidates:
        key = f"{c['letters']}|{c['digits']}"
        if key not in plate_votes:
            plate_votes[key] = {"score": 0.0, "best": c}
        plate_votes[key]["score"] += c["quality_score"]
        if c["quality_score"] > plate_votes[key]["best"]["quality_score"]:
            plate_votes[key]["best"] = c

    best_key = max(plate_votes.items(), key=lambda item: item[1]["score"])[0]
    chosen = dict(plate_votes[best_key]["best"])

    province_votes = defaultdict(float)
    for c in all_candidates:
        if c["province"] != "-":
            province_votes[c["province"]] += c["quality_score"] + c["province_conf"]
    if province_votes:
        chosen["province"] = max(province_votes.items(), key=lambda item: item[1])[0]

    vehicle_votes = defaultdict(float)
    for c in all_candidates:
        inferred_code = infer_vehicle_code(c)
        if inferred_code != "unknown":
            bonus = c["quality_score"] + c.get("bg_conf", 0.0)
            vehicle_votes[inferred_code] += bonus
    if vehicle_votes:
        chosen["vehicle_code"] = max(vehicle_votes.items(), key=lambda item: item[1])[0]

    confs = [c["confidence"] for c in all_candidates if c["letters"] or c["digits"]]
    chosen["confidence"] = round(float(np.mean(confs)), 2) if confs else chosen["confidence"]
    return finalize_metadata(chosen)


@app.get("/")
def home():
    # เปิดหน้าเว็บหลัก
    return render_template("index.html")


@app.post("/api/detect")
def detect_plate():
    # รับรูปจากหน้าเว็บแล้วส่งเข้าโมเดล YOLO OCR ในเครื่อง
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
        "plate_bbox",
        "text_angle",
        "plate_crop_ratio",
        "bg_hint",
        "bg_conf",
        "bg_white_ratio",
        "bg_yellow_ratio",
        "bg_red_ratio",
        "plate_crop",
    ):
        parsed.pop(key, None)
    return jsonify(parsed)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

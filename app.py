import csv
import io
import json
import os
import re
from difflib import get_close_matches

import yaml
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

# ระบุตำแหน่งโฟลเดอร์หลักของโปรเจกต์ และตำแหน่งไฟล์ข้อมูลที่ต้องใช้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROVINCE_CSV_PATH = os.path.join(BASE_DIR, "province_map.csv")
VEHICLE_PROFILE_PATH = os.path.join(BASE_DIR, "vehicle_profiles.yaml")


# โหลดค่าจากไฟล์ .env แบบง่าย ๆ ในรูป KEY=VALUE
# ใช้สำหรับอ่าน API key และค่าตั้งต้นอื่น ๆ โดยไม่ต้องตั้งผ่าน shell ทุกครั้ง
def load_env_file(path):
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


# พยายามอ่านค่า config จาก .env และ .env.local ก่อนเริ่มต้นระบบ
for env_filename in (".env", ".env.local"):
    load_env_file(os.path.join(BASE_DIR, env_filename))

# ค่าตั้งต้นหลักของระบบ OCR ที่ใช้ Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
MAX_IMAGE_DIMENSION = 2200
MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_VEHICLE_CODES = ("car_private", "car_public", "car_auction", "moto_private", "unknown")

app = Flask(__name__)


# โหลดรายชื่อจังหวัดจากไฟล์ CSV เพื่อใช้บังคับผลลัพธ์ของ Gemini
def load_provinces():
    provinces = []
    if not os.path.exists(PROVINCE_CSV_PATH):
        return provinces

    seen = set()
    with open(PROVINCE_CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            province = row["province"].strip()
            if province and province not in seen:
                provinces.append(province)
                seen.add(province)
    return provinces


# โหลดข้อมูล profile ของรถ/ป้ายจาก YAML
# ถ้าไฟล์ไม่มีหรือข้อมูลไม่ครบ จะ fallback ไปใช้ค่ามาตรฐานด้านล่าง
def load_vehicle_profiles():
    default_profiles = {
        "unknown": {
            "vehicle_type": "ไม่สามารถระบุได้",
            "plate_color": "ไม่สามารถระบุได้",
            "usage": "ไม่สามารถระบุได้",
        },
        "car_private": {
            "vehicle_type": "รถยนต์ส่วนบุคคล",
            "plate_color": "ขาว (ส่วนบุคคล)",
            "usage": "ส่วนบุคคล",
        },
        "car_public": {
            "vehicle_type": "รถยนต์สาธารณะ",
            "plate_color": "เหลือง (สาธารณะ)",
            "usage": "สาธารณะ",
        },
        "car_auction": {
            "vehicle_type": "รถยนต์ส่วนบุคคล",
            "plate_color": "ป้ายประมูล",
            "usage": "ส่วนบุคคล",
        },
        "moto_private": {
            "vehicle_type": "รถจักรยานยนต์ส่วนบุคคล",
            "plate_color": "ขาว (รถจักรยานยนต์)",
            "usage": "ส่วนบุคคล",
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
    for code in SUPPORTED_VEHICLE_CODES:
        value = profiles.get(code, {})
        fallback = default_profiles[code]
        normalized[code] = {
            "vehicle_type": value.get("vehicle_type", fallback["vehicle_type"]),
            "plate_color": value.get("plate_color", fallback["plate_color"]),
            "usage": value.get("usage", fallback["usage"]),
        }

    return normalized


# โหลดข้อมูลที่ระบบจะใช้ตลอด runtime ตั้งแต่ตอนเริ่มโปรแกรม
PROVINCES = load_provinces()
VALID_PROVINCES = set(PROVINCES)
VEHICLE_PROFILES = load_vehicle_profiles()

# กำหนด schema ของ JSON ที่อยากได้จาก Gemini
# ช่วยให้ผลลัพธ์มีโครงสร้างแน่นอนและ parse ต่อได้ง่าย
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_number": {
            "type": "string",
            "description": "Thai license plate text only. Return '-' if unreadable.",
        },
        "province": {
            "type": "string",
            "description": "Thai province shown on the plate. Return '-' if not visible.",
            "enum": ["-"] + PROVINCES if PROVINCES else ["-"],
        },
        "vehicle_code": {
            "type": "string",
            "description": "Best guess of plate type.",
            "enum": list(SUPPORTED_VEHICLE_CODES),
        },
        "confidence": {
            "type": "number",
            "description": "Overall confidence from 0 to 100.",
            "minimum": 0,
            "maximum": 100,
        },
    },
    "required": ["plate_number", "province", "vehicle_code", "confidence"],
}


# สร้าง client สำหรับเรียก Gemini API
# ถ้าไม่มี API key จะหยุดทันทีพร้อมข้อความที่ชัดเจน
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("กรุณาตั้งค่า GEMINI_API_KEY ก่อนใช้งาน")
    return genai.Client(api_key=api_key)


# แปลงไฟล์อัปโหลดให้อยู่ในรูปแบบที่เหมาะกับการส่งเข้า Gemini
# ขั้นตอนคือเปิดภาพ, แปลงเป็น RGB, ย่อขนาดถ้าจำเป็น, แล้วบันทึกกลับเป็น JPEG
def prepare_image(file_storage):
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("ไม่พบข้อมูลรูปภาพ")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")

            # ถ้ารูปใหญ่เกินไปจะย่อก่อน เพื่อลดขนาด request และเวลาเรียก API
            if max(image.size) > MAX_IMAGE_DIMENSION:
                image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            normalized_bytes = output.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปได้") from exc

    # กันกรณีไฟล์ยังใหญ่เกินเพดานหลังแปลงแล้ว
    if len(normalized_bytes) > MAX_INLINE_IMAGE_BYTES:
        raise ValueError("ไฟล์รูปใหญ่เกินไปสำหรับการส่งเข้า Gemini API")

    return normalized_bytes, "image/jpeg"


# สร้าง prompt ที่บอก Gemini ว่าต้องอ่านอะไรและต้องตอบกลับแบบไหน
# จุดสำคัญคือบังคับให้ตอบเป็นข้อมูลจริงจากภาพ ไม่ใช่คำอธิบายยาว ๆ
def build_prompt():
    province_hint = ", ".join(PROVINCES) if PROVINCES else "-"
    return f"""
คุณเป็นระบบ OCR สำหรับป้ายทะเบียนไทยจากภาพถ่าย

ให้อ่านเฉพาะข้อมูลที่มองเห็นจากภาพจริงเท่านั้น และคืนค่าตาม JSON schema เท่านั้น

กติกา:
- ถ้าอ่านเลขทะเบียนไม่ชัด ให้คืน plate_number เป็น "-"
- ถ้าอ่านจังหวัดไม่ชัด ให้คืน province เป็น "-"
- plate_number ต้องมีเฉพาะอักษรไทย ตัวเลข และช่องว่างเดียวระหว่าง prefix กับเลข ถ้ามีทั้งสองส่วน
- ป้ายมอเตอร์ไซค์ที่มี 2 บรรทัด ให้รวมเป็นบรรทัดเดียวในรูป "<แถวบน> <แถวล่าง>"
- province ต้องเลือกจากรายการนี้เท่านั้น: {province_hint}
- vehicle_code ต้องเป็น car_private, car_public, car_auction, moto_private หรือ unknown
- ถ้าเห็นป้ายทะเบียนพื้นเหลือง ตัวอักษรเข้ม และเป็นรถสาธารณะ/รถบริการ ให้ใช้ car_public
- ถ้าเห็นรถจักรยานยนต์หรือรูปแบบป้ายของรถจักรยานยนต์ ให้ใช้ moto_private
- ถ้าเห็นป้ายประมูล ให้ใช้ car_auction
- ถ้าเป็นป้ายรถยนต์ส่วนบุคคลทั่วไป ให้ใช้ car_private
- ถ้าไม่แน่ใจประเภทรถ ให้ใช้ unknown
- confidence เป็นตัวเลข 0 ถึง 100
- ห้ามเดาเกินความมั่นใจของภาพ
""".strip()


# ทำความสะอาดข้อความทะเบียนให้เหลือเฉพาะตัวที่ระบบต้องการ
def normalize_plate_number(value):
    text = str(value or "").strip()
    if not text or text == "-":
        return "-"

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0E00-\u0E7F0-9 ]+", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "-"


# ทำให้ชื่อจังหวัดตรงกับชุดข้อมูลของระบบมากที่สุด
# ถ้าไม่ตรงเป๊ะจะลองหาแบบใกล้เคียงก่อน
def normalize_province(value):
    province = str(value or "").strip()
    if not province or province == "-" or not VALID_PROVINCES:
        return "-"

    if province in VALID_PROVINCES:
        return province

    close = get_close_matches(province, PROVINCES, n=1, cutoff=0.82)
    return close[0] if close else "-"


# ตรวจสอบว่าประเภทรถที่โมเดลส่งกลับมาอยู่ในชุดที่ระบบรองรับจริงหรือไม่
def normalize_vehicle_code(value):
    code = str(value or "").strip().lower()
    return code if code in SUPPORTED_VEHICLE_CODES else "unknown"


# แปลง confidence ให้เป็นเลข 0-100 เสมอ
def normalize_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return round(max(0.0, min(100.0, confidence)), 2)


# แปลงผลดิบจาก Gemini ให้เป็นรูปแบบ JSON ที่หน้าเว็บใช้อยู่
def finalize_result(parsed):
    plate_number = normalize_plate_number(parsed.get("plate_number"))
    province = normalize_province(parsed.get("province"))
    vehicle_code = normalize_vehicle_code(parsed.get("vehicle_code"))
    confidence = normalize_confidence(parsed.get("confidence"))

    # ถ้าอ่านทะเบียนไม่ได้เลย ให้เลิกเดาประเภทรถ
    if plate_number == "-":
        vehicle_code = "unknown"

    profile = VEHICLE_PROFILES.get(vehicle_code, VEHICLE_PROFILES["unknown"])
    return {
        "plate_number": plate_number,
        "province": province,
        "vehicle_type": profile["vehicle_type"],
        "plate_color": profile["plate_color"],
        "usage": profile["usage"],
        "confidence": confidence,
    }


# ฟังก์ชันนี้เป็นจุดที่เรียก Gemini เพื่ออ่านทะเบียนจริง
# ส่ง prompt + รูปเข้าไป แล้ว parse JSON ที่ได้กลับมา
def detect_with_gemini(image_bytes, mime_type):
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            build_prompt(),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": GEMINI_RESPONSE_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("Gemini ไม่ส่งผลลัพธ์กลับมา")

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini ส่งผลลัพธ์ในรูปแบบที่อ่านต่อไม่ได้") from exc

    return finalize_result(parsed)


@app.get("/")
def home():
    # เปิดหน้าเว็บหลัก
    return render_template("index.html")


@app.post("/api/detect")
def detect_plate():
    # endpoint นี้รับรูปจากหน้าเว็บ แล้วคืนข้อมูล OCR กลับไปเป็น JSON
    if "image" not in request.files:
        return jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "ไม่พบไฟล์รูปภาพ"}), 400

    try:
        image_bytes, mime_type = prepare_image(file)
        parsed = detect_with_gemini(image_bytes, mime_type)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "ไม่สามารถเรียก Gemini API ได้"}), 502

    return jsonify(parsed)


if __name__ == "__main__":
    # รัน Flask server สำหรับใช้งานในเครื่อง
    app.run(host="0.0.0.0", port=5000, debug=True)

# Thai License Plate OCR

เว็บตรวจสอบป้ายทะเบียนจากภาพ โดยใช้โมเดล YOLO OCR ในเครื่องจากไฟล์ `best.pt`

## ติดตั้ง

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## รันเว็บ

```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์ที่ `http://127.0.0.1:5000`

## วิธีใช้งาน

- อัปโหลดไฟล์ภาพจากเครื่อง
- หรือเปิดกล้องจากเบราว์เซอร์เพื่อถ่ายภาพแล้วส่งเข้า OCR ทันที

หมายเหตุ: ฟังก์ชันกล้องบนหน้าเว็บจะทำงานได้บน `localhost` หรือ `https`

## พฤติกรรมของระบบ

- หน้าเว็บส่งรูปเข้า backend Flask
- backend ใช้โมเดล YOLO OCR ในเครื่องเพื่ออ่านป้ายทะเบียน
- ระบบจะพยายามแยกเลขทะเบียน จังหวัด และประเภทรถจากผลตรวจจับ
- ถ้ารูปไม่ชัด ระบบอาจคืนค่า `-` หรือ `unknown` แทนการเดา

## ไฟล์ที่เกี่ยวข้อง

- `app.py` สำหรับ backend และ OCR pipeline
- `best.pt` สำหรับโมเดลที่ใช้ตรวจจับ
- `vehicle_profiles.yaml` สำหรับ map ประเภทรถ สีป้าย และการใช้งาน
- `main.py` สำหรับเทรนโมเดล
- `test.py` สำหรับทดสอบโมเดล
- `merge_datasets.py` สำหรับรวมชุดข้อมูล

## หมายเหตุเรื่อง `.env`

โหมดปัจจุบันไม่ต้องใช้ API key แล้ว เพราะไม่ได้เรียก Gemini

ไฟล์ `.env` และ `.env.example` ยังเก็บไว้ได้ แต่ไม่จำเป็นต่อการรันเว็บในโหมดนี้

## ประเมินผลกับ `picture-for-test`

ถ้าต้องการวัดว่าระบบเก่งขึ้นจริงหรือไม่ ให้ใช้ชุดไฟล์นี้:

- [picture_for_test_ground_truth.csv](C:/Users/johnn/Downloads/Ai_licenses-main/picture_for_test_ground_truth.csv) สำหรับใส่คำตอบจริงของแต่ละรูป
- [evaluate_picture_for_test.py](C:/Users/johnn/Downloads/Ai_licenses-main/evaluate_picture_for_test.py) สำหรับรันทดสอบแบบ batch

วิธีใช้:

```bash
python evaluate_picture_for_test.py
```

สคริปต์จะ:

- รัน `/api/detect` กับทุกรูปใน `picture-for-test`
- เทียบผลกับ `picture_for_test_ground_truth.csv`
- สรุป accuracy ใน terminal
- เขียนรายงานละเอียดลง `picture_for_test_eval_report.json`

หมายเหตุ:

- ตอนนี้ใน CSV ถูก prefill แค่ `expected_vehicle_family` จากชื่อโฟลเดอร์
- ถ้าต้องการวัดเลขทะเบียน จังหวัด สีป้าย และการใช้งานแบบจริงจัง ให้เติม column ที่เหลือเองก่อนรัน

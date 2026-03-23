# Thai License Plate OCR

เว็บตรวจสอบป้ายทะเบียนจากภาพ โดยส่งรูปเข้า Gemini เพื่ออ่านหมายเลขทะเบียน จังหวัด และประเภทรถ

## ติดตั้ง

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## ตั้งค่า Gemini API ผ่านไฟล์

โปรเจกต์รองรับไฟล์ `.env` และ `.env.local` ที่โฟลเดอร์หลักของโปรเจกต์

1. คัดลอกไฟล์ตัวอย่าง

```powershell
Copy-Item .env.example .env
```

2. แก้ค่าใน `.env`

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite-preview
```

หมายเหตุ:
- ถ้ามีทั้ง `.env` และ `.env.local` ระบบจะอ่านทั้งสองไฟล์
- ถ้าตั้ง environment variable ในเครื่องไว้แล้ว ค่านั้นจะมีลำดับความสำคัญสูงกว่าไฟล์
- `.env` และ `.env.local` ถูกใส่ใน `.gitignore` แล้ว

## รันเว็บ

```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์ที่ `http://127.0.0.1:5000`

หน้าเว็บรองรับ 2 วิธี:
- อัปโหลดไฟล์ภาพจากเครื่อง
- เปิดกล้องจากเบราว์เซอร์เพื่อถ่ายภาพแล้วส่งเข้า OCR ทันที

หมายเหตุ: การเปิดกล้องผ่านเบราว์เซอร์ทำงานได้บน `localhost` หรือผ่าน `https`

## พฤติกรรมของระบบ

- ฝั่งเว็บใช้ Gemini-only สำหรับการอ่านป้ายทะเบียน
- ถ้ารูปไม่ชัด ระบบจะคืนค่า `-` หรือ `unknown` แทนการเดาแรงเกินไป
- ประเภทรถที่หน้าเว็บแสดงจะ map จาก `vehicle_profiles.yaml`
- ระบบพยายามแยก `รถยนต์ส่วนบุคคล`, `รถยนต์สาธารณะ`, `รถจักรยานยนต์`, และ `ป้ายประมูล`

## ไฟล์ dataset และสคริปต์เดิม

โฟลเดอร์ dataset และสคริปต์เทรนเดิมยังอยู่ในโปรเจกต์ เช่น:
- `LPR plate.v1i.yolov11`
- `merged_lpr_motor`
- `motor`
- `main.py`
- `test.py`

ถ้าต้องการกลับไปใช้โมเดล local OCR/YOLO ต้องแก้ `app.py` กลับเป็น pipeline เดิม

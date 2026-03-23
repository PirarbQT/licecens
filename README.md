# Thai License Plate OCR

เว็บตรวจสอบป้ายทะเบียนจากภาพ โดยใช้โมเดล YOLO ที่เทรนไว้ (`best.pt`) และแสดงผลเป็นหมายเลขป้าย + จังหวัด + ประเภทรถ

## ติดตั้ง

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## รันเว็บ

```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์ที่ `http://127.0.0.1:5000`

หน้าเว็บรองรับ 2 วิธี:
- อัปโหลดไฟล์ภาพจากเครื่อง
- เปิดกล้องจากเบราว์เซอร์เพื่อถ่ายภาพแล้วส่งเข้า OCR ทันที

หมายเหตุ: การเปิดกล้องผ่านเบราว์เซอร์ทำงานได้บน `localhost` หรือผ่าน `https`

## รองรับรถมอเตอร์ไซค์

ระบบรองรับการแยก `รถยนต์/รถจักรยานยนต์` ได้เมื่อโมเดลมี class สำหรับประเภทป้าย เช่น `CAR_PLATE`, `MOTO_PLATE`

1. เพิ่ม class ประเภทป้ายใน dataset เทรน (ตัวอย่าง: `CAR_PLATE`, `MOTO_PLATE`)
2. เทรนโมเดลใหม่และแทนไฟล์ `best.pt`
3. ตั้งค่า alias class ในไฟล์ `vehicle_profiles.yaml`

ตัวอย่างใน `vehicle_profiles.yaml`
- `car_private.aliases`: class ที่หมายถึงป้ายรถยนต์
- `moto_private.aliases`: class ที่หมายถึงป้ายมอเตอร์ไซค์

ถ้าโมเดลยังไม่มี class ประเภทป้าย ระบบจะ fallback เป็นรถยนต์ส่วนบุคคลเมื่ออ่านทะเบียนได้

## รวม dataset จากโฟลเดอร์ `motor`

รันคำสั่งนี้เพื่อรวม dataset เดิม + motor แล้ว remap class id อัตโนมัติ:

```bash
python merge_datasets.py --base "LPR plate.v1i.yolov11" --motor "motor" --out "merged_lpr_motor"
```

จากนั้นเทรนด้วย:

```bash
python main.py
```

## รันสคริปต์ทดสอบเดิม

```bash
python test.py --predict path/to/image.jpg --show
```

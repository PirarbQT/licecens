import argparse
import ast
import shutil
from pathlib import Path

# นามสกุลไฟล์รูปที่ยอมรับตอนรวม dataset
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# อ่านรายชื่อ class จาก data.yaml ของแต่ละ dataset
def load_names_from_data_yaml(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    names_line = None
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            names_line = line
            break

    if names_line is None:
        raise ValueError(f"names not found in {path}")

    _, rhs = names_line.split(":", 1)
    names = ast.literal_eval(rhs.strip())
    return [str(x) for x in names]


# หาโฟลเดอร์ images / labels ของ split ที่ต้องการ เช่น train, valid, test
def find_split_dirs(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing {split}/images or {split}/labels under {dataset_dir}")

    return images_dir, labels_dir


# แปลง class index ในไฟล์ label เดิม ให้ตรงกับ index ชุดใหม่หลัง merge
def remap_label_file(src_label: Path, dst_label: Path, idx_map):
    lines_out = []
    with src_label.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            src_idx = int(parts[0])
            if src_idx not in idx_map:
                continue

            parts[0] = str(idx_map[src_idx])
            lines_out.append(" ".join(parts))

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with dst_label.open("w", encoding="utf-8", newline="\n") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")


# คัดลอกข้อมูลของ split เดียวจาก dataset ต้นทางไปยัง output
# พร้อมเติม prefix car_ หรือ moto_ ให้ชื่อไฟล์กันชนกัน
def copy_split(dataset_dir: Path, dataset_tag: str, split: str, out_dir: Path, idx_map):
    try:
        src_images, src_labels = find_split_dirs(dataset_dir, split)
    except FileNotFoundError:
        return 0

    dst_images = out_dir / split / "images"
    dst_labels = out_dir / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in src_images.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = img.stem
        out_stem = f"{dataset_tag}_{stem}"
        dst_img = dst_images / f"{out_stem}{img.suffix.lower()}"
        shutil.copy2(img, dst_img)

        src_label = src_labels / f"{stem}.txt"
        dst_label = dst_labels / f"{out_stem}.txt"
        if src_label.exists():
            remap_label_file(src_label, dst_label, idx_map)
        else:
            dst_label.write_text("", encoding="utf-8")

        count += 1

    return count


# สร้าง data.yaml ใหม่ให้ dataset ที่ merge แล้วใช้งานกับ YOLO ได้ทันที
def write_data_yaml(out_dir: Path, names):
    lines = [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(names)}",
        f"names: {names}",
        "",
        "meta:",
        "  merged_from:",
        "    - LPR plate.v1i.yolov11",
        "    - motor",
    ]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ฟังก์ชันหลักสำหรับรวม dataset รถยนต์กับมอเตอร์ไซค์
def main():
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets with class remap")
    parser.add_argument("--base", default="LPR plate.v1i.yolov11", help="Base dataset directory")
    parser.add_argument("--motor", default="motor", help="Motorcycle dataset directory")
    parser.add_argument("--out", default="merged_lpr_motor", help="Output dataset directory")
    args = parser.parse_args()

    base_dir = Path(args.base).resolve()
    motor_dir = Path(args.motor).resolve()
    out_dir = Path(args.out).resolve()

    # อ่าน class ของ dataset ทั้งสองชุด
    base_names = load_names_from_data_yaml(base_dir / "data.yaml")
    motor_names = load_names_from_data_yaml(motor_dir / "data.yaml")

    # รวมรายชื่อ class โดยกัน class ซ้ำ
    merged_names = list(base_names)
    for n in motor_names:
        if n not in merged_names:
            merged_names.append(n)

    # สร้าง mapping จาก class เดิม -> class ใหม่
    merged_index = {name: i for i, name in enumerate(merged_names)}
    base_map = {i: merged_index[name] for i, name in enumerate(base_names)}
    motor_map = {i: merged_index[name] for i, name in enumerate(motor_names)}

    # ถ้ามี output เดิมอยู่แล้ว ให้ลบทิ้งแล้วสร้างใหม่
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # คัดลอกทั้ง 3 split และสรุปจำนวนรูปที่ย้ายมาได้
    total_counts = {}
    for split in ("train", "valid", "test"):
        c1 = copy_split(base_dir, "car", split, out_dir, base_map)
        c2 = copy_split(motor_dir, "moto", split, out_dir, motor_map)
        total_counts[split] = c1 + c2

    if sum(total_counts.values()) == 0:
        raise RuntimeError("No images copied. Check dataset folders and split structure.")

    write_data_yaml(out_dir, merged_names)

    # แสดงผลสรุปหลัง merge เสร็จ
    print(f"Merged dataset written to: {out_dir}")
    print(f"Total classes: {len(merged_names)}")
    print("Added classes from motor:")
    added = [n for n in motor_names if n not in base_names]
    if added:
        for n in added:
            print(f"  - {n}")
    else:
        print("  (none)")

    print("Image counts:")
    for split, cnt in total_counts.items():
        print(f"  {split}: {cnt}")


if __name__ == "__main__":
    # เริ่มการรวม dataset เมื่อรันไฟล์นี้ตรง ๆ
    main()

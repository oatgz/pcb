import os
import cv2
import torch
import pandas as pd
import sqlite3
from ultralytics import YOLO

# ตรวจสอบว่า GPU ใช้งานได้หรือไม่
print(f"CUDA Available: {torch.cuda.is_available()}")

# ตั้งค่าพาธ
model_path = r"/home/oatgz/pcb/pcb.onnx"
image_test = r"/home/oatgz/pcb/test_640/images"
output_folder = r"/home/oatgz/pcb/output"
os.makedirs(output_folder, exist_ok=True)

# โหลดโมเดล
model = YOLO(model_path)

# สร้าง DataFrame สำหรับเก็บข้อมูลผลลัพธ์
results_data = []

# สร้างหรือเชื่อมต่อกับฐานข้อมูล SQLite
db_path = os.path.join(output_folder, "results.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# สร้างตารางในฐานข้อมูล (หากยังไม่มี)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_results (
        Image TEXT,
        Class TEXT,
        Confidence REAL,
        X1 INTEGER,
        Y1 INTEGER,
        X2 INTEGER,
        Y2 INTEGER
    )
''')

# รัน inference บนภาพในโฟลเดอร์
results = model.predict(image_test, imgsz=640)  # คืนค่าเป็น list ของ Result objects

# ประมวลผลผลลัพธ์
for idx, result in enumerate(results):
    # อ่านภาพต้นฉบับ
    image_path = result.path
    image = cv2.imread(image_path)

    # ดึงข้อมูล bounding boxes
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกรอบ
            confidence = float(box.conf[0].item())  # แปลงจาก Tensor เป็น float
            label = box.cls[0]  # หมายเลขคลาส
            class_name = model.names[int(label)]  # ชื่อคลาส

            # เพิ่มข้อมูลลงใน results_data
            results_data.append({
                "Image": os.path.basename(image_path),
                "Class": class_name,
                "Confidence": confidence,
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })

            # เพิ่มข้อมูลลงในฐานข้อมูล SQLite
            cursor.execute('''
                INSERT INTO detection_results (Image, Class, Confidence, X1, Y1, X2, Y2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (os.path.basename(image_path), class_name, confidence, x1, y1, x2, y2))

            # วาดกรอบบนภาพ
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # บันทึกภาพผลลัพธ์
    output_path = os.path.join(output_folder, f"result_{idx}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

# ยืนยันการบันทึกข้อมูลและปิดการเชื่อมต่อฐานข้อมูล
conn.commit()
conn.close()

print(f"Results saved to SQLite database: {db_path}")

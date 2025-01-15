import os
import cv2
import torch
import pandas as pd
from picamera2 import Picamera2, Preview
from ultralytics import YOLO

# ตรวจสอบว่า GPU ใช้งานได้หรือไม่
print(f"CUDA Available: {torch.cuda.is_available()}")

# ตั้งค่าพาธ
model_path = r"/home/oatgz/pcb/pcb.onnx"
output_folder = r"/home/oatgz/pcb/output"
os.makedirs(output_folder, exist_ok=True)

# โหลดโมเดล
model = YOLO(model_path)

# สร้าง DataFrame สำหรับเก็บข้อมูลผลลัพธ์
results_data = []

# เปิดกล้อง Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)  # ตั้งค่าความละเอียดของกล้อง
picam2.preview_configuration.main.format = "RGB888"   # ตั้งค่าฟอร์แมตภาพ
picam2.configure("preview")
picam2.start()

try:
    print("Press Ctrl+C to stop capturing images...")
    frame_idx = 0

    while True:
        # จับภาพจากกล้อง
        frame = picam2.capture_array()

        # บันทึกภาพต้นฉบับสำหรับใช้งาน (ถ้าจำเป็น)
        image_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
        cv2.imwrite(image_path, frame)

        # รัน inference บนภาพที่ได้จากกล้อง
        results = model.predict(frame, imgsz=640)  # คืนค่าเป็น list ของ Result objects

        # ประมวลผลผลลัพธ์
        for result in results:
            # ดึงข้อมูล bounding boxes
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกรอบ
                    confidence = box.conf[0]  # ค่าความมั่นใจ
                    label = box.cls[0]  # หมายเลขคลาส
                    class_name = model.names[int(label)]  # ชื่อคลาส

                    # เพิ่มข้อมูลลงใน results_data
                    results_data.append({
                        "Image": f"frame_{frame_idx}.jpg",
                        "Class": class_name,
                        "Confidence": confidence,
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2
                    })

                    # วาดกรอบบนภาพ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # แสดงภาพที่มีผลลัพธ์
        cv2.imshow("Inference Results", frame)

        # บันทึกภาพผลลัพธ์
        output_path = os.path.join(output_folder, f"result_{frame_idx}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")

        frame_idx += 1

        # หยุดการทำงานเมื่อกดปุ่ม 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # บันทึกข้อมูลลงในไฟล์ CSV
    csv_output_path = os.path.join(output_folder, "results.csv")
    df = pd.DataFrame(results_data)
    df.to_csv(csv_output_path, index=False)
    print(f"Results saved to CSV: {csv_output_path}")

    # ปิดกล้องและหน้าต่าง
    picam2.stop()
    cv2.destroyAllWindows()


import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import psycopg2

app = Flask(__name__)

# การตั้งค่าโฟลเดอร์
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# โหลดโมเดล YOLO
MODEL_PATH = 'pcb.onnx'
model = YOLO(MODEL_PATH)

# การตั้งค่าฐานข้อมูล
DB_PARAMS = {
    'dbname': 'pcb',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432',
}

def get_db_connection():
    conn = psycopg2.connect(**DB_PARAMS)
    return conn

# Route สำหรับหน้าแสดงผล
@app.route('/')
def index():
    # ดึงข้อมูลจากฐานข้อมูล
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM detection_results')
    results = cursor.fetchall()

    # สรุปข้อมูล
    cursor.execute('''
        SELECT Class, COUNT(*) as Count, AVG(Confidence) as Avg_Confidence
        FROM detection_results
        GROUP BY Class
        ORDER BY Count DESC
    ''')
    summary = cursor.fetchall()

    conn.close()

    return render_template('index.html', results=results, summary=summary)

# Route สำหรับอัปโหลดภาพ
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        if file:
            # บันทึกไฟล์ที่อัปโหลด
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # ทำการ Predict
            result = model.predict(file_path, imgsz=640)[0]
            image = cv2.imread(file_path)
            results_data = []

            # ดึงข้อมูลผลลัพธ์
            conn = get_db_connection()
            cursor = conn.cursor()

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกรอบ
                confidence = float(box.conf[0].item())  # ค่า Confidence
                label = box.cls[0]  # หมายเลขคลาส
                class_name = model.names[int(label)]  # ชื่อคลาส

                # เพิ่มข้อมูลลงในฐานข้อมูล
                cursor.execute('''
                    INSERT INTO detection_results (Image, Class, Confidence, X1, Y1, X2, Y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (file.filename, class_name, confidence, x1, y1, x2, y2))

                # วาดกรอบบนภาพ
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # บันทึกภาพผลลัพธ์
            output_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(output_path, image)
            conn.commit()
            cursor.close()
            conn.close()

            # แสดงผลลัพธ์
            return redirect(url_for('serve_image', filename=file.filename))

    return render_template('upload.html')

# Route สำหรับให้บริการภาพ
@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# อนุญาตให้เครื่องอื่นเข้าใช้งานได้
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

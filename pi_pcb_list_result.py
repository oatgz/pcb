import sqlite3
import pandas as pd

# ตั้งค่าพาธของฐานข้อมูล SQLite
db_path = "/home/oatgz/pcb/output/results.db"

# เชื่อมต่อกับฐานข้อมูล SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# คำสั่ง SQL เพื่อดึงข้อมูลจากตาราง detection_results
cursor.execute("SELECT * FROM detection_results")

# ดึงข้อมูลทั้งหมดมาเป็น list of tuples
rows = cursor.fetchall()

# แปลงข้อมูลที่ดึงมาเป็น DataFrame ของ pandas เพื่อให้สะดวกในการจัดการและแสดงผล
df = pd.DataFrame(rows, columns=["Image", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

# แสดงข้อมูลทั้งหมด
print(df)

# ปิดการเชื่อมต่อฐานข้อมูล
conn.close()

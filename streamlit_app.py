import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import requests
import os

# ฟังก์ชันดาวน์โหลด Model (ป้องกันไฟล์ใหญ่เกิน GitHub)
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import requests
import os

# --- แก้ไขฟังก์ชันดาวน์โหลดใหม่ให้เช็คขนาดไฟล์ด้วย ---
def download_model(url, save_path):
    if not os.path.exists(save_path) or os.path.getsize(save_path) < 1000000: # ถ้าไม่มีไฟล์ หรือไฟล์เล็กผิดปกติ (< 1MB)
        with st.spinner("กำลังดาวน์โหลด AI Model (ประมาณ 500MB) โปรดรอสักครู่..."):
            try:
                # ใช้ stream=True เพื่อจัดการไฟล์ขนาดใหญ่ได้ดีขึ้น
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                st.success("ดาวน์โหลด Model สำเร็จ!")
            except Exception as e:
                st.error(f"ดาวน์โหลดล้มเหลว: {e}")
                if os.path.exists(save_path): os.remove(save_path) # ลบไฟล์ที่เสียทิ้ง

# --- เปลี่ยน Link ใหม่ที่เสถียรกว่า (Link ตรงจาก Hugging Face) ---
model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
model_path = "inswapper_128.onnx"
download_model(model_url, model_path)

# ส่วนที่เหลือของโค้ด load_models() และ UI เหมือนเดิมครับ...
st.title("🎭 JAAO Face Swap Studio")

# 1. เตรียม Model
model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
model_path = "inswapper_128.onnx"
download_model(model_url, model_path)

@st.cache_resource
def load_models():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(model_path, download=False)
    return app, swapper

app, swapper = load_models()

# 2. ส่วนอัปโหลดรูป
col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader("รูปหน้าของคุณ (Source)", type=['jpg', 'png'])
with col2:
    target_file = st.file_uploader("รูปที่จะเอาไปใส่ (Target)", type=['jpg', 'png'])

if source_file and target_file:
    # แปลงไฟล์เป็น format ที่ OpenCV ใช้ได้
    source_img = Image.open(source_file)
    target_img = Image.open(target_file)
    
    source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    if st.button("🚀 เริ่มสลับใบหน้า!"):
        with st.spinner("กำลังสลับใบหน้า..."):
            # ตรวจจับใบหน้า
            source_faces = app.get(source_cv)
            target_faces = app.get(target_cv)

            if source_faces and target_faces:
                # ทำการสลับหน้า (ใช้ใบหน้าที่เจออันแรก)
                res = target_cv.copy()
                res = swapper.get(res, target_faces[0], source_faces[0], paste_back=True)
                
                # แสดงผล
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="สลับหน้าสำเร็จแล้ว!")
                
                # ปุ่มดาวน์โหลดผลลัพธ์
                result_pil = Image.fromarray(res_rgb)
                st.download_button("ดาวน์โหลดรูป", data=source_file.getvalue(), file_name="swapped.png")
            else:
                st.error("ตรวจไม่พบใบหน้าในรูปภาพครับ ลองเปลี่ยนรูปใหม่ดูนะ")

import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os

# --- SETTINGS ---
st.set_page_config(page_title="Monster Detector", page_icon="👤")
st.title("👤 Monster Detector Pro")

# Improved lighting stabilizer
def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

@st.cache_resource
def load_and_train():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces_data = []
    labels = []
    
    # Check for a folder of images or just the single file
    image_paths = []
    if os.path.exists("my_faces"):
        image_paths = [os.path.join("my_faces", f) for f in os.listdir("my_faces") if f.endswith(('.jpg', '.png'))]
    if os.path.exists("me.jpg"):
        image_paths.append("me.jpg")

    if not image_paths:
        return face_cascade, recognizer, False

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Normalize the image
        img = cv2.resize(img, (400, 400)) 
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (100, 100))
            roi = apply_clahe(roi) # Advanced lighting fix
            faces_data.append(roi)
            labels.append(1) # ID 1 = Monster

    if len(faces_data) > 0:
        recognizer.train(faces_data, np.array(labels))
        return face_cascade, recognizer, True
            
    return face_cascade, recognizer, False

face_cascade, recognizer, model_ready = load_and_train()

class FaceIDProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process small version for speed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        
        faces = face_cascade.detectMultiScale(gray_small, 1.1, 5)

        for (x, y, w, h) in faces:
            x_up, y_up, w_up, h_up = x*2, y*2, w*2, h*2
            
            if model_ready:
                roi = gray_small[y:y+h, x:x+w]
                roi = cv2.resize(roi, (100, 100))
                roi = apply_clahe(roi)
                
                id_, confidence = recognizer.predict(roi)
                
                # With more images, Dist usually drops. 
                # We'll stick to 130 as a safe 'Monster' threshold.
                dist_text = f"Dist: {int(confidence)}"
                
                if id_ == 1 and confidence < 130:
                    label, color = "Monster", (0, 255, 0)
                else:
                    label, color = "Not monster", (0, 0, 255)
                
                cv2.rectangle(img, (x_up, y_up), (x_up+w_up, y_up+h_up), color, 2)
                cv2.putText(img, label, (x_up, y_up-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(img, dist_text, (x_up, y_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if model_ready:
    st.sidebar.success(f"✅ Trained on {len(os.listdir('my_faces')) if os.path.exists('my_faces') else 1} images")
else:
    st.error("Upload images to 'my_faces' folder or 'me.jpg' to GitHub.")

webrtc_streamer(
    key="monster-detector-pro",
    video_processor_factory=FaceIDProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
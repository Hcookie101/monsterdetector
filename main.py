import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

st.title("🛡️ Secure ID Scanner")

# Load model once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

@st.cache_resource
def train_model():
    if not os.path.exists("me.jpg"): return False
    img = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        recognizer.train([img[y:y+h, x:x+w]], np.array([1]))
        return True
    return False

model_ready = train_model()

# This is just a placeholder to keep the connection alive without math
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        return frame

ctx = webrtc_streamer(
    key="snapshot-mode",
    video_transformer_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_transformer and model_ready:
    if st.button("📸 IDENTIFY PERSON"):
        # Get the latest frame from the video stream
        img = ctx.video_transformer.last_frame
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                st.warning("No face detected. Get closer to the camera.")
            
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                id_, conf = recognizer.predict(roi)
                
                if conf < 120:
                    st.success(f"ACCESS GRANTED: Owner identified (Dist: {int(conf)})")
                    st.balloons()
                else:
                    st.error(f"ACCESS DENIED: Unknown person (Dist: {int(conf)})")
                
                # Show the snapshot taken
                st.image(img[y:y+h, x:x+w], caption="Scanned Face", width=200)
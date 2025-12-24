import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os

# --- SETTINGS ---
st.set_page_config(page_title="Monster Detector", page_icon="👤")
st.title("👤 Monster Detector")

@st.cache_resource
def load_and_train():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    if os.path.exists("me.jpg"):
        # 1. Load reference image
        img = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
        # 2. Find the face in that image
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # 3. Crop and resize to standard 100x100
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (100, 100))
            roi = cv2.equalizeHist(roi) # Normalize lighting
            
            # 4. Train model on this specific face
            recognizer.train([roi], np.array([1]))
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
            # Scale coords back up to original image size
            x_up, y_up, w_up, h_up = x*2, y*2, w*2, h*2
            
            if model_ready:
                # SYMMETRIC PROCESSING: Must match training steps exactly
                roi = gray_small[y:y+h, x:x+w]
                roi = cv2.resize(roi, (100, 100))
                roi = cv2.equalizeHist(roi)
                
                id_, confidence = recognizer.predict(roi)
                
                # UI Logic
                dist_text = f"Dist: {int(confidence)}"
                
                # ADJUST THIS: Since your baseline is 120, we set threshold to 140
                if id_ == 1 and confidence < 140:
                    label, color = "Monster", (0, 255, 0)
                else:
                    label, color = "Not monster", (0, 0, 255)
                
                # Draw Box and Text
                cv2.rectangle(img, (x_up, y_up), (x_up+w_up, y_up+h_up), color, 2)
                cv2.putText(img, label, (x_up, y_up-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(img, dist_text, (x_up, y_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if model_ready:
    st.sidebar.success("✅ Model Trained on 'me.jpg'")
else:
    st.error("Missing 'me.jpg'! Upload it to GitHub and refresh.")

webrtc_streamer(
    key="secure-id-v5",
    video_processor_factory=FaceIDProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
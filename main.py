import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
import os

st.title("👤 Specific Person Detector")

# 1. Load the Detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. Train the model on YOU
@st.cache_resource
def train_on_me():
    if not os.path.exists("me.jpg"):
        return None
    
    # Load your photo
    img = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        # We give "You" the ID of 1
        recognizer.train([roi], np.array([1]))
        return True
    return False

trained = train_on_me()

if not trained:
    st.error("Upload a clear photo of yourself named 'me.jpg' to your GitHub repo!")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Predict who it is
        id_, confidence = recognizer.predict(roi_gray)
        
        # Confidence: Lower is better for LBPH
        if id_ == 1 and confidence < 80:
            label = "Owner (Access Granted)"
            color = (0, 255, 0) # Green
        else:
            label = "Stranger (Access Denied)"
            color = (0, 0, 255) # Red

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="person-id",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
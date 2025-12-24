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

if trained:
    st.sidebar.success("✅ Reference photo loaded!")
else:
    st.sidebar.error("❌ Could not find a face in 'me.jpg'. Try a clearer photo.")

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

def video_frame_callback(frame):
    st.session_state.frame_count += 1
    img = frame.to_ndarray(format="bgr24")
    
    # ONLY process every 5th frame to prevent CPU overload
    if st.session_state.frame_count % 5 != 0:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Lower resolution for calculation
    small_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        # Scale coords back up (5x because fx=0.2)
        x_up, y_up, w_up, h_up = x*5, y*5, w*5, h*5
        roi = gray[y:y+h, x:x+w]
        
        try:
            # The Recognizer is the heavy part - we wrap it in safety
            id_, conf = recognizer.predict(roi)
            
            if id_ == 1 and conf < 150:
                label, color = "OWNER", (0, 255, 0)
            else:
                label, color = "SCANNING", (255, 255, 255)
                
            cv2.rectangle(img, (x_up, y_up), (x_up+w_up, y_up+h_up), color, 2)
            cv2.putText(img, f"{label} ({int(conf)})", (x_up, y_up-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception:
            continue # If math fails, just skip to the next frame

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="monster-v3", # Changing the key forces a fresh connection
    video_frame_callback=video_frame_callback,
    # Lower bitrate and resolution for phones
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 10} # 10 FPS is plenty for detection
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
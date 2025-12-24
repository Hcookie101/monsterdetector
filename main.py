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
    
    # STEP 1: Create a tiny version of the image for faster processing
    # Processing a 480p image is much faster than 1080p
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    gray_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # STEP 2: Detect faces on the SMALL image
    faces = face_cascade.detectMultiScale(gray_small, 1.1, 5)

    for (x, y, w, h) in faces:
        # Scale the coordinates back UP to match the original big image
        x_orig, y_orig, w_orig, h_orig = x*4, y*4, w*4, h*4
        
        roi_gray = gray_small[y:y+h, x:x+w]
        
        try:
            id_, confidence = recognizer.predict(roi_gray)
            
            if id_ == 1 and confidence < 110:
                label = f"Owner (Match: {int(100 - (confidence/1.5))}%)"
                color = (0, 255, 0)
            else:
                label = "Stranger"
                color = (0, 0, 255)
        except:
            label = "Scanning..."
            color = (255, 255, 255)

        # Draw on the ORIGINAL high-quality image
        cv2.rectangle(img, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
        cv2.putText(img, label, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="person-id",
    video_frame_callback=video_frame_callback,
    # This part is CRITICAL for mobile data/phones:
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    }
)
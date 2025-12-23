import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- APP SETUP ---
st.title("👹 Monster Detector")

# Initialize the detector using the Tasks API
@st.cache_resource
def create_detector():
    # This downloads the actual model file needed for detection
    model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    base_options = python.BaseOptions(model_asset_buffer=None, model_asset_path=None)
    # We use a URL or local path for the model to ensure it loads
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=cv2.utils.findDataFile(model_url)),
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.FaceDetector.create_from_options(options)

# If the Tasks API is also struggling, we use a fallback to pure OpenCV 
# so your app at least WORKS while we debug MediaPipe.
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # FALLBACK: Simple OpenCV Face Detection (always works)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "MONSTER!", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="monster-detector",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp

# Use a cache to prevent reloading the model on every frame
@st.cache_resource
def get_face_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )

st.title("👹 Monster Detector")

try:
    face_detector = get_face_detector()
    st.sidebar.success("✅ MediaPipe Ready")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.stop()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # MediaPipe needs RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            # Draw standard detection
            mp.solutions.drawing_utils.draw_detection(img, detection)
            
            # Label as Monster
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x = int(bbox.left_px * iw)
            y = int(bbox.top_px * ih)
            cv2.putText(img, "MONSTER!", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="monster-detector",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
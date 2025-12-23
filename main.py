import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp

# Setup MediaPipe outside the loop for speed
mp_faces = mp.solutions.face_detection
face_detector = mp_faces.FaceDetection(model_selection=0, min_detection_confidence=0.5)

st.title("👹 Monster Detector")
st.write("If the camera doesn't start, check the 'lock' icon in your browser bar.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # 1. Convert to RGB (MediaPipe requirement)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Process
    results = face_detector.process(img_rgb)

    # 3. Draw if a "Monster" (Human) is found
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.left_px * iw), int(bboxC.top_px * ih), \
                         int(bboxC.width_px * iw), int(bboxC.height_px * ih)
            
            # Green box for the monster
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, "MONSTER FOUND", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="monster-detector",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer

st.title("Fast Face ID (MediaPipe)")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            x, y, w, h = int(bboxC.left_px * iw), int(bboxC.top_px * ih), \
                         int(bboxC.width_px * iw), int(bboxC.height_px * ih)
            
            # For "Specific Person" recognition, we normally compare 
            # feature vectors here. For now, let's draw the stable detection.
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Face Detected", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="mediapipe-filter",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
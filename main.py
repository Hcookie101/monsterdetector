import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer

# DIRECT IMPORT to avoid AttributeError
try:
    from mediapipe.python.solutions import face_detection as mp_face
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    
    face_detector = mp_face.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )
    st.success("✅ MediaPipe Linked!")
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

st.title("👹 Monster Detector")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_detector.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            # Drawing the monster box
            mp_drawing.draw_detection(img, detection)
            
            # Add custom text
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x = int(bbox.left_px * iw)
            y = int(bbox.top_px * ih)
            cv2.putText(img, "MONSTER", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="monster-cam",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import av

st.title("🛡️ Secure ID Scanner")

# Load models with cache to save memory
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained = False
    if os.path.exists("me.jpg"):
        img = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            recognizer.train([img[y:y+h, x:x+w]], np.array([1]))
            trained = True
    return face_cascade, recognizer, trained

face_cascade, recognizer, trained = load_models()

# The new "Processor" style
class SnapshotProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Store the frame so we can use it when the button is clicked
        self.last_frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="snapshot-v4",
    video_processor_factory=SnapshotProcessor, # Updated from transformer
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    if st.button("📸 IDENTIFY PERSON"):
        img = ctx.video_processor.last_frame
        
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                st.warning("No face detected. Look directly at the camera.")
            
            for (x, y, w, h) in faces:
                if trained:
                    roi = gray[y:y+h, x:x+w]
                    id_, conf = recognizer.predict(roi)
                    
                    if conf < 120:
                        st.success(f"ACCESS GRANTED (Dist: {int(conf)})")
                        st.balloons()
                    else:
                        st.error(f"ACCESS DENIED (Dist: {int(conf)})")
                
                st.image(img[y:y+h, x:x+w], caption="Scanned Face", width=200)
        else:
            st.info("Wait for the camera to start...")
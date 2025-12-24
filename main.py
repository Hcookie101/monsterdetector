import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title="AI Face ID", page_icon="👤")
st.title("👤 Secure Person Detector")

@st.cache_resource
def load_and_train():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces_data = []
    labels = []
    
    # helper to process images for training
    def prepare_image(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        img = cv2.equalizeHist(img) # Fix lighting
        return img

    # Label 1: You
    if os.path.exists("me.jpg"):
        faces_data.append(prepare_image("me.jpg"))
        labels.append(1)
        
    # Label 2: Not You (Stranger Baseline)
    if os.path.exists("stranger.jpg"):
        faces_data.append(prepare_image("stranger.jpg"))
        labels.append(2)
        
    if len(labels) > 0:
        recognizer.train(faces_data, np.array(labels))
        return face_cascade, recognizer, True
    return face_cascade, recognizer, False

face_cascade, recognizer, model_ready = load_and_train()

# --- 2. THE VIDEO PROCESSOR ---
class FaceIDProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = "Scanning..."
        self.last_color = (255, 255, 255)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process at a lower resolution to avoid freezing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        
        faces = face_cascade.detectMultiScale(gray_small, 1.1, 5)

        for (x, y, w, h) in faces:
            # Scale coordinates back up
            x_up, y_up, w_up, h_up = x*2, y*2, w*2, h*2
            
            if model_ready:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                roi = cv2.equalizeHist(roi)
                
                # 1. Get the ID and the raw distance (Dist)
                id_, confidence = recognizer.predict(roi)
                
                # 2. Determine the label and color
                if id_ == 1 and confidence < 95:
                    label = "OWNER"
                    color = (0, 255, 0)
                else:
                    label = "UNKNOWN"
                    color = (0, 0, 255)
                
                # 3. PRINT "DIST" ON SCREEN (The new line)
                # This puts the raw distance number right under the label
                dist_text = f"Dist: {int(confidence)}"
                
                # Draw Box
                cv2.rectangle(img, (x_up, y_up), (x_up+w_up, y_up+h_up), color, 2)
                # Draw Label (OWNER/UNKNOWN)
                cv2.putText(img, label, (x_up, y_up-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # Draw Distance value
                cv2.putText(img, dist_text, (x_up, y_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE UI ---
if not model_ready:
    st.error("Missing photos! Upload 'me.jpg' and 'stranger.jpg' to GitHub.")
else:
    st.sidebar.success("✅ Security Model Active")
    st.sidebar.info("Tip: If it misidentifies you, lower the confidence threshold in the code (line 69).")

webrtc_streamer(
    key="face-id-final",
    video_processor_factory=FaceIDProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
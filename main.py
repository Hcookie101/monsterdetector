import streamlit as st
import cv2
import face_recognition
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer

st.title("Real-Time Face ID")

# --- Step 1: Load your reference photo ---
# Upload a clear photo of yourself named 'me.jpg' to your project folder
try:
    my_image = face_recognition.load_image_file("him.jpg")
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
    known_face_encodings = [my_face_encoding]
    known_face_names = ["Darsil Patel"] # Your Name Here
except Exception as e:
    st.error("Please upload a 'me.jpg' file to the folder to start recognition.")
    st.stop()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Resize for faster processing
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches YOU
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_face_names[0]
            color = (0, 255, 0) # Green for you
        else:
            color = (0, 0, 255) # Red for strangers

        # Scale back up (since we resized to 0.25)
        top, right, bottom, left = top*4, right*4, bottom*4, left*4

        # Draw the box and name
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="face-id", video_frame_callback=video_frame_callback)
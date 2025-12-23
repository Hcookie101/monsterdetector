import streamlit as st
import cv2
import numpy as np

st.title("Face Detector")

# This specific line is what triggers the browser popup
picture = st.camera_input("First, let's test your camera")

if picture:
    # Convert the photo to a format OpenCV can read
    img_bytes = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Load the detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Show the result
    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    st.success(f"Found {len(faces)} face(s)!")
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Face Mask Detection", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00FFD1;
}
.subtitle {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 20px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #161B22;
    box-shadow: 0px 0px 15px rgba(0,255,209,0.2);
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<p class="title">😷 Face Mask Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time CNN Based Detection</p>', unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = load_model("mask_detector.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 224

# -------------------- ALERT SOUND --------------------
def play_alert():
    st.audio("https://www.soundjay.com/buttons/beep-07.wav", autoplay=True)

# -------------------- PREDICTION FUNCTION --------------------
def predict_face(face):
    face = cv2.convertScaleAbs(face, alpha=1.5, beta=30)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(face, verbose=0)

    no_mask = pred[0][1]
    mask = pred[0][0]

    if mask > no_mask:
        return "Mask 😷", (0,255,0), mask
    else:
        return "No Mask ❌", (0,0,255), no_mask

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Control Panel")
option = st.sidebar.radio("Select Mode", ["📷 Upload Image", "🎥 Live Detection"])

# -------------------- UPLOAD MODE --------------------
if option == "📷 Upload Image":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        # 🎯 Loading animation
        with st.spinner("Analyzing Image..."):
            time.sleep(1)

        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        col1.image(image, caption="Uploaded Image", use_container_width=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("⚠️ No face detected")

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

            label, color, conf = predict_face(face)

            # 📊 Confidence meter
            st.progress(float(conf))

            if label == "No Mask ❌":
                st.error("🚨 ALERT: No Mask Detected!")
                play_alert()

            text = f"{label} ({conf*100:.2f}%)"

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, text, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col2.image(img, caption="Detected Output", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- LIVE MODE --------------------
elif option == "🎥 Live Detection":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    run = st.checkbox("▶ Start Camera")

    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]

            label, color, conf = predict_face(face)

            # 📊 Update confidence meter
            progress_bar.progress(float(conf))

            if label == "No Mask ❌":
                play_alert()

            text = f"{label} ({conf*100:.2f}%)"

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, text, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

    st.markdown('</div>', unsafe_allow_html=True)
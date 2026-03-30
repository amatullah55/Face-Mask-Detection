import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mask_detector.h5")

# Use built-in haarcascade (no download needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        pred = model.predict(face, verbose=0)
        (mask, noMask) = pred[0]

        label = "Mask 😷" if mask > noMask else "No Mask ❌"
        color = (0, 255, 0) if label == "Mask 😷" else (0, 0, 255)

        confidence = max(mask, noMask) * 100
        text = f"{label}: {confidence:.2f}%"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
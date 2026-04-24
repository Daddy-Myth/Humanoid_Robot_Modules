import os
import cv2
import time
from ultralytics import YOLO


# ─── Model Setup ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, 'yolov8n.pt'))


# ─── Detection Function ──────────────────────────────────────────────────────────

def detect_objects(frame, conf=0.4):
    results = model(frame, conf=conf, verbose=False)
    annotated = results[0].plot()
    return annotated


# ─── Detection Loop ──────────────────────────────────────────────────────────────

def detect(cam=0, conf=0.5):
    vdo = cv2.VideoCapture(cam)
    prev_time = time.time()

    while True:
        ret, frame = vdo.read()
        if not ret:
            break

        frame = detect_objects(frame)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow('output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vdo.release()
    cv2.destroyAllWindows()


# ─── Entry Point ─────────────────────────────────────────────────────────────────

WEBCAM = 0
DROIDCAM = 1

if __name__ == '__main__':
    detect(DROIDCAM)
    #detect(WEBCAM)

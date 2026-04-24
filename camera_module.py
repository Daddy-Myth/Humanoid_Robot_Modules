"""
camera_module.py - Humanoid Robot Vision
Combines face recognition and object detection on a shared camera feed.

Camera IDs:  0 = laptop webcam  |  1 = DroidCam
"""

import sys, os, time
import cv2, torch, numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_path   = os.path.join(BASE_DIR, "Face_Recognition")
object_path = os.path.join(BASE_DIR, "Object_Detection")

sys.path.insert(0, face_path)
sys.path.insert(0, object_path)

# ── Module imports ────────────────────────────────────────────────────────────
from face_module   import mtcnn, encode, all_people_faces
from object_module import detect_objects

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_ID      = 1      # 0 = webcam, 1 = DroidCam
FACE_THRESHOLD = 0.7
OBJ_CONF       = 0.45
WINDOW_NAME    = "Humanoid — Face + Object Detection"


# ── Face recognition helper ───────────────────────────────────────────────────
def apply_face_recognition(frame: np.ndarray) -> np.ndarray:
    if not all_people_faces:
        return frame
    try:
        batch_boxes, cropped_images = mtcnn.detect_box(frame)
    except Exception:
        return frame
    if cropped_images is None:
        return frame

    for box, cropped in zip(batch_boxes, cropped_images):
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if len(cropped.shape) == 3:
            cropped = cropped.unsqueeze(0)

        emb   = encode(cropped)
        dists = {name: (e - emb).norm().item() for name, e in all_people_faces.items()}
        best  = min(dists, key=dists.get)
        label = best if dists[best] < FACE_THRESHOLD else "Unknown"
        color = (0, 220, 0) if label != "Unknown" else (0, 0, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"CUDA: {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}")
        return
    print(f"Camera {CAMERA_ID} opened. Press Q to quit.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = detect_objects(frame, conf=OBJ_CONF)
            frame = apply_face_recognition(frame)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame     = draw_hud(frame, fps)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Q pressed — shutting down...")
                break

    except KeyboardInterrupt:
        print("\nCtrl-C — shutting down...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

import os
import cv2
import torch
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType


# ─── Model Setup ────────────────────────────────────────────────────────────────

def encode(img):
    res = resnet(torch.Tensor(img))
    return res

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


# Load models
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)


# ─── Load Known Faces ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
saved_pictures = os.path.join(BASE_DIR, "Saved")
all_people_faces = {}

for file in os.listdir(saved_pictures):
    if file.endswith('.jpg') or file.endswith('.png'):
        person_face = os.path.splitext(file)[0]
        image_path = os.path.join(saved_pictures, file)

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = mtcnn(img_rgb)

        if cropped is not None:
            if len(cropped.shape) == 3:
                cropped = cropped.unsqueeze(0)  # Add batch dim

            embedding = encode(cropped).detach()
            all_people_faces[person_face] = embedding


# ─── Detection Loop ──────────────────────────────────────────────────────────────

def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    prev_time = time.time()

    while True:
        ret, img0 = vdo.read()
        if not ret:
            break

        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(v) for v in box]

                if len(cropped.shape) == 3:
                    cropped = cropped.unsqueeze(0)

                img_embedding = encode(cropped)

                detect_dict = {}
                for name, embedding in all_people_faces.items():
                    detect_dict[name] = (embedding - img_embedding).norm().item()

                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'

                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img0, min_key, (x + 5, y + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                )

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            img0, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow('output', img0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vdo.release()
    cv2.destroyAllWindows()


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    detect(0)

# Humanoid_Robot_Modules

## Modules

### 1. Speech-to-Text (STT) ✅
- Status: Done (.py)
- Input: audio (numpy array)
- Output: text
- Uses: Whisper

### 2. Face Detection & Recognition ✅
- Status: Done (.py)
- Input: image/frame
- Output:
  - bounding boxes
  - identity (if recognized)
- Uses:
  - MTCNN (face detection)
  - FaceNet (embeddings)

### 3. Object Detection ✅
- Status: Done (.py)
- Input: image/frame
- Output: annotated frame with bounding boxes + labels
- Uses: YOLOv8n (pretrained COCO, 80 classes)
- Key classes: laptop, keyboard, person, chair,
               bottle, phone, book, tv, mouse,
               dining table, bed, couch...

### 4. Camera Module ✅
- Status: Done (.py)
- Input: live camera feed
- Output: single window with face + object annotations overlaid
- Uses: face_module + object_module
- Notes: STT excluded — will be integrated separately

## Next Steps
- [x] Build object detection (ipynb → .py)
- [x] Clean STT module (remove debug / optimize)
- [x] Write camera_module.py (face + object on shared feed)
- [ ] Integrate STT into main pipeline
- [ ] Write main.py (combine all modules)
- [ ] Test all together on live camera
- [ ] Optimize performance (FPS, latency)

## Structure
## Structure
```
Humanoid/
├── STT/
│   └── stt_module.py
├── Face_Recognition/
│   ├── face_module.py
│   └── Saved/              ← known face images (.jpg / .png)
├── Object_Detection/
│   ├── object_module.py
│   └── yolov8n.pt
├── Notebooks/
│   ├── STT_module.ipynb
│   ├── RTFD.ipynb
│   └── object.ipynb
├── camera_module.py
└── README.md
```

## Rule

Each module must:
- take clean input
- return clean output
- run independently


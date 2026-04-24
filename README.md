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


## Next Steps

- [x] Build object detection (ipynb → .py)
- [x] Clean STT module (remove debug / optimize)
- [ ] Write main.py (combine all 3 modules)
- [ ] Test all 3 together on live camera
- [ ] Optimize performance (FPS, latency)

## Structure
```
Humanoid/
├── STT/
│   └── stt_module.py
├── Face_Recognnition/
│   └── face_module.py
└── Object_Detection/
│   └── object_module.py
├── Notebooks/
│   ├── STT_module.ipynb
│   ├── RTFD.ipynb
│   └── object.ipynb
├── main.py
└── README.md
```

## Rule

Each module must:
- take clean input
- return clean output
- run independently


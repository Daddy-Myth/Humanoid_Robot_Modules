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


### 3. Object Detection ❌
- Status: Not started
- Plan:
  - Prototype in notebook (YOLO / pretrained model)
  - Convert to `.py`
  - Create `detect_objects(frame)`


## Next Steps

- [ ] Build object detection (ipynb → `.py`)
- [ ] Clean STT module (remove debug / optimize)
- [ ] Integrate all modules into `main.py`
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


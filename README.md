# Humanoid_Robot_Modules

## Modules

### 1. Speech-to-Text (STT) ✅
- Status: Done (.py)
- Input: audio (numpy array)
- Output: text
- Uses: Whisper


### 2. Face Detection ⚠️
- Status: In notebook → convert to `.py`
- Tasks:
  - Extract logic
  - Remove notebook code
  - Create `detect_faces(frame)`


### 3. Object Detection ❌
- Status: Not started
- Plan:
  - Prototype in notebook (YOLO / pretrained model)
  - Convert to `.py`
  - Create `detect_objects(frame)`


## Next Steps

- [ ] Convert face detection → `.py`
- [ ] Build object detection (ipynb → `.py`)
- [ ] Clean STT module
- [ ] Integrate all modules


## Structure
```
humanoid/
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


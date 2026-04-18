import threading
import time
import openwakeword
import sounddevice as sd
import numpy as np
import torch
from openwakeword.model import Model
from openwakeword.utils import download_models
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
BLOCKSIZE     = 512
SILENCE_LIMIT = 60      # ~1.9s of silence before cutting off recording
WAKE_THRESHOLD = 0.3

# ─── Shared Events ────────────────────────────────────────────────────────────
wake_word_detected = threading.Event()
stop_event         = threading.Event()

# ─── Phase 1: Wake Word Listener ──────────────────────────────────────────────
def wake_word_listener():
    model = Model(wakeword_models=['hey_jarvis'], inference_framework="onnx")
    print("Listening for Wake Word...")
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280) as stream:
        while not stop_event.is_set():
            if wake_word_detected.is_set():
                time.sleep(0.1)
                continue
            audio_chunk, _ = stream.read(1280)
            prediction = model.predict(audio_chunk.flatten())
            if prediction["hey_jarvis"] > 0.3:
                print(f"Wake word detected! ({prediction['hey_jarvis']:.4f})")
                wake_word_detected.set()
                time.sleep(1.5)  # cooldown to avoid re-trigger

# ─── Phase 5: Transcription (Whisper Apex) ────────────────────────────────────
def load_asr_pipeline():
    model_id   = "Oriserve/Whisper-Hindi2Hinglish-Apex"
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    dtype      = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading ASR model on {device}...")
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if device == "cuda" else -1
    )
    print("ASR model loaded.")
    return asr_pipeline


def transcribe(audio, asr_pipeline):
    audio = audio.flatten().astype("float32")
    result = asr_pipeline(
        audio,
        generate_kwargs={
            "task": "transcribe",
            "language": None,           # auto-detect per utterance
            "forced_decoder_ids": None  # suppresses deprecation conflict
        }
    )
    return result["text"]

# ─── Phase 6: Output Handler ──────────────────────────────────────────────────
def output_handler(transcript):
    transcript = transcript.strip().lower()
    print(f"Output: {transcript}")
    # LLM hook goes here
    return transcript

# ─── Phase 2–4: Audio Capture + VAD + Preprocessing ──────────────────────────
def record_audio(asr_pipeline, vad_model):
    while not stop_event.is_set():
        wake_word_detected.wait()           # wait for "hey jarvis"
        if stop_event.is_set():
            break

        print("++++++++++++++++++++++++++++++++++++++++++++++++++|Conversation Started|++++++++++++++++++++++++++++++++++++++++++++++++++\n")

        # INNER loop — keeps recording until "bye jarvis"
        while not stop_event.is_set() and wake_word_detected.is_set():
            audio_buffer   = []
            silence_counter = 0
            print("Recording...")

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=BLOCKSIZE) as stream:
                while True:
                    chunk, _ = stream.read(BLOCKSIZE)
                    flat = chunk.flatten()
                    audio_buffer.append(flat)

                    # Phase 3: VAD
                    chunk_tensor = torch.from_numpy(flat).unsqueeze(0)  # [1, 512]
                    speech_prob  = vad_model(chunk_tensor, SAMPLE_RATE).item()

                    if speech_prob < 0.3:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    if silence_counter >= SILENCE_LIMIT:
                        break

            # Phase 4: Preprocessing
            audio = np.concatenate(audio_buffer)
            print(f"Audio length: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)")

            # Phase 5: Transcription
            transcript = transcribe(audio, asr_pipeline)
            print(f"Transcript: {transcript}")

            if not transcript or transcript.strip().lower() in ("", "nan"):
                if stop_event.is_set():
                    break
                print("Empty transcript, re-listening...")
                continue

            if any(phrase in transcript.lower() for phrase in
                   ["bye jarvis", "bye, jarvis", "bye jarvis", "bai jarvis", "by jarvis","bye zaarves"]):
                print("Goodbye Have a nice day!")
                print("--------------------------------------------------|Conversation Ended|---------------------------------------------------\n")
                wake_word_detected.clear()
                break   # exit inner loop → back to waiting for wake word

            # Phase 6
            output_handler(transcript)
            # no clear() — loop back to record next sentence

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"CUDA: {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    download_models()
    asr_pipeline = load_asr_pipeline()
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")

    stop_event.clear()
    wake_word_detected.clear()

    t1 = threading.Thread(target=wake_word_listener, daemon=True)
    t2 = threading.Thread(target=record_audio, args=(asr_pipeline, vad_model), daemon=True)
    t1.start()
    t2.start()

    try:
        while t1.is_alive() and t2.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        wake_word_detected.set()    # unblock .wait() so threads can exit

    t1.join(timeout=3)
    t2.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()

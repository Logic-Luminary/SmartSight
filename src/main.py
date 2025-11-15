import os
from detector import ObjectDetector
from tts import TextToSpeech
from app import SmartSightApp
from memory import DetectionMemory
from camera import CameraStream

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "../models/yolov5s.onnx")

    detector = ObjectDetector(model_path)
    tts = TextToSpeech(cooldown = 3)
    memory = DetectionMemory()
    camera = CameraStream(src=0)

    app = SmartSightApp(detector, tts, memory, camera)
    app.run()

if __name__ == "__main__":
    main()
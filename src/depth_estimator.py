import cv2
import numpy as np
import onnxruntime as ort

class DepthEstimator:
    def __init__(self, model_path = "../models/midas_small.onnx"):
        self.session = ort.InferenceSession(model_path, providers = ["CPUExecutionProvider"])
        self.height = 256
        self.width = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype = np.float32)

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height)).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def estimate(self, frame):
        inp = self.preprocess(frame)
        depth = self.session.run(None, {"input": inp})[0][0]

        # Normalize the depth map range to [0 1] for better accuracy
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.ones_like(depth)  # Avoid division by zero
        return depth
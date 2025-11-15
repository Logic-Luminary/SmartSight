import onnxruntime as ort
import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model_path, confidence_threshold=0.4, iou_threshold=0.45):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = open("../models/coco.names").read().strip().split("\n")

    def preprocess(self, frame):
        image = cv2.resize(frame, (640, 640))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, image.shape[:2], frame.shape[:2]

    def detect(self, frame):
        input_tensor, resized_shape, original_shape = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(outputs, original_shape, resized_shape)

    def postprocess(self, outputs, original_shape, resized_shape):
        detections = []
        boxes = []
        confidences = []
        class_ids = []

        output = outputs[0]
        rows = output.shape[1]

        orig_h, orig_w = original_shape
        resized_h, resized_w = resized_shape

        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        for i in range(rows):
            row = output[0][i]
            confidence = row[4]
            if confidence < self.confidence_threshold:
                continue

            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            if score < self.confidence_threshold:
                continue

            cx, cy, w, h = row[0:4]
            x = int((cx - w / 2) * scale_x)
            y = int((cy - h / 2) * scale_y)
            w_box = int(w * scale_x)
            h_box = int(h * scale_y)

            boxes.append([x, y, w_box, h_box])
            confidences.append(float(score))
            class_ids.append(class_id)

        # Apply NMS to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w_box, h_box = boxes[i]
            x2 = x + w_box
            y2 = y + h_box
            detections.append({
                "label": self.class_names[class_ids[i]],
                "confidence": confidences[i],
                "box": (x, y, x2, y2)
            })

        return detections
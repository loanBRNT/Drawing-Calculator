import operator

import cv2
import torch
import json
class Detection:

    model = None
    digit = ["0","1","2","3","4","5","6","7","8","9"]
    operation = ["+","-","/","*"]

    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def resolve(self,prediction):
        return eval("".join(prediction))

    def load_model(self, model_path):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.7:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr)
        return frame

    def predict(self, frame):
        frame = cv2.resize(frame, (640, 640))

        results = self.score_frame(frame)

        out = []
        temp = []

        for n in range(len(results[0])):
            i = 0
            pos_x = ( results[1][n][0] + results[1][n][2] ) / 2
            while i < len(out) and pos_x > temp[i]:
                i += 1

            out.insert(i, self.class_to_label(results[0][n]))
            temp.insert(i,pos_x)

        return out

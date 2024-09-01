import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import smtplib
import json
import supervision as sv
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# place where the results will be saved
os.chdir("/home/hany_jr/Ai/Data sets/Yolo runs")


data = {}
# Open and read the JSON file
with open("/home/hany_jr/Ai/email_settings.json", "r") as file:
    data = json.load(file)


def send_email(number_of_people=0):
    # Email settings
    email = data["from_email"]
    password = data["password"]
    to_email = data["to_email"]
    print(
        email,
        password,
    )
    # Email content
    subject = "Security Alert"
    body = "There is an intruder in your house\nNumber of people detected: " + str(
        number_of_people
    )

    # Email server settings
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, password)

    # Email message
    message = f"Subject: {subject}\n\n{body}"

    # Send email
    server.sendmail(email, to_email, message)
    server.quit()
    return


class Object_detection:

    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.email_sent = False

        self.deivce = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model()

        self.CLASS_NAMES = self.model.names

        self.box_annotator = sv.BoxAnnotator()

    def load_model(self):
        model = YOLO("yolov8m.pt")
        model.fuse()

        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_names = []

        # extract the bounding boxes, confidences, and class IDs

        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            # if 0 is detected, then it is a person
            if class_id == 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.scores.cpu().numpy()

                xyxys.append(xyxy)
                confidences.append(confidence)
                class_names.append(class_id)

        detection = sv.Detections.from_ultralytics(results[0])
        frame = self.box_annotator.annotate(frame, detection)

        return frame, class_names

    def __call__(self):

        # Open the camera to capture the video
        cap = cv2.VideoCapture(self.capture_index)

        # Check if the camera is opened
        assert cap.isOpened()
        # Yolo resoloution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        frame_count = 0
        while True:

            start_time = time()

            ret, frame = cap.read()

            assert ret

            results = self.predict(frame)  # Predict the frame

            frame, class_names = self.plot_bboxes(results, frame)

            if len(class_names) > 0:
                number_of_people = len(class_names)
                if not self.email_sent:
                    send_email(number_of_people)
                    self.email_sent = True
            else:
                self.email_sent = False

            end_time = time()

            fps = 1 / np.round(end_time - start_time)

            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv2.imshow("frame", frame)

            frame_count += 1

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

            cap.release()
            cv2.destroyAllWindows()


detector = Object_detection(0)
detector()

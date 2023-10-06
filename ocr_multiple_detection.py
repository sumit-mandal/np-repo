import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import time
import numpy as np
import pandas as pd
import cv2
import time
import os
from datetime import datetime

margin = 20

import detection_cls
run = detection_cls.Ppe_Detection_1()

import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import time

# OCR configuration for text recognition
pytesseract.pytesseract.tesseract_cmd = 'tesseract'  # Use the system PATH

# Load YOLOv3 config and weights for OCR
# ocr_net = cv2.dnn.readNet('ocr-tiny-v3-PI-256_200000.weights', 'ocr-tiny-v3-PI-256.cfg')

# Load class names for OCR (if available)
ocr_classes = []
with open("../np.names", "r") as f:
    ocr_classes = [line.strip() for line in f]

# Load PPE detection class
import detection_cls
run = detection_cls.Ppe_Detection_1()

class Ppe_Detection():

    def __init__(self):
        # Load YOLOv3 config and weights for PPE detection
        self.weightfile = '../files/new/number_plate_reduced_mish_final.weights'
        self.cfgfile = '../files/new/number_plate_reduced_mish.cfg'
        self.PpeNet = cv2.dnn.readNet(self.weightfile, self.cfgfile)
        self.classes = self.get_classes()
        layer_names = self.PpeNet.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.PpeNet.getUnconnectedOutLayers()]

        # Initialize desired FPS, delay, and frame skip
        self.desired_fps = 30 # You can change this value as needed
        self.delay = int(1000 / self.desired_fps)
        self.frame_skip =5  # Adjust this to skip frames

        # Create a directory for saving frames
        now = datetime.now()
        self.folder_name = now.strftime("%d-%b-%Y %H-%M-%S")
        os.makedirs(self.folder_name)

    def get_classes(self):
        with open("../np.names", "r") as f:
            self.classes_val = [line.strip() for line in f.readlines()]
        return self.classes_val

    def detection(self, img):
        # Your existing object detection code here (YOLOv3)
        start = time.perf_counter()
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.PpeNet.setInput(blob)
        outs = self.PpeNet.forward(self.output_layers)
        time_took = time.perf_counter() - start
        fps = str(int(1 / time_took))

        # getting the list
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                # print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle Coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

        info = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                conf = confidences[i]
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                type = '{}'.format(self.classes[class_ids[i]])
                info.append([x, y, w, h, type, conf])

        new_frame_time = time.time()

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            for i in indexes:
                x, y, w, h, label, conf = info[0]
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, fps, (10, 30), font, 3, color, 3)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        return info

    def cropping_detection(self):
        count = 0
        margin = 20
        cap = cv2.VideoCapture("../test_vid.mp4")

    # Create a window for the main frame
        cv2.namedWindow("Main Frame", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            r, f = cap.read()

        # Skip frames based on frame_skip value
            if count % self.frame_skip == 0:
                detection = self.detection(f)

                try:
                    x, y, w, h, cls, conf = detection[0]
                except Exception as e:
                    pass

                try:
                    if cls:
                       print("Plate")

                       cropped_img = f[y - margin:y + h + margin, x - margin:x + w + margin]

                    # Perform OCR on the cropped image
                       recognized_text = self.perform_ocr(cropped_img)
                       print('Recognized Text:', recognized_text)

                    # Display the main frame
                       cv2.imshow("Main Frame", f)

                    # Display the cropped frame with recognized text
                       cv2.putText(cropped_img, recognized_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       cv2.imshow("Cropped Frame", cropped_img)

                    # Save frames
                       self.save_frames(f, cropped_img)

                       if cv2.waitKey(1) & 0xff == ord('q'):
                            break
                    else:
                        pass
                except Exception as e:
                    print("In except part")

            count += 1  # Increment frame count

    cv2.destroyAllWindows()



    def save_frames(self, main_frame, cropped_frame):
        # Save the main frame
        main_frame_filename = f"{self.folder_name}/main_frame_{time.time()}.jpg"
        cv2.imwrite(main_frame_filename, main_frame)

        # Save the cropped frame
        cropped_frame_filename = f"{self.folder_name}/cropped_frame_{time.time()}.jpg"
        cv2.imwrite(cropped_frame_filename, cropped_frame)

    def perform_ocr(self, img):
        # Perform OCR on the cropped image
        text = pytesseract.image_to_string(img)
        return text

# Create an instance of Ppe_Detection
ppe = Ppe_Detection()

# Start the cropping_detection process
ppe.cropping_detection()

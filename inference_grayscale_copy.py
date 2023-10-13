
import cv2
import numpy as np
from multiple_detection_copy import Ppe_Detection
margin = 20
from detection_cls import Ppe_Detection_1
from flask import Flask, render_template, Response

desired_frame_size = (640, 480)
# Initialize the line parameters (y-coordinate where the line will be drawn)
line_y = 350  # You can adjust this value based on your setup


app = Flask(__name__)


def draw_line(frame, line_y):
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

def final_inference(frame):
    # Load YOLOv3 config and weights
    net = cv2.dnn.readNet('../files/new/ocr-tiny-v3-PI-256_200000.weights', '../files/new/ocr-tiny-v3-PI-256.cfg')

    # Load class names (if available)
    classes = []
    with open("../ocr.names", "r") as f:
        classes = [line.strip() for line in f]

        

        cropped_img = None  # Initialize cropped_img here
        run = Ppe_Detection().detection(frame)
        print("run",run)

        try:
            #run = Ppe_Detection().detection(frame)
            x, y, w, h, cls, conf = run[0]


            if cls and (y + h / 2) > line_y:
                print("Number Plate detected:", cls)
                cropped_img = frame[y - margin:y + h + margin, x - margin:x + w + margin]
                gray_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                blob = cv2.dnn.blobFromImage(gray_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(net.getUnconnectedOutLayersNames())
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:  # You can adjust this confidence threshold
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        width = int(detection[2] * frame.shape[1])
                        height = int(detection[3] * frame.shape[0])

                        # Calculate coordinates for the top-left corner of the bounding box
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to remove overlapping bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes on the frame
            for i in range(len(boxes)):
                if i in indices:
                    x, y, width, height = boxes[i]
                    label = str(classes[class_ids[i]])
                    
                    cv2.imwrite("Cropped_img.png",cropped_img)
                    print(label)

                    # print("\n\n")
                    confidence = confidences[i]
                    #cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    # cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except Exception as e:
            print("e",e)
            pass

    
    return frame,cropped_img
        # Display the result frame

        #process the outputs and draw bounding boxes - similar to your previous code)

def generate_frames():
     # Open a video file or capture video from a camera
    cap = cv2.VideoCapture("../test_vid.mp4")

    # Calculate delay based on desired frame rate (e.g., 30 frames per second)
    fps = 20
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Display the frame


        frame = cv2.resize(frame, desired_frame_size)
        draw_line(frame, line_y)
        frame, cropped_img = final_inference(frame)
        # cv2.imshow("Number Plate Detection", frame)
        vehicle = Ppe_Detection_1().detection(frame)
        print("Vehicle",vehicle)

        # if cropped_img is not None and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
            # cv2.imshow("Number Plate Detection Cropped", cropped_img)
        # if cropped_img:
        #     print("Cropped_img",cropped_img)
        #     cv2.imwrite("Write.jpg",cropped_img)

        # # Break the loop if 'q' key is pressed
        # if cv2.waitKey(delay) & 0xFF == ord('q'):
        #     break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    # Release the video capture object and close all windows
    # cap.release()
    # cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

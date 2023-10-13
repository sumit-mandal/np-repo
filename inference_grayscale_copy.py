# import cv2
# import numpy as np
# from multiple_detection_copy import Ppe_Detection
# margin = 20
# from detection_cls import Ppe_Detection_1





# def final_inference():
#     # Load YOLOv3 config and weights
#     net = cv2.dnn.readNet('../files/new/ocr-tiny-v3-PI-256_200000.weights', '../files/new/ocr-tiny-v3-PI-256.cfg')

#     # Load class names (if available)
#     classes = []
#     with open("../ocr.names", "r") as f:
#         classes = [line.strip() for line in f]

#     # Open a video file
#     # video_path = 'video.mp4'
#     cap = cv2.VideoCapture("../test_vid.mp4")
#     # cap = cv2.VideoCapture(0)

#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
#         if not ret:
#             break  # Break the loop if we reach the end of the video
        
#         # roi = frame[0:1200,200:600]
        
#         vehicle = Ppe_Detection_1().detection(frame)
#         print("Vehicle",vehicle)

#         try:
#             x,y,w,h,cls,conf = vehicle[0]

#             if cls:
#                 print("Hola I am working",cls)

#                 vehicle_cropped = frame[y-margin:y+h+margin,x-margin:x+w+margin]

            


        
        


#             run = Ppe_Detection().detection(vehicle_cropped)
#             print("run",run)

#         # try:
#             x,y,w,h,cls,conf = run[0]

#             if cls:
#                 print("Number Plate detected",cls)
#                 cropped_img = vehicle_cropped[y-margin:y+h+margin,x-margin:x+w+margin]
#         # except Exception as e:
#         #     # print("cropped_omg waal exception",e)
#         #     pass

#         # try:
#         # Convert the frame to grayscale
#             gray_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        

            

#             # Resize the grayscale frame to the YOLOv3 input size (416x416)
#             blob = cv2.dnn.blobFromImage(gray_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

#             # Set the input blob for the network
#             net.setInput(blob)

#             # Run forward pass
#             outs = net.forward(net.getUnconnectedOutLayersNames())
        

#             # Process the outputs to get bounding boxes and confidence scores
#             boxes = []
#             confidences = []
#             class_ids = []

#             for out in outs:
#                 for detection in out:
#                     scores = detection[5:]
#                     class_id = np.argmax(scores)
#                     confidence = scores[class_id]

#                     if confidence > 0.5:  # You can adjust this confidence threshold
#                         center_x = int(detection[0] * frame.shape[1])
#                         center_y = int(detection[1] * frame.shape[0])
#                         width = int(detection[2] * frame.shape[1])
#                         height = int(detection[3] * frame.shape[0])

#                         # Calculate coordinates for the top-left corner of the bounding box
#                         x = int(center_x - width / 2)
#                         y = int(center_y - height / 2)

#                         boxes.append([x, y, width, height])
#                         confidences.append(float(confidence))
#                         class_ids.append(class_id)

#             # Apply non-maximum suppression to remove overlapping bounding boxes
#             indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#             # Draw bounding boxes on the frame
#             for i in range(len(boxes)):
#                 if i in indices:
#                     x, y, width, height = boxes[i]
#                     label = str(classes[class_ids[i]])
#                     print(label)
#                     confidence = confidences[i]
#                     cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
#                     # cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         except Exception as e:
#             print("e",e)
#             pass
#         # Display the result frame
#         cv2.imshow("YOLOv3 Object Detection", frame)

#         # Break the loop if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close the display window
#     cap.release()
#     cv2.destroyAllWindows()

# final_inference()



import cv2
import numpy as np
from multiple_detection_copy import Ppe_Detection
margin = 20
from detection_cls import Ppe_Detection_1
desired_frame_size = (640, 480)
# Initialize the line parameters (y-coordinate where the line will be drawn)
line_y = 350  # You can adjust this value based on your setup
def draw_line(frame, line_y):
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

def final_inference():
    # Load YOLOv3 config and weights
    net = cv2.dnn.readNet('../files/new/ocr-tiny-v3-PI-256_200000.weights', '../files/new/ocr-tiny-v3-PI-256.cfg')

    # Load class names (if available)
    classes = []
    with open("../ocr.names", "r") as f:
        classes = [line.strip() for line in f]

    # Open a video file or capture video from a camera
    cap = cv2.VideoCapture(0)

    # Calculate delay based on desired frame rate (e.g., 30 frames per second)
    fps = 20
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, desired_frame_size)
        draw_line(frame, line_y)

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
                    print(label)
                    confidence = confidences[i]
                    #cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    # cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except Exception as e:
            print("e",e)
            pass
        # Display the result frame

        #process the outputs and draw bounding boxes - similar to your previous code)

        # Display the frame
        cv2.imshow("Number Plate Detection", frame)
        vehicle = Ppe_Detection_1().detection(frame)
        print("Vehicle",vehicle)

        if cropped_img is not None and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
            cv2.imshow("Number Plate Detection Cropped", cropped_img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break


    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


final_inference()

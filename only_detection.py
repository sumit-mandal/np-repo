import numpy as np
import pandas as pd
import cv2
import time

def detection(img):
    height,width,channel = img.shape

    # Detecting objects

    ppenet = cv2.dnn.readNet('backup/yolov4_best.weights','./yolov4.cfg')


    layer_names = ppenet.getLayerNames()

    output_layers = [layer_names[i - 1] for i in ppenet.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
    ppenet.setInput(blob)
    outs = ppenet.forward(output_layers)

    with open("obj.names","r") as f:
        classes_val = [line.strip() for line in f.readlines()]


    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        # print(out)
        for detection in out:
            # print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4:

                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x-w/2)
                y = int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)

    info = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y = boxes[i][0], boxes[i][1]
            w,h = boxes[i][2], boxes[i][3]

            conf = confidences[i]
            if x<0:
                x = 0
            if y < 0:
                y = 0
            type = '{}'.format(classes_val[class_ids[i]])
            info.append([x,y,w,h,type,conf])

    new_frame_time = time.time()

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        for i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes_val[class_ids[i]])
            color = (0,255,145)
            # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            # cv2.putText(img,fps,(10,30),font,3,color,3)
            # cv2.putText(img,label,(x,y+30),font,3,color,3)

    # print(boxes,confidences,class_ids)
    print("opened")
    # print(info)
    # print(info)
    return info

# IMG = cv2.imread("img_0.jpg")
# detection(IMG)
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 960)  # set video width
    cap.set(4, 720)  # set video height
    while True:
        # begin = time.time()
        ret, frame = cap.read()
        detection(frame)
        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


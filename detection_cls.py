import numpy as np
import pandas as pd
import cv2
import time




class Ppe_Detection_1():

    def __init__(self):
   
        self.weightfile =  '../yolov4-tiny.weights'
        self.cfgfile =  '../yolov4-tiny.cfg'
        self.PpeNet = cv2.dnn.readNet(self.weightfile,self.cfgfile)
        self.classes = self.get_classes()
        layer_names = self.PpeNet.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.PpeNet.getUnconnectedOutLayers()]

                            # = [self.PpeNet.getLayerNames()[(i[0] - 1)] for i in self.PpeNet.getUnconnectedOutLayers()]
    def get_classes(self):
        # self.classes = []

        with open("../coco.names","r") as f:
            self.classes_val = [line.strip() for line in f.readlines()]


        return self.classes_val

    # def print_all(self):
    #     print(self.classes)

    def detection(self,img):


        start = time.perf_counter()
        height,width,channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
        self.PpeNet.setInput(blob)
        outs = self.PpeNet.forward(self.output_layers)
        time_took = time.perf_counter() - start
        fps = str(int(1/time_took))

        # getting the list
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:
                    # object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*width)

                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # Rectangle Coordinates
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
                type = '{}'.format(self.classes[class_ids[i]])
                info.append([x,y,w,h,type,conf])

        new_frame_time = time.time()


        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            for i in indexes:
                x,y,w,h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = (0,255,145)
                # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                # cv2.putText(img,fps,(10,30),font,3,color,3)
                # cv2.putText(img,label,(x,y+30),font,3,color,3)

        # print(boxes,confidences,class_ids)
        # print("opened")
        # print("info",info)

        return info




# ppe = Ppe_Detection_1()
# ppe.get_classes()
# ppe.print_all()


#############
# def inference_image():
#     img = cv2.imread("sumit_off.jpg")
#     ppe.detection(img)
#     cv2.imshow("Image",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def run_video():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         r,f = cap.read()
#         try:
#             info = ppe.detection(f)
#         except Exception as e:
#             print("______",e)
#         cv2.imshow("image",f)

#         if cv2.waitKey(1) & 0xFF == ord("q") :
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# run_video()
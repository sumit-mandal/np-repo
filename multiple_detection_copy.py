import numpy as np
import pandas as pd
import cv2
import time
margin = 20


import detection_cls
run = detection_cls.Ppe_Detection_1()




class Ppe_Detection():

    def __init__(self):
        self.weightfile = '../files/new/number_plate_reduced_mish_final.weights'
        self.cfgfile = '../files/new/number_plate_reduced_mish.cfg'
        self.PpeNet = cv2.dnn.readNet(self.weightfile,self.cfgfile)
        self.classes = self.get_classes()
        layer_names = self.PpeNet.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.PpeNet.getUnconnectedOutLayers()]

                            # = [self.PpeNet.getLayerNames()[(i[0] - 1)] for i in self.PpeNet.getUnconnectedOutLayers()]
    def get_classes(self):
        # self.classes = []

        with open("../np.names","r") as f:
            self.classes_val = [line.strip() for line in f.readlines()]


        return self.classes_val

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
                # print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:
                    # object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)

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
                color = (0,0,145)
                # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                # cv2.putText(img,fps,(10,30),font,3,color,3)
                # cv2.putText(img,label,(x,y+30),font,3,color,3)



        # print(boxes,confidences,class_ids)
        # print("opened")
        # print(info)
        return info

    # def cropping_detection(self,detection,frame):

        # count = 0
        # cropped_frame = []
        # value_img = []
        # cap = cv2.VideoCapture("../test_vid.mp4")
        # cap = cv2.VideoCapture(0)
        # while cap.isOpened():

        #     r,f = cap.read()
        #     # counts +=1

        #     detection = ppe.detection(f)

        #     print("detection",detection)
        #     try:
        #         x,y,w,h,cls,conf = detection[0]
        #     except Exception as e:
        #         print("eeeeee",e)
        #         pass
        #     try:
        #         if cls:
        #             print("car detected")

        #             cropped_img = f[y-margin:y+h+margin,x-margin:x+w+margin]

                    

        #             final = run.detection(cropped_img)

        #             print('final',final)

                    
        #             cv2.imshow("Hi",cropped_img)
        #             if cv2.waitKey(1) & 0xff==ord('q'):
        #                 break
                    

        #         else:
        #             pass
        #     except Exception as e:
        #         print("In except part",e)



        # cap.release()
        # cv2.destroyAllWindows()
       
        # return cropped_img


    # detection = ppe_frame.detection(f)
        # try:
        #     x,y,w,h,cls,conf = detection[0]
        # except Exception as e:
        #     print("eeeeee",e)
        #     pass 
        # try:
        #     if cls:
        #         cropped_img = frame[y-margin:y+h+margin,x-margin:x+w+margin]
        #         print("car detected")
        # except:
        #     pass
        
        # return cropped_img




 

# ppe = Ppe_Detection()
# ppe.cropping_detection()

from only_detection import detection
import cv2
import threading
from time import time
import multiprocessing
import os
import datetime
import boto3
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


# def get_hls_url():
#     STREAM_NAME = "ucCPm3MMSbPmzxWcvd1zb5"
#     kvs = boto3.client("kinesisvideo")
#     # Grab the endpoint from GetDataEndpoint
#     endpoint = kvs.get_data_endpoint(
#         APIName="GET_HLS_STREAMING_SESSION_URL",
#         StreamName=STREAM_NAME
#     )['DataEndpoint']
#
#     # print(endpoint)
#
#     # # Grab the HLS Stream URL from the endpoint
#     kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
#     url = kvam.get_hls_streaming_session_url(
#         StreamName=STREAM_NAME,
#         PlaybackMode="LIVE"
#     )['HLSStreamingSessionURL']
#     return url


def run_video():
    previous = time()
    run_video.delta = 0
    run_video.count = 0
    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        current = time()

        run_video.delta += current - previous
        previous = current


        r,f = cap.read()
        try:
            # write_video(f)
            create_alert(f)
            # run_video.count+=1

                # ppe.time_screenshot()

        except Exception as e:
            print("______",e)
        # out.write(f)
        cv2.imshow("image",f)


        new_frame_time = time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        print('fps',fps)

        if cv2.waitKey(1) & 0xFF == ord("q") :
            break

    cap.release()
    cv2.destroyAllWindows()


def create_alert(frame):

    try:

        info = detection(frame)

        print(run_video.count)
        x,y,w,h,label,conf = info[0]
        ct = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if label == "aadhar":
            crop_img = frame[y:y+h,x:x+w]
            run_video.count+=1
            cv2.imwrite("screen_shot/img_"+str(ct)+".jpg",crop_img)
            # time_screenshot()

            # take_screenshot(frame,run_video.count)


            # print( .count)
            print("Alert!!! Condition not met")

            # cv2.imwrite("img_"+str(count)+".jpg",f)
        if label == "pan":
            crop_img = frame[y:y+h,x:x+w]
            run_video.count+=1
            cv2.imwrite("screen_shot/pan/img_"+str(ct)+".jpg",crop_img)

    except Exception as e:
        print("_______-",e)

# def take_screenshot(frame,count):
#     if not os.path.exists('screen_shot'):
#        os.makedirs('screen_shot')
#     if run_video.delta > 3:
#         cv2.imwrite("screen_shot/img_"+str(count)+".jpg",frame)
#         # print("Ready to capture")
#         run_video.delta = 0


# def write_video(frame):
#




run_video()
# thread2.join()
# thread1.join()

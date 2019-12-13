from multiprocessing import Process, Queue
import time
import cv2 as cv
import torch
import amcrest
import numpy as np
import sys
import imutils

ip = "192.168.8.9"
port = "80"
username = "admin"
password = "internsarethebest1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cam = amcrest.AmcrestCamera(ip, port, username, password).camera
speed=1
batch_size=16

def process_frames(fq, pq, frame_length):
    while True:
        start = time.monotonic()
        frame_list = []       
        for i in range(batch_size):
            frame = fq.get()
            frame_list.append(frame)
        for i in range(batch_size):
            pq.put(frame_list[i])
        end = time.monotonic()
        print(end - start)
        print(fq.qsize(), pq.qsize())

def display_processed_frames(pq, frame_length):
    while True:
        start = time.monotonic()
        frame = pq.get()
        cv.imshow(('Camera'), frame)
        end = time.monotonic()
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            sys.exit(0)
        time.sleep(max(0, frame_length - (end - start)))
    

if __name__ == '__main__':
    url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
    video = cv.VideoCapture(url)
    
    fq = Queue()
    pq = Queue()
    frame_length = 1/60.0
    pf = Process(target=process_frames, args=(fq, pq, frame_length))
    pf.start()
    dpf = Process(target=display_processed_frames, args=(pq, frame_length))
    dpf.start()
    
    while True:
        start = time.monotonic()
        _, frame = video.read()
        frame = imutils.resize(frame, width=800)
        fq.put(frame)
        end = time.monotonic()

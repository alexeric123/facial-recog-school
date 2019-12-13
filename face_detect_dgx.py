#from torch.multiprocessing import Pool, Process, set_start_method, Queue
from torch.multiprocessing import Process, Queue
import cv2 as cv
import time
import numpy as np
import pickle
import amcrest
import torch
import net
import dlib
from align_dlib import AlignDlib
import imutils
import argparse
import os

ip = "192.168.8.9"
port = "80"
username = "admin"
password = "internsarethebest1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cam = amcrest.AmcrestCamera(ip, port, username, password).camera
speed=1
batch_size=32

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--embedding-model", default="net.pth",
                    help="path to the deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", default="output/recognizer.pickle",
                    help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default="output/le.pickle",
                    help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.45,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-d", "--detector", default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")

    args = vars(ap.parse_args())



    

    fq = Queue()
    pq = Queue()
    frame_length = 1/45.0
    pf = Process(target=get_frames, args=(fq,))
    pf.start()
    dpf = Process(target=display_processed_frames, args=(pq, frame_length))
    dpf.start()
    process_frames(fq, pq, args)
    
def get_frames(fq):
    url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
    video = cv.VideoCapture(url)
    i = 0
    while True:
        if i % 3 == 0:
            _, frame = video.read()
            frame = imutils.resize(frame, width=800)
            fq.put(frame)
        i += 1

def batch_process(frame_list, detector, face_aligner, embedder, recognizer, le, args):
    (ih, iw) = frame_list[0].shape[:2]
    recognized = []
    for i, frame in enumerate(frame_list):
        # grab the image dimensions
        # construct a blob from the image
        image_blob_array[i] = cv.dnn.blobFromImage(
            cv.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)


    boxes, index_list = create_face_blob(frame_list, image_blob_array, detector, face_aligner, iw, ih)
    if len(boxes) == 0:
        return

    inputs = torch.from_numpy(blobArray[0:len(boxes)]).to(device)
    vec = embedder.forward(inputs).cpu().numpy()
    # perform classification to recognize the face
    predsArray = recognizer.predict_proba(vec)
    #index_list = [index_list[-1]] + index_list[:-1]
    #index_list.reverse()
    for i in range(len(boxes)):
        (x, y, endX, endY) = boxes[i].astype("int")
        proba, name = find_predictions(predsArray[i], le)
        if proba < args["confidence"]:
            text = "{}: {:.2f}%".format("Unknown", proba * 100)
        else:
            text = "{}: {:.2f}%".format(name, proba * 100)
        recognized.append((x, y, endX, endY, text, index_list[i]))
    for face in recognized:
        cv.rectangle(frame_list[face[5]], face[:2], face[2:4], (255,255,0), 2)  
        cv.putText(frame_list[face[5]], face[4], face[:2], cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

def process_frames(fq, pq, args):
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv.dnn.readNetFromCaffe(protoPath, modelPath)
    
    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_aligner = AlignDlib(predictor_model)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    torch.set_grad_enabled(False)
    embedder = net.model
    embedder.load_state_dict(torch.load('net.pth'))
    embedder.to(device)
    embedder.eval()

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    while True:
        start = time.monotonic()
        frame_list = []       
        for i in range(batch_size):
            frame = fq.get()
            frame_list.append(frame)
        batch_process(frame_list, detector, face_aligner, embedder, recognizer, le, args)
        for i in range(batch_size):
            pq.put(frame_list[i])
        end = time.monotonic()

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

def find_predictions(preds, le):
    j = np.argmax(preds)
    proba = preds[j]
    return proba, le.classes_[j]

max_faces = 20
blobArray = np.zeros((max_faces * batch_size, 3, 96, 96), dtype=np.float32)
image_blob_array = np.zeros((batch_size, 3, 300, 300), dtype=np.float32)

def create_face_blob(frame_list, image_blob_array, detector, face_aligner, iw, ih):
    blobArray[:] = 0
    boxes = []
    index_list = []

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(image_blob_array)
    detections = detector.forward()
    min_confidence = 0.45
    j = 0
    for i in range(detections.shape[2]): 
        # extract the confidence (i.e., probability) associated with the
        # predictions
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        if startX < 0 or startY < 0 or endX > iw or endY > ih:
            continue

        frame_index = int(detections[0, 0, i, 0])
        # align the face
        rect = dlib.rectangle(startX, startY, endX, endY)
        face = face_aligner.align(
                96,
                frame_list[frame_index],
                rect,
                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE
        )
        index_list.append(frame_index)
        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        blobArray[j] = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
        boxes.append(box)
        j += 1
    
    return boxes, index_list


if __name__ == '__main__':
    main()

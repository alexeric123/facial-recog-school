
import cv2 as cv
import time
import numpy as np
import pickle
import serial #
import torch
import net
import dlib
from align_dlib import AlignDlib
import imutils
import argparse
import math #
import time #
from time import sleep #
import os

ip = "192.168.8.9"
port = "80"
username = "admin"
password = "internsarethebest1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speed=1

ser = serial.Serial('/dev/ttyACM0', 115200)
names = []

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
    url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
    video = cv.VideoCapture(0)


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

    detect_timer = 3

    faces = []
    recognized = []

    while True:
        _, frame = video.read()
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
       

        if detect_timer == 0:
            # resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio) 
            image = imutils.resize(frame, width=600)
            boxes = create_face_blob(frame, detector, face_aligner)
            if len(boxes) == 0:
                continue
            recognized = []
            
            inputs = torch.from_numpy(blobArray[0:len(boxes)]).to(device)
            vec = embedder.forward(inputs).cpu().numpy()
            # perform classification to recognize the face
            predsArray = recognizer.predict_proba(vec)
            detect_timer = 3
            for i in range(len(boxes)):
                (x, y, endX, endY) = boxes[i].astype("int")
                proba, name = find_predictions(predsArray[i], le)
                if proba < args["confidence"]:
                    text = "{}: {:.2f}%".format("Unknown", proba * 100)
                else:
                    text = "{}: {:.2f}%".format(name, proba * 100)
		    
                recognized.append((x, y, endX, endY, text))

        print(recognized)
        for face in recognized:
            cv.rectangle(frame, face[:2], face[2:4], (255,255,0), 2)  
            cv.putText(frame, face[4], face[:2], cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv.imshow(('Camera'), frame)

        key = cv.waitKey(1) & 0xff 
        if key == ord('q'):
             break
        detect_timer -= 1

    cv.destroyAllWindows()

def find_predictions(preds, le):
    j = np.argmax(preds)
    proba = preds[j]
    print(proba)
    return proba, le.classes_[j]


max_faces = 20
blobArray = np.zeros((max_faces, 3, 96, 96), dtype=np.float32)

def create_face_blob(image, detector, face_aligner):
    blobArray[:] = 0
    boxes = []
    # grab the image dimensions
    (ih, iw) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv.dnn.blobFromImage(
        cv.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    min_confidence = 0.45
    for i in range(min(detections.shape[2], max_faces)): 
        # extract the confidence (i.e., probability) associated with the
        # predictions
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            break
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        if startX < 0 or startY < 0 or endX > iw or endY > ih:
            continue


        # align the face
        rect = dlib.rectangle(startX, startY, endX, endY)
        face = face_aligner.align(
                96,
                image,
                rect,
                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE
        )

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        blobArray[i] = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
        boxes.append(box)

    return boxes


if __name__ == '__main__':
    main()

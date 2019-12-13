# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import glob
import torch
import net
import dlib
from align_dlib import AlignDlib
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testdir", default="test_dataset/",
                    help="path to test dataset")
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

    # load our serialized face detector from disk
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

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

    test_dir = args["testdir"]
    if not test_dir.endswith("/"):
        test_dir += "/"
    print(test_dir)

    num_images = 0
    num_correct = 0
    num_guesses = 0

    image_files = glob.glob(test_dir + '**/*.jpg', recursive=True)
    num_images = len(image_files)
    for i in range(0, len(image_files), batch_size):
        image_files_batch = image_files[i:i + batch_size]
        identities = []
        for image_file in image_files_batch:
            identities.append(os.path.basename(os.path.dirname(image_file)))
        names = recognize_batch(
            image_files_batch,
            le,
            recognizer,
            detector,
            face_cascade,
            embedder,
            face_aligner,
            args["confidence"])

        for j in range(len(image_files_batch)):
            name = names[j]
            identity = identities[j]
            if name is not None:
                num_guesses += 1
                print(name, identity)
                if name == identity:
                    num_correct += 1
    print("Final Acc: " + str(float(num_correct) / num_images))
    print("Guesses Acc: " + str(float(num_correct) / num_guesses))


def create_face_blob(image_file, face_cascade, detector, face_aligner):
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions

    image = cv2.imread(image_file)
    image = imutils.resize(image, width=600)
    (ih, iw) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # extract the confidence (i.e., probability) associated with the
    # first prediction
    confidence = detections[0, 0, 0, 2]

    min_confidence = 0
    # filter out weak detections
    if confidence > min_confidence:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, 0, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        if startX < 0 or startY < 0 or endX > iw or endY > ih:
            return None

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
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        return faceBlob
    else:
        return None


blobArray = np.zeros((batch_size, 3, 96, 96), dtype=np.float32)

def recognize_batch(image_files, le, recognizer, detector,
                    face_cascade, embedder, face_aligner, confidence):
    blobArray[:] = 0
    for i in range(len(image_files)):
        blob = create_face_blob(image_files[i], face_cascade, detector, face_aligner)
        if blob is not None:
            blobArray[i] = blob
    start = time.time()
    inputs = torch.from_numpy(blobArray).to(device)
    vec = embedder.forward(inputs).cpu().numpy()
    end = time.time()
    print("embedder: " + str(end-start))
    start = time.time()
    # perform classification to recognize the face
    predsArray = recognizer.predict_proba(vec)
    end = time.time()
    print("recognizer: " + str(end-start))
    names = []
    for i in range(len(image_files)):
        names.append(find_predictions(predsArray[i], confidence, le))
    return names


def find_predictions(preds, confidence, le):
    j = np.argmax(preds)
    proba = preds[j]
    print(proba)
    if proba > confidence:
        return le.classes_[j]
    return None


if __name__ == '__main__':
    main()

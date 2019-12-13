# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import torch
import net
import sys
import dlib
from align_dlib import AlignDlib

# To extract embeddings for the test dataset
# python3 extract_embeddings.py -e output/test_embeddings.pickle -i
# test_dataset

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", default="unprocessed_dataset",
                help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", default="output/embeddings.pickle",
                help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", default="face_detection_model",
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", default="net.pth",
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", help="draw bounding boxes", action="store_true")
args = vars(ap.parse_args())
print(args)

# load our serialized face detector from disk
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
embedder = net.model
embedder.load_state_dict(torch.load('net.pth'))
embedder.to(device)
embedder.eval()

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (ih, iw) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # analyze the first detection, since we set up the dataset to have
    # only one face per image
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, 0, 2]

    min_confidence = 0
    # filter out weak detections
    if confidence > min_confidence:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, 0, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        if startX < 0 or startY < 0 or endX > iw or endY > ih:
            continue
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
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        inputs = torch.from_numpy(faceBlob).to(device)
        vec = embedder.forward(inputs).cpu().numpy()

        # add the name of the person + corresponding face
        # embedding to their respective lists
        knownNames.append(name)
        knownEmbeddings.append(vec.flatten())
        total += 1


        if args['visualize']:
            draw_image = image.copy()
            cv2.rectangle(draw_image, (startX, startY), (endX, endY), (255,255,0), 2)
            # show the output image
            cv2.imshow("Image", face) #draw_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                sys.exit(0)



# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()

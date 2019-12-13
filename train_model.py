from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings",
	help="path to serialized db of facial embeddings",
        default="output/embeddings.pickle")
ap.add_argument("-r", "--recognizer",
        help="path to output model trained to recognize faces",
        default="output/recognizer.pickle")
ap.add_argument("-l", "--le",
	help="path to output label encoder",
        default="output/le.pickle")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=10, kernel="rbf", probability=True, gamma="scale")
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()



# necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def calculate_eye_aspect_ratio(eye):
	# calculate the euclidean distances between the two sets of
	# vertical eye landmarks in (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# calculate the euclidean distance between the two sets of
	#horizontal eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	eye_aspect_ratio = (A + B) / (2.0 * C)

	return eye_aspect_ratio

def play_alarm_sound(path):
	# play alarm when drowsiness is detected
	playsound.playsound(path)
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# eye aspect ratio threshold which indicates a blink
EYE_AR_THRESH = 0.3

#the number of consecutive frames the eye must be below
#the threshold to play the alarm
EYE_AR_CONSEC_FRAMES = 47

COUNT = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
	# frame is read and converted to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detectes faces in the grayscale frame
	rects = detector(gray, 0)

	for rect in rects:
		# determine the facial landmarks for the face region
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates
		# eye aspect ratio for the coordinates
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		left_EAR = calculate_eye_aspect_ratio(leftEye)
		right_EAR = calculate_eye_aspect_ratio(rightEye)

		# mean of the eye aspect ratio for both eyes
		eye_aspect_ratio = (left_EAR + right_EAR) / 2.0

		# visualize the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check the eye aspect ratio
		if eye_aspect_ratio < EYE_AR_THRESH:
			COUNT += 1

			if COUNT >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True

					# thread to have the alarm sound played in the background
					if args["alarm"] != "":
						t = Thread(target=play_alarm_sound,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# text on the frame
				cv2.putText(frame, "DROWSY ALERT!", (10, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, reset the counter and alarm
		else:
			COUNT = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame 
		cv2.putText(frame, "EAR: {:.2f}".format(eye_aspect_ratio), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
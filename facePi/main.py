# facePi

import argparse
import time
import os
import cv2
import numpy as np
import imutils
import rainbowhat
from flask import Flask, render_template, Response, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

#-----------------------------------------------------------------
PROTOTXT_FILE_PATH = "tensorflow_files/deploy.prototxt"
MODEL_FILE_PATH = "tensorflow_files/mask_detector.model"
CAFFEMODEL_FILE_PATH = "tensorflow_files/res10_300x300_ssd_iter_140000.caffemodel"
SCREENSHOT_EXPORT_PATH = "screenshots/"
#-----------------------------------------------------------------

app = Flask(__name__)
save_img = None
isAlarmActive = True
isViedoFeedActive = True

print("DEBUG: loading face detector model")
faceNet = cv2.dnn.readNet(PROTOTXT_FILE_PATH, CAFFEMODEL_FILE_PATH)

print("DEBUG: loading face mask detector model")
maskNet = load_model(MODEL_FILE_PATH)

print("DEBUG: activating Camera")
videoStream = VideoStream(src=0).start()

def mainVideoFeedLoop():
	global isViedoFeedActive
	global save_img
	global isAlarmActive
	while True:
		frame = videoStream.read()
		frame = cv2.flip(frame, 0)
		frame = imutils.resize(frame, width=500)
		(boxes, predictions) = detect_mask(frame, faceNet, maskNet)

		withCounter = 0
		withoutCounter = 0
		for (box, prediction) in zip(boxes, predictions):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = prediction

			if mask > withoutMask:
				label = "Thank You. Mask On."
				color = (0, 255, 0)
				withCounter+=1

			else:
				label = "No Face Mask Detected"
				color = (0, 0, 255)
				withoutCounter+=1
				if(isAlarmActive and isViedoFeedActive):
					rainbowhat.buzzer.midi_note(60, 1)

			cv2.putText(frame, label, (startX-50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		RGB_IMG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if(not isViedoFeedActive):
			RGB_IMG = cv2.imread('./static/logo.jpeg', 0) #return logo anstatt kamerabild
			rainbowhat.display.print_str('----')
		else:
			rainbowhat.display.print_str('0'+str(withCounter)+'0'+str(withoutCounter))

		save_img = RGB_IMG
		retValue, image = cv2.imencode('.jpg', RGB_IMG)

		rainbowhat.display.show()
		yield(decodeData(image.tobytes()))
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break
	cv2.destroyAllWindows()
	videoStream.stop()

def detect_mask(frame, faceNet, maskNet):
	faces = []
	boxes = []
	predictions = []

	(height, width) = frame.shape[:2] #länge breite von Bild ermitteln
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) #blob aus Bild generieren (Binary Large Objects)

	faceNet.setInput(blob)
	detections = faceNet.forward() #Neurales Netzt aus blob masken erkennen

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2] #Prozentangebe wie sicher sich das Neurale Netzt ist eine Maske erkannt zu haben

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(width - 1, endX), min(height - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			boxes.append((startX, startY, endX, endY))

	if len(faces) > 0: #wenn mindestesns 1 Gesicht gefunden wurde
		faces = np.array(faces, dtype="float32")
		predictions = maskNet.predict(faces, batch_size=32) #(　’ ‘)ﾉﾉ⌒●~*

	return (boxes, predictions)

@app.route('/')
def loadHTML():
	return render_template('index.html')

def decodeData(imageBytes):
	return (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + imageBytes + b'\r\n\r\n')

@app.route('/alarm')
def alarm():
	global isAlarmActive
	isAlarmActive = not isAlarmActive
	return 'alarm'

@app.route('/picture')
def picture():
	global save_img
	timestr = time.strftime("%Y%m%d_%H%M%S")
	cv2.imwrite(SCREENSHOT_EXPORT_PATH + timestr + '_screenshot.jpg', save_img)
	return 'picture'

@app.route('/activateViedoFeed')
def activateViedoFeed():
	global isViedoFeedActive
	isViedoFeedActive = True
	return 'activateViedoFeed'

@app.route('/disableViedoFeed')
def disableViedoFeed():
	global isViedoFeedActive
	isViedoFeedActive = False
	return 'disableViedoFeed'

@app.route('/video_feed')
def video_feed():
	return Response(mainVideoFeedLoop(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False)

# facePi

import time
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
isVideoFeedActive = True

print("DEBUG: loading face detector model") #laden des face detector models
faceNet = cv2.dnn.readNet(PROTOTXT_FILE_PATH, CAFFEMODEL_FILE_PATH)

print("DEBUG: loading face mask detector model") #laden des mask detector models
maskNet = load_model(MODEL_FILE_PATH)

print("DEBUG: activating Camera") #Kamera am Port 0 aktivieren (PiCam)
videoStream = VideoStream(src=0).start()

def mainVideoFeedLoop(): #main loop, wird ausgeführt wenn Website aufgerufen wird
	global isVideoFeedActive #Laden der globalen Variablen
	global save_img
	global isAlarmActive
	while True:
		frame = videoStream.read() #lesen des Kamerabildes, spiegeln (PiCam kopfüber montiert), Bildgröße verkleinern
		frame = cv2.flip(frame, 0)
		frame = imutils.resize(frame, width=500)
		(boxes, predictions) = detect_mask(frame, faceNet, maskNet) #Aufruf unserer Maskenerkennungsfunktion

		withCounter = 0
		withoutCounter = 0
		for (box, prediction) in zip(boxes, predictions): #loop über erkannte Gesichter
			(startX, startY, endX, endY) = box #Koordinaten extrahieren
			(mask, withoutMask) = prediction

			if mask > withoutMask: #Entscheidung mit oder ohne Maske
				label = "Thank You. Mask On."
				color = (0, 255, 0)
				withCounter+=1

			else:
				label = "No Face Mask Detected"
				color = (0, 0, 255)
				withoutCounter+=1
				if(isAlarmActive and isVideoFeedActive):
					rainbowhat.buzzer.midi_note(60, 1) #beep

			cv2.putText(frame, label, (startX-50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) #Box um Gesicht malen und Text dazuschreiben
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		RGB_IMG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Kamerabild in RGB-Bild umwandeln

		if(not isVideoFeedActive):
			RGB_IMG = cv2.imread('./static/logo.jpeg') #Kamerabild mit Logo ersetzen
			rainbowhat.display.print_str('----')
		else:
			rainbowhat.display.print_str('0'+str(withCounter)+'0'+str(withoutCounter)) #Anzeige erkannte Personen

		save_img = RGB_IMG #Kamerabild in globaler Variable speichern
		retValue, image = cv2.imencode('.jpg', RGB_IMG) #Kamerabild in jpg umwandeln

		rainbowhat.display.show()
		yield(decodeData(image.tobytes())) #dauerhaftes return des Kamerabildes

def detect_mask(frame, faceNet, maskNet):
	faces = []
	boxes = []
	predictions = []

	(height, width) = frame.shape[:2] #Länge, Breite von Bild ermitteln
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) #blob aus Bild generieren (Binary Large Objects)

	faceNet.setInput(blob)
	detections = faceNet.forward() #blob an das Netz schicken um Gesichter zu erkennen

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2] #Prozentangebe wie sicher sich das neurale Netz ist eine Maske erkannt zu haben

		if confidence > 0.5: #nur anzeigen wenn Zuversicht > 50%
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height]) #Berechnung der Koordinaten für die Box um das Gesicht
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY)) #sicherstellen dass Box auf das Bild passt
			(endX, endY) = (min(width - 1, endX), min(height - 1, endY))

			face = frame[startY:endY, startX:endX] #erkanntes Gesicht für die Maskenerkennung vorbereiten
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			boxes.append((startX, startY, endX, endY))

	if len(faces) > 0: #wenn mindestesns ein Gesicht gefunden wurde
		faces = np.array(faces, dtype="float32")
		predictions = maskNet.predict(faces, batch_size=32) #(　’ ‘)ﾉﾉ⌒●~* magic

	return (boxes, predictions)

@app.route('/') #default Route
def loadHTML():
	return render_template('index.html')

def decodeData(imageBytes): #Bilder für Übertrag dekodieren
	return (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + imageBytes + b'\r\n\r\n')

@app.route('/alarm')
def alarm():
	global isAlarmActive
	isAlarmActive = not isAlarmActive
	return 'alarm'

@app.route('/picture')
def picture():
	global save_img
	timestr = time.strftime("%Y%m%d_%H%M%S") #Zeit + Datum
	cv2.imwrite(SCREENSHOT_EXPORT_PATH + timestr + '_screenshot.jpg', save_img)
	return 'picture'

@app.route('/activateViedeoFeed')
def activateViedeoFeed():
	global isVideoFeedActive
	isVideoFeedActive = True
	return 'activateVideoFeed'

@app.route('/disableVideoFeed')
def disableVideoFeed():
	global isVideoFeedActive
	isVideoFeedActive = False
	return 'disableVideoFeed'

@app.route('/video_feed')
def video_feed():
	return Response(mainVideoFeedLoop(), mimetype='multipart/x-mixed-replace; boundary=frame') #http-header für dynamische Websiteupdates

if __name__ == '__main__': #starten des Flaskservers
	app.run(host='0.0.0.0', debug=False)

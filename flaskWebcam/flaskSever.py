from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
import time
import numpy as np
import cv2

app = Flask(__name__)

webcam = VideoStream().start()
webcam.stream.set(3, 320)
webcam.stream.set(4, 240)
time.sleep(2.0)

@app.route('/')
def loadHTML():
    return render_template('coreSite.html')
    
def decodeData(imageBytes):
    return (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + imageBytes + b'\r\n\r\n')

def generateCameraFrame(camera):
    #get camera frame
    while True:
        retValue, image = cv2.imencode('.jpg', camera.read())
        if(retValue):
            yield(decodeData(image.tobytes()))

@app.route('/video_feed')
def video_feed():
    return Response(generateCameraFrame(webcam), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
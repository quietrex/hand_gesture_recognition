# main.py# import the necessary packages
from flask import flash, Flask, render_template, Response, make_response
from camera import VideoCamera

import logging

app = Flask(__name__)

prediction_text = ""

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
    
def gen(camera):
    while True:
        #get camera frames
        frame = camera.get_frame()
        predictions = camera.predict()
        
        if predictions != None:
            # print(predictions)
            prediction_text = predictions

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
               
@app.route('/')
def generatePredMetrics():
    # read text file and output
    with open("prediction.txt", "r") as file:
        first_line = file.readline()
        for last_line in file:
            pass

    # play / stop the music
    return render_template('index.html', name=last_line)

@app.route('/metrics')
def metrics():
    response = make_response(generatePredMetrics(), 200)
    response.mimetype = "text/plain"
    return response

@app.route('/metrics2')
def metrics2():
    return generatePredMetrics()

@app.route('/audio_feed')
def audio_feed():
    def generate():
        with open("hello.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
import os
import re
import cv2
import numpy as np
from PIL import Image
from flask_cors import CORS
from flask import Flask, Response
import matplotlib.pyplot as plt
from pytesseract import Output
from pytesseract import pytesseract
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__,template_folder='templates')
CORS(app)

def logoDetect(file_path):
    img = cv2.imread(file_path)
    h1 = int(img.shape[0] * 0.1)
    h2 = int(img.shape[0] * 0.9)
    w1 = int(img.shape[1] * 0.8)
    w2 = int(img.shape[1])
    img = img[0:h1, w1:w2]  # 1400:1600,0:100]

    text = pytesseract.image_to_string(img)

    if text.find("Reddy") != -1:
        return True
    else:
        return False

def textCount(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)

    lis = re.split(r" |\n", text)
    lis = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in lis]
    lis = [i for i in lis if i != ""]
    textno = len(lis)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    textArea = 0
    fontSize = []
    fontSizeUnique = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        textArea += (w*h)
        fontSize.append(h)

    for x in fontSize:
        if x not in fontSizeUnique:
            fontSizeUnique.append(x)

    print(fontSizeUnique)
    finalText = text.replace("\n", " ")
    finalText = finalText.replace("\x0c", " ")
    textCntAns = [textno, textArea, finalText]
    return textCntAns

def objectDetect(file_path):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.imread(file_path)

    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    obj = 0
    imgArea = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            obj+=1
            imgArea+=(w*h)
    # print(len(boxes))
    if imgArea == 0:
        height, width = img.shape[:2]
        imgArea = height*width
    objDectAns = [obj,imgArea]
    print(objDectAns)
    return objDectAns

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("uploading.html")

@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        TextCount = textCount(file_path) #{count,area}
        objCount = objectDetect(file_path) #{count,area}

        if TextCount[0] == 0:
            compAreaToText = -1
        else:
            compAreaToText = objCount[1]/TextCount[1]

        logoPresent = logoDetect(file_path)

        output = {
            'textCount':TextCount[0],
            'objectCount':objCount[0],
            'logoPresent':logoPresent,
            'text':TextCount[2],
            'compAreaToTextComp':compAreaToText
        }

        return render_template('uploading.html', output=output)
    return None


if __name__ == "__main__":
    app.run(debug=True)
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('C:\\Users\\pc\\Desktop\\image processing project\\BrainTumor16EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "Based on the results of your provided image, I can reassure you that you do not have a brain tumor."
    elif classNo == 1:
        return "I regret to inform you that the provided image indicate the presence of a brain tumor."

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    
    # Use predict instead of predict_classes
    prediction = model.predict(input_img)
    
    # Find the index of the maximum value in the prediction array
    result = np.argmax(prediction, axis=1)
    
    return result[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Specify the directory where you want to save the uploaded files
        upload_directory = os.path.join(os.path.dirname(__file__), 'uploads')

        # Check if the directory exists, if not, create it
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)

        file_path = os.path.join(upload_directory, secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value) 
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)

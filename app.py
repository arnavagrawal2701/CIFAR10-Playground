from flask import Flask, jsonify, request, render_template, redirect
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io

model = keras.models.load_model('cifar10.h5')

app = Flask(__name__)

classes=["Airplane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

@app.route('/')
def front():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST' and 'file' in request.files:
        image_file = request.files['file']

        image = preprocess(image_file)

        prediction = model.predict(image)

        response = {'result': classes[np.argmax(prediction)]}

        return jsonify(response)

def preprocess(image_file):
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(image_file.read()), target_size=(32,32))
    
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = image/255.0

    image = np.expand_dims(image, axis=0)

    return image

if __name__ == "__main__":
    app.run(debug=True)
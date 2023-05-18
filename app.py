import numpy as np
import pandas as pd
import pyttsx3
from flask import Flask, render_template, request, jsonify
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
app = Flask(__name__)

loc = "Crop_recommendation.csv"
data = pd.read_csv(loc)

NITROGEN = list(data["N"])
PHOSPHORUS = list(data["P"])
POTASSIUM = list(data["K"])
TEMPERATURE = list(data["temperature"])
HUMIDITY = list(data["humidity"])
PH = list(data["ph"])
RAINFALL = list(data["rainfall"])

features = np.array([NITROGEN, PHOSPHORUS, POTASSIUM,
                    TEMPERATURE, HUMIDITY, PH, RAINFALL])

le = preprocessing.LabelEncoder()
crop = le.fit_transform(list(data["label"]))

features = features.transpose()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, crop)




@app.route('/')
def home():
    """Render homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Invalid Content-Type'}), 400
    data = request.json
    nitrogen_content = data['nitrogen_content']
    phosphorus_content = data['phosphorus_content']
    potassium_content = data['potassium_content']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    test_data = np.array([nitrogen_content, phosphorus_content,
                          potassium_content, temperature, humidity, ph, rainfall]).reshape(1, -1)
    predicted_crop = model.predict(test_data)[0]
    predicted_crop_name = le.inverse_transform([predicted_crop])[0]

    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    engine.setProperty('voice', voices[0].id)
    engine.say(
        f"According to the data you provided, the best crop to grow is {predicted_crop_name}")
    engine.runAndWait()

    result = {
        "prediction_text": f"The best crop to grow is {predicted_crop_name}"
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

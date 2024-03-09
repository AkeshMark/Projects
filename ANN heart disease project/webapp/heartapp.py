from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.models import load_model

model = load_model('model/model.h5')
scaler_data = joblib.load('model/scaler_data.sav')
scaler_target = joblib.load('model/scaler_target.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('patient_details.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    form_data = request.form

    name = form_data['name']
    age = float(form_data['age'])
    sex = float(form_data['sex'])
    cp = float(form_data['cp'])
    trestbps = float(form_data['trestbps'])
    chol = float(form_data['chol'])
    restecg = float(form_data['restecg'])
    thalach = float(form_data['thalach'])
    exang = float(form_data['exang'])
    oldpeak = float(form_data['oldpeak'])
    slope = float(form_data['slope'])
    ca = float(form_data['ca'])
    thal = float(form_data['thal'])

    test_data = np.array([age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    test_data = scaler_data.transform(test_data)

    prediction = model.predict(test_data)
    prediction = scaler_target.inverse_transform(prediction)

    result_dict = {"name": name, "risk": round(prediction[0][0],2)*100}

    return render_template('patient_results.html', results=result_dict)

app.run(debug=True)

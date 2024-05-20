from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

with open('insurance.pkl', 'rb') as file:
    model = pickle.load(file)

with open('gbmodel.pkl', 'rb') as file:
    gb = pickle.load(file)


sex_map = {'male': 0, 'female': 1}
smoker_map = {'yes': 1, 'no': 0}
region_map = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}


@app.route('/hello')
def hello():
    return "Hello World!"


@app.route('/api/predict_premium', methods=['POST'])
def predict_premium():
    data = request.get_json()
    if any(value is None or value == '' for value in data.values()):
        return jsonify({'msg': 'All fields are required'}), 400

    age = int(data.get('age', 0))
    sex = sex_map.get(data.get('sex'))
    smoker = smoker_map.get(data.get('smoker'))
    region = region_map.get(data.get('region'))
    bmi = float(data.get('bmi', 0.0))
    children = int(data.get('children', 0))

    prediction = model.predict([[age, sex, bmi, children, smoker, region]])

    prediction_value = float(prediction[0])
    return jsonify({'prediction': prediction_value}), 200


@app.route('/api/predict_fitness', methods=['POST'])
def predict_fitness():
    data = request.get_json()
    if any(value is None or value == '' for value in data.values()):
        return jsonify({'msg': 'All fields are required'}), 400

    fitness_model = gb['model']
    scaler = gb['scaler']

    mood = data.get('mood')
    step_count = int(data.get('step_count', 0))
    calories_burned = int(data.get('calories_burned', 0))
    hours_sleep = int(data.get('hours_sleep', 0))
    weight_kg = int(data.get('weight_kg', 0))

    mood_neutral = 0
    mood_happy = 0
    if mood == "neutral":
        mood_neutral = 1
    elif mood == "happy":
        mood_happy = 1

    input_data = np.array(
        [[mood_neutral, mood_happy, step_count, calories_burned, hours_sleep, weight_kg]])
    input_data_scaled = scaler.transform(input_data)
    prediction = fitness_model.predict(input_data_scaled)
    prediction_val = prediction.tolist()[0]

    if prediction_val == 1:
        prediction_val = 'Active'
    elif prediction_val == 0:
        prediction_val = 'Non-Active'

    return jsonify({'prediction': prediction_val}), 200


if __name__ == '__main__':
    app.run(debug=True)

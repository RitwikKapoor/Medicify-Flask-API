from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

with open('insurance.pkl', 'rb') as file:
    model = pickle.load(file)

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

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('insurance.pkl', 'rb') as file:
    model = pickle.load(file)

sex_map = {'male': 0, 'female': 1}
smoker_map = {'yes': 1, 'no': 0}
region_map = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}

@app.route('/hello')
def hello():
    return "Hello World!"

@app.route('/predict_premium', methods=['POST'])
def predict():
    data = request.get_json()

    age = data['age']
    sex = sex_map[data['sex']]
    bmi = data['bmi']
    children = data['children']
    smoker = smoker_map[data['smoker']]
    region = region_map[data['region']]

    prediction = model.predict([[age, sex, bmi, children, smoker, region]])

    prediction_value = float(prediction[0])

    return jsonify({'prediction': prediction_value})

if __name__ == '__main__':
    app.run(debug=True)
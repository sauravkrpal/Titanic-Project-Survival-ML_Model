from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Titanic Survival Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input
        data = request.get_json(force=True)

        # Extract input features
        pclass = int(data['Pclass'])
        sex = data['Sex']
        age = float(data['Age'])
        embarked = data['Embarked']
        sibsp = int(data['SibSp'])
        parch = int(data['Parch'])
        fare = float(data['Fare'])

        # Feature engineering
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        fare_log = np.log1p(fare)  # Using log1p for numerical stability
        # Prepare DataFrame for prediction
        input_df = pd.DataFrame([{
            'Sex': sex,
            'Pclass': pclass,
            'Age': age,
            'Embarked': embarked,
            'Fare_log': fare_log,
            'FamilySize': family_size,
            'IsAlone': is_alone
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Survived" if prediction == 1 else "Did not survive"

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
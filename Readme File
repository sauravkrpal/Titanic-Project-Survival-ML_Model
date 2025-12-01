# 🚢 Titanic Survival Prediction API

A machine‑learning powered REST API that predicts whether a passenger would have survived the Titanic disaster. Built with **Python**, **Scikit‑Learn**, and **Flask**, using a full preprocessing + RandomForest pipeline.

---

## 🔥 Features

* End‑to‑end ML pipeline (scaling + encoding + model).
* Pre‑engineered features (FamilySize, IsAlone, Fare Log).
* Clean JSON‑based REST API.
* Easy to deploy on Render, Railway, AWS, or Cloud Run.
* Model stored using `joblib`.

---

## 📦 Project Structure

```
├── app.py                # Flask API
├── titanic_model.pkl     # Trained ML pipeline
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🧠 Model Training Summary

The ML model was trained on the Titanic dataset from DataScienceDojo using:

* **RandomForestClassifier**
* **Feature Engineering:**

  * FamilySize = SibSp + Parch + 1
  * IsAlone (binary)
  * Log‑transformed Fare
* **Preprocessing:**

  * StandardScaler → numeric features
  * OneHotEncoder → categorical features

The final model is saved as:

```
titanic_model.pkl
```

---

## 🚀 API Setup & Run

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Run the API**

```
python app.py
```

Your API will start at:

```
http://127.0.0.1:5000/
```

---

## 🏁 API Endpoints

### **GET /**

Health‑check endpoint.

```
{
  "message": "Titanic Survival Prediction API is running!"
}
```

### **POST /predict**

Send passenger details to get a survival prediction.

#### 🔹 **Sample JSON Request**

```
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "Embarked": "S",
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25
}
```

#### 🔹 **Sample JSON Response**

```
{
  "prediction": 0,
  "result": "Did not survive"
}
```

---

## 🔧 How Prediction Works

The API:

1. Reads JSON input.
2. Computes engineered features:

   * FamilySize
   * IsAlone
   * Fare_log
3. Passes the data into the saved ML pipeline.
4. Returns `0/1` and the human‑readable result.

---

## ☁️ Deployment Options

You can deploy this API on:

* Render (Free)
* Railway
* AWS EC2
* AWS Lambda + API Gateway
* Google Cloud Run
* Azure App Service

If you want deployment files (Dockerfile, render.yaml, etc.), tell me.

---

## 📄 Requirements

Example `requirements.txt`:

```
Flask
numpy
pandas
scikit-learn
joblib
```

---

## 📝 Author

Built by **Saurav Kumar Pal**, Data Scientist & ML Developer.

If you'd like, I can also create:

* A clean API logo
* A Streamlit UI for the model
* Deployment‑ready Docker setup
* A portfolio write‑up

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import mysql.connector

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Swami@28",
    database="pcod_db"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    cycle = int(request.form["cycle"])
    acne = int(request.form["acne"])
    hair_growth = int(request.form["hair_growth"])

    # Input features
    features = np.array([
        [age, bmi, cycle, acne, hair_growth]
    ])

    # Prediction
    pred = model.predict(features)[0]

    # Probability of class 1
    prob = model.predict_proba(features)[0][1]

    # Risk level
    if prob >= 0.70:
        result = "High Risk"
    elif prob >= 0.40:
        result = "Medium Risk"
    else:
        result = "Low Risk"

    # Save result
    cursor = db.cursor()

    cursor.execute(
        """
        INSERT INTO records
        (age, bmi, cycle, acne, hair, result)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            age,
            bmi,
            cycle,
            acne,
            hair_growth,
            result
        )
    )

    db.commit()
    cursor.close()

    return render_template(
        "index.html",
        prediction_text=result,
        risk_score=round(prob * 100, 2)
    )


@app.route("/data")
def data():

    cursor = db.cursor()

    cursor.execute(
        "SELECT id, result FROM records"
    )

    rows = cursor.fetchall()

    cursor.close()

    labels = [str(row[0]) for row in rows]

    values = [
        1 if row[1] == "High Risk"
        else 0.5 if row[1] == "Medium Risk"
        else 0
        for row in rows
    ]

    return jsonify({
        "labels": labels,
        "values": values
    })


if __name__ == "__main__":
    app.run(debug=True)
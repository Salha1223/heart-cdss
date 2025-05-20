from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")
feature_names = model.feature_names_in_

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        # نخزن البيانات مؤقتًا في session أو نرسلها مباشرة
        return redirect(url_for("predict", **request.form))
    return render_template("form.html")

@app.route("/predict")
def predict():
    try:
        # نحول البيانات من الاستعلام إلى نوع مناسب
        form = request.args
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = [0] * len(feature_names)

        input_data.at[0, "Age"] = int(form["age"])
        input_data.at[0, "RestingBP"] = int(form["resting_bp"])
        input_data.at[0, "Cholesterol"] = int(form["cholesterol"])
        input_data.at[0, "FastingBS"] = int(form["fasting_bs"])
        input_data.at[0, "MaxHR"] = int(form["max_hr"])
        input_data.at[0, "Oldpeak"] = float(form["oldpeak"])

        if "Sex_M" in input_data.columns and form["sex"].upper() == "M":
            input_data.at[0, "Sex_M"] = 1
        if f"ChestPainType_{form['cp']}" in input_data.columns:
            input_data.at[0, f"ChestPainType_{form['cp']}"] = 1
        if f"RestingECG_{form['resting_ecg']}" in input_data.columns:
            input_data.at[0, f"RestingECG_{form['resting_ecg']}"] = 1
        if "ExerciseAngina_Y" in input_data.columns and form["exercise_angina"].upper() == "Y":
            input_data.at[0, "ExerciseAngina_Y"] = 1
        if f"ST_Slope_{form['st_slope']}" in input_data.columns:
            input_data.at[0, f"ST_Slope_{form['st_slope']}"] = 1

        prediction = model.predict(input_data)[0]
        result = "❤️ تم اكتشاف مرض القلب!" if prediction == 1 else "✅ لا يوجد مرض قلب"
        return render_template("result.html", result=result)
    except Exception as e:
        return f"حدث خطأ: {e}"

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]]
        prediction = model.predict([np.array(features)])
        result = "🚨 يوجد احتمال لمرض القلب" if prediction[0] == 1 else "✅ لا يوجد مرض قلب"
        recommendation = "ينصح بتحويل الحالة لمزيد من الفحوصات" if prediction[0] == 1 else "لا حاجة لتدخل إضافي حالياً"
        return render_template('index.html', prediction_text=result, recommendation_text=recommendation)
    except Exception as e:
        return f"خطأ في الإدخال: {e}"

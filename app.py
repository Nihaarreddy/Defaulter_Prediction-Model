from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load your trained model
model = joblib.load('best_final_model.pkl')

# Feature names for form and CSV
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
            'Education', 'Mortgage', 'Securities Account', 
            'CD Account', 'Online', 'CreditCard']

@app.route('/')
def home():
    return render_template('base.html')  # Or your main landing page

@app.route('/predict', methods=['GET', 'POST'])
def single_predict():
    prediction = None
    probability = None
    if request.method == 'POST':
        input_data = [float(request.form[feature]) for feature in features]
        pred = model.predict([input_data])[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([input_data])[0][1]
            probability = f"{proba*100:.2f}%"
        prediction = "Defaulter" if pred == 1 else "Non-Defaulter"
        return render_template('index_single.html', prediction=prediction, probability=probability)
    return render_template('index_single.html', prediction=prediction, probability=probability)

@app.route('/predictfile', methods=['GET', 'POST'])
def batch_predict():
    message = None
    if request.method == 'POST':
        file = request.files['file']
        user = request.form.get('user')
        pw = request.form.get('pw')
        db = request.form.get('db')
        if not file:
            message = "No file uploaded!"
            return render_template('index_file.html', message=message)
        try:
            df = pd.read_csv(file)
            predictions = model.predict(df[features])
            df['Prediction'] = ['Defaulter' if p == 1 else 'Non-Defaulter' for p in predictions]
            output = io.BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='predictions.csv'
            )
        except Exception as e:
            message = f"Error processing file: {e}"
            return render_template('index_file.html', message=message)
    return render_template('index_file.html')

if __name__ == '__main__':
    app.run(debug=True)

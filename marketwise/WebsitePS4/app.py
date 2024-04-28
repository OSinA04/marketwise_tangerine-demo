from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('PS4/model.pkl')

# Load the scaler
scaler = joblib.load('PS4/scaler.pkl')

# import dataset
df = pd.read_csv('PS4/cleaned_data.csv')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ps4', methods=['GET', 'POST'])
def ps4():
    if request.method == 'POST':
        # 'Industry', 'Company Size', 'Package Type', 'Sponsorship Amount','Duration (months)', 'Marketing Activities', 'Event Type','Event Month', 'Attendee Demographics', 'ROI Metrics', 'Feedback'
        input_data = {
            'Industry': int(request.form['industry']),
            'Company Size': int(request.form['company_size']),
            'Package Type': int(request.form['package_type']),
            'Sponsorship Amount': int(request.form['sponsorship_amount']),
            'Duration (Months)': int(request.form['duration']),
            'Marketing Activities': int(request.form['marketing_activities']),
            'Event Type': int(request.form['event_type']),
            'Event Month': int(request.form['event_month']),
            'Attendee Demographics': int(request.form['attendee_demographics']),
            'Feedback': int(request.form['feedback'])
        }

        df.loc[len(df)] = input_data

        input_data = scaler.transform(np.array(list(input_data.values())).reshape(1, -1))

        prediction = model.predict(input_data)
        return render_template('feedback.html', prediction=round(prediction[0], 2))
        
    return render_template('ps4.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_data = {
            'actual_roi_metrics': float(request.form['actual_roi_metrics']),
        }

        df.loc[len(df) - 1, 'ROI Metrics'] = feedback_data['actual_roi_metrics']
        df.to_csv('PS4/cleaned_data.csv', index=False)

        print(df.tail(5))

        return redirect('/')
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
    # Save the model
    joblib.dump(model, 'PS4/model.pkl')

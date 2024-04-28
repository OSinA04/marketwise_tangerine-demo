from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('PS3/linear_regression_model.pkl')

# Load the scaler
scaler = joblib.load('PS3/scaler.pkl')

# import dataset
df = pd.read_csv('PS3/events_dataset_numerical.csv')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ps3', methods=['GET', 'POST'])
def ps3():
    if request.method == 'POST':
        input_data = {
            'Event_Type': int(request.form['event_type']),
            'Location': int(request.form['location']),
            'Ticket_Price': float(request.form['ticket_price']),
            'Weather_Condition': int(request.form['weather_condition']),
            'Day_of_Week': int(request.form['day_of_week']),
            'Month': int(request.form['month']),
        }

        df.loc[len(df)] = input_data

        input_data = scaler.transform(np.array(list(input_data.values())).reshape(1, -1))

        prediction = model.predict(input_data)
        return render_template('feedback.html', prediction=round(prediction[0]))
        
    return render_template('ps3.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_data = {
            'actual_attendance_count': int(request.form['actual_attendance_count']),
        }

        df.loc[len(df) - 1, 'Attendance_Count'] = feedback_data['actual_attendance_count']
        df.to_csv('PS3/events_dataset_numerical.csv', index=False)

        print(df.tail(5))

        new_X = df.drop(['Attendance_Count'], axis=1).tail(1)
        new_y = df['Attendance_Count'].tail(1)

        model.fit(new_X, new_y)

        return redirect('/')
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
    # Save the model
    joblib.dump(model, 'PS3/linear_regression_model.pkl')
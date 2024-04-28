from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_data():
    df_exhibitors = pd.read_csv('PS1/exhibitors_tokenized.csv')
    df_visitors = pd.read_csv('PS1/visitors_tokenized.csv')
    df_attended_events = pd.read_csv('PS1/attended_events.csv')

    return df_exhibitors, df_visitors, df_attended_events

df_exhibitors, df_visitors, df_attended_events = load_data()

def get_cosine_similarity():
    # convert tokenized data to string
    tfidf = TfidfVectorizer()
    tfidf_matrix_exhibitors = tfidf.fit_transform(df_exhibitors['combined'])
    tfidf_matrix_visitors = tfidf.transform(df_visitors['combined'])

    # get cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix_exhibitors, tfidf_matrix_visitors)
    cosine_sim_exhibitors = cosine_similarity(tfidf_matrix_exhibitors)
    return cosine_sim, cosine_sim_exhibitors

temp_exhibitors = pd.read_csv('PS1/exhibitors.csv')
company_names = temp_exhibitors['company_name']
company_repNames = temp_exhibitors['company_repName']
company_repMobileNumbers = temp_exhibitors['mobile_no']

cosine_sim, cosine_sim_exhibitors = get_cosine_similarity()

visitor_attended = dict()
for index, row in df_attended_events.iterrows():
    if row['visitor'] in visitor_attended:
        visitor_attended[row['visitor']].append((row['exhibitor'], row['rating']))
    else:
        visitor_attended[row['visitor']] = [(row['exhibitor'], row['rating'])]

def get_recommendations(visitor_id, top_n=5):
    recs = list()
    recommendations = list(enumerate(cosine_sim[:, visitor_id]))
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommendations = recommendations[:top_n]
    for recommendation in recommendations:
        recs.append((
            company_names[recommendation[0]],
            company_repNames[recommendation[0]],
            company_repMobileNumbers[recommendation[0]],
            recommendation[0]))
    
    if visitor_id in df_attended_events['visitor'].values:
        temp_recs = list()
        for exhibitor, rating in visitor_attended[visitor_id]:
            recommendations = list(enumerate(cosine_sim_exhibitors[int(exhibitor)]))
            recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
            recommendations = recommendations[:2]
            for recommendation in recommendations:
                temp_recs.append((
                    company_names[recommendation[0]],
                    company_repNames[recommendation[0]],
                    company_repMobileNumbers[recommendation[0]],
                    recommendation[0],
                    rating))
        temp_recs.sort(key=lambda x: x[4], reverse=True)
        temp_recs = [rec[:-1] for rec in temp_recs]
        # insert temp_recs into recs at even intervals
        recs = [recs[i//2] if i % 2 == 0 else temp_recs[i//2] for i in range(len(recs) + len(temp_recs))]

    return recs
        

def tokenize_data(data):
    return (data['profession'].lower() + ' ' + data['city'].lower() + ' ' + data['state'].lower())

# Define a function to save feedback data to CSV
def save_to_csv(data):
    # insert into attended_events data
    df = [data['visitor'], data['exhibitor'], data['rating']]
    # append data to df_attended_events
    df_attended_events.loc[len(df_attended_events)] = df
    df_attended_events.to_csv('PS1/attended_events.csv', index=False)

# Check if CSV file exists to determine if header should be written
def df_exists():
    try:
        pd.read_csv('attended_events.csv')
        return True
    except FileNotFoundError:
        return False

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ps1', methods=['GET', 'POST'])
def ps1():
    global df_visitors, cosine_sim, cosine_sim_exhibitors
    if request.method == 'POST':
        input_data = {
            'profession': request.form['profession'],
            'city': request.form['city'],
            'state': request.form['state']
        }
        input_data = tokenize_data(input_data)
        # append input_data to df_visitors
        df_visitors.loc[len(df_visitors)] = input_data
        print(df_visitors.tail(5))
        print(len(df_visitors))
        # find cosine similarity between input_data and exhibitors
        cosine_sim, cosine_sim_exhibitors = get_cosine_similarity()

        recommendations = get_recommendations(len(df_visitors) - 1)
        # send recommendations to feedback page
        return render_template('feedback.html', recommendations=recommendations, visitor_id=len(df_visitors) - 1)
    return render_template('ps1.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_data = {
            'visitor': request.form['visitor'],
            'exhibitor': request.form['exhibitor'],
            'rating': request.form['rating']
        }
        save_to_csv(feedback_data)
        return redirect('/')
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
    df_visitors.to_csv('PS1/visitors_tokenized.csv', index=False)

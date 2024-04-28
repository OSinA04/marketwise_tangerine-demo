from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///leads.db'
db = SQLAlchemy(app)

# Initialize NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Lead model
class Lead(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    job_title = db.Column(db.String(100))
    company = db.Column(db.String(100))
    interest = db.Column(db.String(100))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        job_title = request.form['job_title']
        company = request.form['company']
        interest = request.form['interest']

        # Perform NLP processing
        job_title_keywords = extract_keywords(job_title)
        company_keywords = extract_keywords(company)
        interest_keywords = extract_keywords(interest)

        # Create a new lead object
        new_lead = Lead(name=name, email=email, job_title=job_title_keywords, company=company_keywords, interest = interest_keywords)

        # Add the lead to the database
        db.session.add(new_lead)
        db.session.commit()

        return redirect(url_for('thank_you', name=name))
    
    return render_template('index.html')

@app.route('/thank_you/<name>')
def thank_you(name):
    return render_template('thank_you.html', name=name)

def extract_keywords(text):
    # Tokenize and tag words
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)

    # Extract nouns and adjectives as keywords
    keywords = [word for word, tag in tagged_words if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    return ' '.join(keywords)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

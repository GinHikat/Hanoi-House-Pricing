from flask import *
import numpy as np
import pandas as pd
import pickle
import math
import os
import csv

application = Flask(__name__)
application.secret_key = 'ml1_midterm'

app = application
with open('artifacts/processor.pkl', 'rb') as file:
    processor = pickle.load(file)
    
with open('artifacts/model.pkl', 'rb') as file:
    model = pickle.load(file)

class Data:
    def __init__(self, area: float, bedrooms: int, property_type: str, 
                 furniture: str, legal_status: str, distance_to_center: float, bathrooms: int, floors: int, name: str, age: int, review: str, score: int):
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.floors = floors
        self.property_type = property_type
        self.furniture = furniture
        self.legal_status = legal_status
        self.distance_to_center = distance_to_center
        self.name = name
        self.age = age
        self.review = review
        self.score = score

    def get_user(self):
        data = {
            'username': self.name,
            'age': self.age
        }
        return data
    
    def get_review(self):
        return self.review
    
    def get_data(self):
        data_input = {
            'area': [self.area],
            'bedrooms': [self.bedrooms],
            'bathrooms': [self.bathrooms],
            'floors': [self.floors],
            'property_type': [self.property_type],
            'furniture': [self.furniture],
            'legal_status': [self.legal_status],
            'distance_to_center': [self.distance_to_center]
        }
        return pd.DataFrame(data_input)

@app.template_filter('comma')
def comma_format(value):
    return "{:,}".format(value)

@app.template_filter('exp')
def exp_filter(value):
    try:
        return math.exp(float(value))
    except:
        return value

@app.route('/', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')

        session['name'] = name
        session['age'] = age

        file_path = 'data/user_data.csv'
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: #Check if file exists
                writer.writerow(['Name', 'Age', 'Review'])
            writer.writerow([name, age, '']) #Keep review empty for later

        return redirect(url_for('predict'))

    return render_template('home.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        data = Data(
            area=float(request.form.get('area') or 0),
            bedrooms=int(request.form.get('bedrooms') or 0),
            bathrooms=int(request.form.get('bathrooms') or 0),
            floors=int(request.form.get('floors') or 0),
            property_type=request.form.get('property_type') or 'other',
            furniture=request.form.get('furniture') or 'other',
            legal_status=request.form.get('legal_status') or 'other',
            distance_to_center=float(request.form.get('distance_to_center') or 0),
            name = '',
            age = 0,
            review = '',
            score = 0
        )
        
        pred = data.get_data()
        
        pred = pd.DataFrame(processor.transform(pred))
        
        result = model.predict(pred)
        
        return render_template('result.html', result = result)
    
@app.route('/submit-review', methods=['POST'])
def submit_review():
    review = request.form.get('review', '')
    score = request.form.get('score', '')
    name = session.get('name')
    age = session.get('age')

    if name and age:
        try:
            df = pd.read_csv('data/user_data.csv')
            mask = (df['Name'] == name) & (df['Age'] == int(age))
            df.loc[mask, 'Review'] = review
            df.loc[mask, 'Score'] = score
            df.to_csv('data/user_data.csv', index=False)
        except FileNotFoundError:
            df = pd.DataFrame([{'Name': name, 'Age': age, 'Review': review, 'Score': score}])
            df.to_csv('data/user_data.csv', index=False)

    return redirect(url_for('welcome'))
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000, debug=True)   



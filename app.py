from flask import *
import numpy as np
import pandas as pd
import pickle
import math
import os
import csv

from main.pipelines.full_process import Final
from main.pipelines.auto_predict import Data, Prediction

application = Flask(__name__)
application.secret_key = 'ml1_midterm'
app = application

#Auto execute all previous steps
final_executor = Final()
final_executor.all_step()


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

        file_path = 'user_data.csv'
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: #Check if file exists
                writer.writerow(['Name', 'Age', 'Review','Score'])
            writer.writerow([name, age, '', '']) #Keep review and score empty for later

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
        
        predictor = Prediction()
        result = predictor.init_predict(pred)
        
        return render_template('result.html', result = result)
    
@app.route('/submit-review', methods=['POST'])
def submit_review():
    review = request.form.get('review', '')
    score = request.form.get('score', '')
    name = session.get('name')
    age = session.get('age')

    if name and age:
        try:
            df = pd.read_csv('user_data.csv')
            mask = (df['Name'] == name) & (df['Age'] == int(age))
            df.loc[mask, 'Review'] = review
            df.loc[mask, 'Score'] = score
            df.to_csv('user_data.csv', index=False)
        except FileNotFoundError:
            df = pd.DataFrame([{'Name': name, 'Age': age, 'Review': review, 'Score': score}])
            df.to_csv('user_data.csv', index=False)

    return redirect(url_for('welcome'))
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000, debug=True)   



from flask import Flask, render_template, request
import sys
import pathlib
from pycaret.regression import *
import pandas as pd

if sys.platform == "win32":
  temp = pathlib.PosixPath
  pathlib.PosixPath = pathlib.WindowsPath
  del temp


app = Flask(__name__)

insurance_model = load_model('./model/insurance_gbr_model') 

# Route for the insurance form page (GET request)
@app.route('/test', methods=['GET'])
def test():
    return "hello world"


# Route for the insurance form page (GET request)
@app.route('/', methods=['GET'])
def home():
    return render_template('insurance.html')


# Route for form submission (POST request)
@app.route('/insurance_predict', methods=['POST'])
def process_form():
    data = {}
    data['age'] = [request.form.get('age')]
    data['sex'] = [request.form.get('sex')]
    data['bmi'] = [request.form.get('bmi')]
    data['children'] = [request.form.get('children')]
    data['smoker'] = [request.form.get('smoker')]
    data['region'] = [request.form.get('region')]
    data = pd.DataFrame(data)
    result = predict_model(insurance_model, data=data)  
    return render_template('result.html', result=result.values[0][6])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

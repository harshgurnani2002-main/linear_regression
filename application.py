from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

## import ridge regressor and standard sclaer 

ridge_mmodel=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata/', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        region = float(request.form['Region'])
        classes = float(request.form['Classes']) 

        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, region,classes]])
        scaled_data = standard_scaler.transform(input_data)
        result = ridge_mmodel.predict(scaled_data)

        return render_template('home.html', results=f"Predicted FWI: {result[0]:.4f}")
    else:
        return render_template('home.html')




if __name__=='__main__':
    app.run(host='0.0.0.0')

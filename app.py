# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from flask import Flask,render_template,url_for,request


app=Flask(__name__)
model = pickle.load(open('prediction-of-distance.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0][0],2)
    return render_template('index.html',prediction_text="The predicted distance is {}".format(output))
if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)




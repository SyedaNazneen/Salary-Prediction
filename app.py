import flask
import numpy as np
import pickle
with open("SLR.pkl", "rb") as f:
    model = pickle.load(f)
from flask import render_template, request, Flask

app=Flask(__name__)
@app.route('/')
def sample():
    return render_template('index.html')
@app.route("/predict",methods = ['GET','POST'])
def fun3():
    a = [float(i) for i in request.form.values()]
    b = [np.array(a)]
    predictions = model.predict(b)
    predictions = predictions[0]
    return render_template('index.html',prediction_text = predictions)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import model

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])

def home():
    return render_template("home.html")

@app.route("/dengue", methods = ["GET","POST"])

def dengue():
    
    predictedDisease = 0
    if request.method == "POST":
        features = []
        for i in range(1,8):
            s = "feature"+ str(i)
            features.append(float(request.form[s]))
        
        

        predictedDisease = model.dengue_prediction(features)

    return render_template("dengue.html", pd = predictedDisease)


@app.route("/thyroid", methods = ["GET","POST"])

def thyroid():

    predictedDisease = 0
    if request.method == "POST":
        features = []

        for i in range(1,28):
            s = "feature"+ str(i)
            features.append(float(request.form[s]))

        predictedDisease = model.thyroid_prediction(features)

    return render_template("thyroid.html", pd = predictedDisease)


@app.route("/diabetes", methods = ["GET","POST"])

def diabetes():

    predictedDisease = 0
    print('ASDASDASDADASDASDSADSA')
    if request.method == "POST":
        features = []

        for i in range(1,9):
            s = "feature"+ str(i)
            features.append(float(request.form[s]))

        predictedDisease = model.diabetes_prediction(features)

    return render_template("diabetes.html", pd = predictedDisease)


@app.route("/all", methods = ["GET","POST"])

def disease():
    predictedDisease = 0
    if request.method == "POST":
        features = []

        for i in range(1,43):
            s = "feature"+ str(i)
            features.append(float(request.form[s]))

        predictedDisease = model.disease_prediction(features)

    return render_template("index.html", pd = predictedDisease)



if __name__ == "__main__":
    app.run(debug= True)



"""
@app.route("/sub", methods = ["POST"])
def submit():
    if request.method =="POST":
        name = request.form["username"]

    return render_template("submit.html",n = name)"""

"""     features.append(request.form["feature2"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])
        features.append(request.form["feature1"])"""


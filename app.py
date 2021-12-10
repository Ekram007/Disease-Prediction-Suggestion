from flask import Flask, render_template, request
import pandas as pd
import model

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])

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


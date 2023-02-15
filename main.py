from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

with open("iris.pickle",'rb') as f:
    classifier = pickle.load(f)

@app.route("/predict", methods=["GET"])
def predicet_type():
    """check type of flower
    This is using docstrings for specifications.
    ---
    parameters:
      - name: sepal length (cm)
        in: query
        type: number
        required: true
      - name: sepal width (cm)
        in: query
        type: number
        required: true
      - name: petal length (cm)
        in: query
        type: number
        required: true
      - name: petal width (cm)
        in: query
        type: number
        required: true
    responses:
        200:
            description: the output type is

    """
    sepal_length = request.args.get('sepal length (cm)')
    sepal_width = request.args.get('sepal width (cm)')
    petal_length = request.args.get('petal length (cm)')
    petal_width = request.args.get('petal width (cm)')
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction==0:
        return 'Setosa'
    elif prediction==1:
        return "Virginica"
    else:
        return "Versicolor"


@app.route("/")
def welcomwe():
    return "welcome"









if __name__ == '__main__':
    app.run()

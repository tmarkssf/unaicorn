from django.shortcuts import render
from . import test_model
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def result(request):
    pclass = int(request.GET["pclass"])
    age = int(request.GET["age"])
    sibsp = int(request.GET["sibsp"])
    parch = int(request.GET["parch"])
    fare = int(request.GET["fare"])
    male = int(request.GET["male"])
    q = int(request.GET["q"])
    s = int(request.GET["s"])
    prediction = ml_predict.titanic_prediction_model(pclass, age, sibsp, parch, fare, male, q, s)
    return render(request, 'result.html', {'prediction': prediction})

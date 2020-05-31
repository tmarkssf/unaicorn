def titanic_prediction_model(pclass, age, sibsp, parch, fare, male, q, s):
    import pickle
    X = [[pclass, age, sibsp, parch, fare, male, q, s]]
    logreg_model = pickle.load(open('titanic_model_aws.sav','rb'))
    prediction = logreg_model.predict(X)
    if prediction == 0:
        prediction = 'You did not survive'
    elif prediction == 1:
        prediction = 'You survived!'
    else:
        prediction = 'Not sure if you survived or not'
    return prediction

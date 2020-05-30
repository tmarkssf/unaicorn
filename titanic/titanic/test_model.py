def test_predict(user_age):
    if user_age > 10:
        prediction = "survived"
    else:
        prediction = "Super survived"
    return prediction

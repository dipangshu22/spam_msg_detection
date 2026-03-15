from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
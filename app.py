from flask import Flask, render_template, request, jsonify
import joblib

from preprocess import clean_text


app = Flask(__name__)

# Load trained model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    news_text = data.get("news", "").strip()

    if not news_text:
        return jsonify({
            "error": "Please enter news text."
        }), 400

    # Clean input
    cleaned_text = clean_text(news_text)

    # Convert to TF-IDF
    text_vector = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = model.predict(text_vector)[0]

    # Probability
    probabilities = model.predict_proba(text_vector)[0]

    classes = model.classes_

    probability_dict = dict(zip(classes, probabilities))

    confidence = max(probabilities) * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "fake_probability": round(
            probability_dict.get("FAKE", 0) * 100, 2
        ),
        "real_probability": round(
            probability_dict.get("REAL", 0) * 100, 2
        )
    })


if __name__ == "__main__":
    app.run(debug=True)
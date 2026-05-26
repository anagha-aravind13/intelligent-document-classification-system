import joblib

from app.utils.preprocessing import preprocess_text


# Load model and vectorizer

model = joblib.load("app/trained_models/model.pkl")

vectorizer = joblib.load(
    "app/trained_models/vectorizer.pkl"
)


def classify_document(text):

    processed_text = preprocess_text(text)

    vectorized_text = vectorizer.transform(
        [processed_text]
    )

    prediction = model.predict(vectorized_text)[0]

    probability = model.predict_proba(
        vectorized_text
    ).max()

    return {
        "predicted_category": prediction,
        "confidence": round(probability * 100, 2)
    }
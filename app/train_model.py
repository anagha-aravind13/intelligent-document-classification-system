import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.utils.preprocessing import preprocess_text


# Load dataset

df = pd.read_csv("dataset/documents.csv")

print(df.head())


# Remove missing values

df = df.dropna()


# Preprocess text

df['processed_text'] = df['text'].apply(preprocess_text)

# Features and labels

X = df['processed_text']
y = df['label']


# TF-IDF vectorization

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# Model

model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)


# Evaluation

print(classification_report(y_test, predictions))


# Save model

joblib.dump(model, "app/trained_models/model.pkl")
joblib.dump(vectorizer, "app/trained_models/vectorizer.pkl")

print("Model trained and saved successfully")
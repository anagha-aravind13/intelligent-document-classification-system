# Intelligent Document Classification System

This project classifies uploaded PDF documents into business-oriented categories using a small NLP and machine learning pipeline. It combines a Streamlit frontend for file upload and preview, a FastAPI backend for inference, and a trained scikit-learn model for prediction.

## Highlights

- Upload and preview PDF files in the Streamlit interface
- Extract document text with `PyPDF2`
- Clean and normalize text with NLTK-based preprocessing
- Transform text with TF-IDF vectorization
- Predict document categories with logistic regression
- Return a confidence score and confidence level for each classification

## Architecture

- [Architecture diagram](./architecture.png)
- [Screenshots folder](./screenshots/)

## Project Structure

```text
document_classification_system/
|-- app/
|   |-- main.py
|   |-- train_model.py
|   |-- services/
|   |   `-- classifier.py
|   |-- trained_models/
|   |   |-- model.pkl
|   |   `-- vectorizer.pkl
|   `-- utils/
|       |-- pdf_extractor.py
|       `-- preprocessing.py
|-- dataset/
|   `-- documents.csv
|-- screenshots/
|-- temp_uploads/
|-- test_documents/
|-- requirements.txt
`-- streamlit_app.py
```

## How It Works

1. A user uploads a PDF in the Streamlit app.
2. The frontend sends the file to the FastAPI classification endpoint.
3. The backend stores the uploaded file temporarily.
4. `pdf_extractor.py` reads the PDF and extracts plain text.
5. `preprocessing.py` normalizes the text and removes stopwords.
6. `classifier.py` loads the saved TF-IDF vectorizer and logistic regression model.
7. The backend returns the predicted category, confidence score, and confidence level.
8. The Streamlit UI displays the result and a progress bar.

## Model Training

Training is handled by [app/train_model.py](./app/train_model.py). The script:

- loads `dataset/documents.csv`
- drops missing rows
- preprocesses the document text
- vectorizes text with `TfidfVectorizer`
- trains a `LogisticRegression` model
- saves the model artifacts to `app/trained_models/`

## Expected Document Categories

Based on the dataset and sample files in this repository, the system is currently oriented around categories such as:

- `invoice`
- `ShippingOrder`
- `purchase order` style documents
- `resume`
- `sales report`

The exact classes available at runtime depend on the labels present in `dataset/documents.csv`.

## Running the Project

## 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install streamlit nltk requests
```

## 3. Train or refresh the model

```powershell
python -m app.train_model
```

## 4. Start the backend API

```powershell
uvicorn app.main:app --reload
```

## 5. Start the Streamlit frontend

```powershell
streamlit run streamlit_app.py
```

The frontend and backend should be started in separate terminals. Keep your runtime-specific connection details in local configuration or code as needed, rather than publishing them in project documentation.

## API Response Shape

The classification endpoint returns a response in this format:

```json
{
  "filename": "sample.pdf",
  "predicted_category": "invoice",
  "confidence_score": "96.41%",
  "confidence_level": "High"
}
```

## Notes

- Uploaded files are temporarily stored in `temp_uploads/`.
- The current implementation is focused on PDF classification.
- `requirements.txt` does not currently include every imported runtime package used by the app UI and preprocessing flow, so the extra install step above is intentional.

import os

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from app.services.classifier import classify_document
from app.utils.pdf_extractor import extract_text_from_pdf


app = FastAPI(
    title="Intelligent Document Classification API"
)


UPLOAD_FOLDER = "temp_uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():

    return {
        "message": "API Running Successfully"
    }


@app.post("/classify-document")
async def classify_uploaded_document(
    file: UploadFile = File(...)
):

    try:

        # Save uploaded file

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text

        extracted_text = extract_text_from_pdf(
            file_path
        )

        if not extracted_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Could not extract text"
            )

        # Predict category

        result = classify_document(
            extracted_text
        )

        confidence = result["confidence"]

        if confidence >= 90:
            confidence_level = "High"

        elif confidence >= 70:
            confidence_level = "Medium"

        else:
            confidence_level = "Low"

        return {
            "filename": file.filename,
            "predicted_category": result["predicted_category"],
            "confidence_score": f"{confidence}%",
            "confidence_level": confidence_level
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
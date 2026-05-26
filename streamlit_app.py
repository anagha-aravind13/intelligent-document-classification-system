import requests
import streamlit as st
import base64


st.set_page_config(
    page_title="Document Classification System",
    layout="wide"
)


st.title("📄 Intelligent Document Classification System")

st.write(
    "Upload a PDF document and classify it "
    "using AI/ML NLP pipeline."
)


# -----------------------------------
# PDF Preview Function
# -----------------------------------

def display_pdf(uploaded_file):

    base64_pdf = base64.b64encode(
        uploaded_file.read()
    ).decode('utf-8')

    pdf_display = f'''
        <div style="
            display:flex;
            justify-content:center;
            margin-bottom:10px;
        ">
            <iframe
                src="data:application/pdf;base64,{base64_pdf}"
                width="300"
                height="180"
                type="application/pdf"
                style="
                    border:1px solid #ccc;
                    border-radius:10px;
                ">
            </iframe>
        </div>
    '''

    st.markdown(
        pdf_display,
        unsafe_allow_html=True
    )

# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type=["pdf"]
)


# -----------------------------------
# Show Preview
# -----------------------------------

if uploaded_file is not None:

    st.subheader("📑 Document Preview")

    display_pdf(uploaded_file)

    uploaded_file.seek(0)

    st.divider()

    # -----------------------------------
    # Classification Button
    # -----------------------------------

    if st.button("🚀 Classify Document"):

        with st.spinner("Classifying document..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }

            response = requests.post(
                "http://127.0.0.1:8000/classify-document",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Document Classified Successfully"
                )

                st.subheader("📊 Prediction Result")

                predicted_category = result[
                    "predicted_category"
                ]

                confidence_score = result[
                    "confidence_score"
                ]

                confidence_level = result[
                    "confidence_level"
                ]

                st.write(
                    f"### Category: "
                    f"{predicted_category}"
                )

                st.write(
                    f"### Confidence Score: "
                    f"{confidence_score}"
                )

                st.write(
                    f"### Confidence Level: "
                    f"{confidence_level}"
                )

                confidence_value = float(
                    confidence_score.replace("%", "")
                )

                st.progress(
                    confidence_value / 100
                )

            else:

                st.error(
                    "Document classification failed"
                )

                st.write(response.text)
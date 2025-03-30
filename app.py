import streamlit as st
import ranking
import base64  
# Streamlit Page Configuration
st.set_page_config(page_title="ATS Resume Checker", layout="wide")

# Title
st.title("📄 AI-Powered ATS Resume Screening")

# Create Two Columns for Job Description & Resume Upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter Job Description")
    job_desc = st.text_area("Paste the job description here")

with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload one or more resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

# Function to Encode PDF as Base64
def get_pdf_download_link(pdf_file):
    """Encodes PDF file to base64 for download link."""
    pdf_bytes = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode()  # ✅ Convert binary to base64
    return f"data:application/pdf;base64,{pdf_base64}"

# Button to process resumes
if st.button("Check ATS Score") and uploaded_files and job_desc:
    results = []
    
    for uploaded_file in uploaded_files:
        # Extract text from the uploaded PDF
        resume_text = ranking.extract_text_from_pdf(uploaded_file)
        
        # Compute ATS Score using BERT
        ats_score = ranking.bert_similarity(job_desc, resume_text) * 100  # Convert to percentage

        # Save results
        results.append({
            "Resume Name": uploaded_file.name,
            "Resume Link": f"[Download Resume]({get_pdf_download_link(uploaded_file)})",  # ✅ Corrected
            "ATS Score": f"{ats_score:.2f}%"
        })

    # Display ATS Scores in a table
    st.subheader("📊 ATS Score Results")
    st.table(results)

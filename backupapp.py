import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pytesseract  # OCR library
import cv2  # For image processing
import google.generativeai as genai
from google.generativeai import GenerativeModel
from pinecone import Pinecone, ServerlessSpec
from crewai import Agent, Task, Crew
import uuid
import json
import re
from dotenv import load_dotenv
from datetime import datetime
import traceback
import plotly.express as px
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# Load environment variables from .env file
load_dotenv()

# Configuration and Setup
st.set_page_config(page_title="Student Admission AI System", layout="wide")

# Initialize environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Check if API keys are available
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found. Please add it to your .env file.")
    st.stop()

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "student-admission-data"

# Create index if it doesn't exist
try:
    # First, check if the index already exists
    existing_indexes = pc.list_indexes()
    existing_index_names = [index.name for index in existing_indexes]
    
    if index_name not in existing_index_names:
        # Create the index with us-east-1 AWS region
        st.info("Creating index in us-east-1 AWS region")
        
        pc.create_index(
            name=index_name,
            dimension=768,  # Gemini embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    st.success(f"Successfully connected to Pinecone index: {index_name}")
    
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    index = None

# Helper Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF document with improved error handling and OCR"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    text = ""
    try:
        doc = fitz.open(tmp_path)
        
        # First try the normal text extraction
        for page in doc:
            page_text = page.get_text()
            text += page_text
        
        # If minimal text found, it might be an image-based PDF, try OCR
        if len(text.strip()) < 100:  # More flexible threshold
            st.info("Detected image-based PDF, performing OCR...")
            text = extract_text_with_ocr(doc)
        
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    finally:
        os.unlink(tmp_path)
    
    return text

def extract_text_with_ocr(doc):
    """Extract text from image-based PDF using OCR with improved image preprocessing"""
    all_text = ""
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)
        
        # Convert page to image with higher resolution for better OCR quality
        pix = page.get_pixmap(matrix=fitz.Matrix(350/72, 350/72))  # Higher resolution
        img_bytes = pix.tobytes("png")
        
        # Convert to format OpenCV can read
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Enhanced preprocessing for better OCR
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding instead of global
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply OCR with improved configuration
        try:
            # Use PSM 6 (block of text) for better structured text recognition
            text = pytesseract.image_to_string(
                img_thresh, 
                lang='eng',
                config='--psm 6 --oem 3'
            )
            all_text += text + "\n\n"
        except Exception as e:
            st.warning(f"OCR error on page {page_num+1}: {e}")
            
    return all_text

def generate_embeddings(text):
    """Generate embeddings using Gemini"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def extract_structured_data_from_document(document_text, data_type="marksheet"):
    """Extract structured data from document text with improved error handling and field extraction"""
    try:
        # Create more detailed and structured prompts based on document type
        if data_type == "marksheet":
            prompt = f"""
            Extract the following information from this 12th standard marksheet.
            Be extremely precise about separating and identifying the numerical values and calculate the marks like theory +practical so it is always around the range of 70-100.
            You can also calculate the % marks by calculating the avg of core subjects like maths physics chem english bio or economics or computer science or economics it is mostly 5 in case of cbse board exams 
            1. Student name (full name as written)
            2. Roll number/ID (exactly as printed)
            3. School/Board name
            4. Year of passing/examination
            5. Physics marks (only the numerical value)
            6. Chemistry marks (only the numerical value)
            7. Mathematics marks (only the numerical value)
            8. English marks (if available, only the numerical value)
            9. Any other subject marks (if available)
            10. Total marks (if provided)
            11. Percentage (if provided)
            
            Carefully distinguish between different subjects. Ensure Physics, Chemistry and Mathematics marks are correctly identified with the right subject.
            
            Document Text:
            {document_text}
            
            Format your response ONLY as a JSON object with keys like "student_name", "roll_number", "school_name", "board_name", "year_of_passing", "physics_marks", "chemistry_marks", "mathematics_marks", etc.
            Ensure all numerical values are returned as numbers, not strings.
            If a particular field is not found, use null as its value. Return only the JSON object, nothing else.
            """
        else:  # entrance exam scorecard
            prompt = f"""
            Extract the following information from this entrance exam (JEE/WBJEE) scorecard.
            Be extremely precise about separating and identifying the numerical values.
            
            1. Student name (full name as written)
            2. Registration/Roll number (exactly as printed)
            3. Exam name (JEE/WBJEE) and year
            4. Overall rank (only the numerical value)
            5. Category rank (if available, only the numerical value)
            6. Physics score/marks (if available, only the numerical value)
            7. Chemistry score/marks (if available, only the numerical value)
            8. Mathematics score/marks (if available, only the numerical value)
            9. Total score/marks (only the numerical value)
            10. Date of birth (if available)
            
            Document Text:
            {document_text}
            
            Format your response ONLY as a JSON object with keys like "student_name", "registration_number", "exam_name", "overall_rank", "total_score", etc.
            Ensure all numerical values are returned as numbers, not strings.
            If a particular field is not found, use null as its value. Return only the JSON object, nothing else.
            """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response to ensure it's proper JSON
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        # Parse the JSON
        extracted_data = json.loads(response_text)
        
        # Ensure numerical fields are properly typed
        if data_type == "marksheet":
            for field in ["physics_marks", "chemistry_marks", "mathematics_marks", "english_marks", "total_marks", "percentage"]:
                if field in extracted_data and extracted_data[field] is not None:
                    try:
                        extracted_data[field] = float(extracted_data[field])
                    except (ValueError, TypeError):
                        pass  # Keep as is if conversion fails
        else:  # entrance exam
            for field in ["overall_rank", "category_rank", "physics_score", "chemistry_score", "mathematics_score", "total_score"]:
                if field in extracted_data and extracted_data[field] is not None:
                    try:
                        extracted_data[field] = float(extracted_data[field])
                    except (ValueError, TypeError):
                        pass  # Keep as is if conversion fails
        
        return extracted_data
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}. Attempting enhanced extraction...")
        # More robust fallback method
        return enhanced_regex_extraction(document_text, data_type)
    except Exception as e:
        st.error(f"Data extraction error: {str(e)}")
        return {"error": f"Data extraction failed: {str(e)}"}

def enhanced_regex_extraction(document_text, data_type):
    """Enhanced regex-based extraction as fallback method"""
    extracted_data = {}
    
    # Common extraction
    student_name_match = re.search(r'(?:Name|NAME)[:\s]*([A-Za-z\s]+)', document_text, re.IGNORECASE)
    if student_name_match:
        extracted_data["student_name"] = student_name_match.group(1).strip()
    
    if data_type == "marksheet":
        # Extract roll number with more patterns
        roll_patterns = [
            r'Roll\s*(?:No|Number|#)[.:\s]*([A-Z0-9]+)',
            r'(?:Reg|Registration)[.:\s]*(?:No|Number|#)[.:\s]*([A-Z0-9]+)',
            r'(?:Enroll|Enrollment)[.:\s]*(?:No|Number|#)[.:\s]*([A-Z0-9]+)'
        ]
        
        for pattern in roll_patterns:
            roll_match = re.search(pattern, document_text, re.IGNORECASE)
            if roll_match:
                extracted_data["roll_number"] = roll_match.group(1).strip()
                break
        
        # Extract board name
        board_patterns = [
            r'(CENTRAL BOARD OF SECONDARY EDUCATION|CBSE)',
            r'(WEST BENGAL BOARD|WBCHSE)',
            r'(COUNCIL FOR.*EXAMINATION|ICSE)',
            r'(STATE BOARD OF.*)'
        ]
        
        for pattern in board_patterns:
            board_match = re.search(pattern, document_text, re.IGNORECASE)
            if board_match:
                extracted_data["board_name"] = board_match.group(1).strip()
                break
        
        # Extract year with more patterns
        year_patterns = [
            r'(?:EXAMINATION|EXAM)[,\s]*(\d{4})',
            r'(?:YEAR|SESSION)[:\s]*(\d{4})',
            r'(?:RESULT|MARKS)[.\s]*(\d{4})'
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, document_text, re.IGNORECASE)
            if year_match:
                extracted_data["year_of_passing"] = year_match.group(1)
                break
        
        # Extract subject marks with more precise patterns and context checking
        subject_patterns = {
            "physics_marks": [
                r'PHYSICS[.\s]*(\d+)',
                r'PHY(?:SICS)?[.\s]*(\d+)'
            ],
            "chemistry_marks": [
                r'CHEMISTRY[.\s]*(\d+)',
                r'CHEM(?:ISTRY)?[.\s]*(\d+)'
            ],
            "mathematics_marks": [
                r'MATHEMATICS[.\s]*(\d+)',
                r'MATH(?:EMATICS)?[.\s]*(\d+)'
            ],
            "english_marks": [
                r'ENGLISH[.\s]*(\d+)',
                r'ENG(?:LISH)?[.\s]*(\d+)'
            ]
        }
        
        for subject, patterns in subject_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, document_text, re.IGNORECASE)
                if match:
                    try:
                        extracted_data[subject] = int(match.group(1))
                    except ValueError:
                        extracted_data[subject] = None
                    break
                    
    else:  # entrance exam scorecard
        # Extract registration number with more patterns
        reg_patterns = [
            r'(?:Registration|Reg)[.\s]*(?:No|Number|#)[.:\s]*([A-Z0-9]+)',
            r'(?:Roll|Application)[.\s]*(?:No|Number|#)[.:\s]*([A-Z0-9]+)',
            r'(?:Candidate|Student)[.\s]*(?:ID|Number)[.:\s]*([A-Z0-9]+)'
        ]
        
        for pattern in reg_patterns:
            reg_match = re.search(pattern, document_text, re.IGNORECASE)
            if reg_match:
                extracted_data["registration_number"] = reg_match.group(1).strip()
                break
        
        # Extract exam name and year
        exam_patterns = [
            r'(JEE|WBJEE)[-\s]*(MAIN|ADVANCED)?[-\s]*(\d{4})',
            r'(Joint Entrance Examination|West Bengal Joint Entrance)[-\s]*(\d{4})'
        ]
        
        for pattern in exam_patterns:
            exam_match = re.search(pattern, document_text, re.IGNORECASE)
            if exam_match:
                if exam_match.group(2):
                    extracted_data["exam_name"] = f"{exam_match.group(1)}-{exam_match.group(2)}-{exam_match.group(3)}"
                else:
                    extracted_data["exam_name"] = f"{exam_match.group(1)}-{exam_match.group(3) if len(exam_match.groups()) >= 3 else ''}"
                break
        
        # Extract rank with more patterns
        rank_patterns = [
            r'(?:Overall|Merit|General|CRL)[.\s]*(?:Rank|Position)[.:\s]*(\d+)',
            r'(?:Rank|Position)[.:\s]*(\d+)'
        ]
        
        for pattern in rank_patterns:
            rank_match = re.search(pattern, document_text, re.IGNORECASE)
            if rank_match:
                try:
                    extracted_data["overall_rank"] = int(rank_match.group(1))
                except ValueError:
                    extracted_data["overall_rank"] = None
                break
        
        # Extract score patterns
        score_patterns = {
            "physics_score": [
                r'PHYSICS[.\s]*(?:SCORE|MARKS)[.:\s]*(\d+(?:\.\d+)?)',
                r'PHY(?:SICS)?[.\s]*(\d+(?:\.\d+)?)'
            ],
            "chemistry_score": [
                r'CHEMISTRY[.\s]*(?:SCORE|MARKS)[.:\s]*(\d+(?:\.\d+)?)',
                r'CHEM(?:ISTRY)?[.\s]*(\d+(?:\.\d+)?)'
            ],
            "mathematics_score": [
                r'MATHEMATICS[.\s]*(?:SCORE|MARKS)[.:\s]*(\d+(?:\.\d+)?)',
                r'MATH(?:EMATICS)?[.\s]*(\d+(?:\.\d+)?)'
            ],
            "total_score": [
                r'(?:TOTAL|AGGREGATE)[.\s]*(?:SCORE|MARKS)[.:\s]*(\d+(?:\.\d+)?)',
                r'(?:SCORE|MARKS)[.:\s]*(?:TOTAL|AGGREGATE)[.:\s]*(\d+(?:\.\d+)?)'
            ]
        }
        
        for score_type, patterns in score_patterns.items():
            for pattern in patterns:
                score_match = re.search(pattern, document_text, re.IGNORECASE)
                if score_match:
                    try:
                        extracted_data[score_type] = float(score_match.group(1))
                    except ValueError:
                        extracted_data[score_type] = None
                    break
    
    return extracted_data

def store_in_pinecone(student_id, data, embeddings):
    """Store student data and embeddings in Pinecone"""
    try:
        if index is None:
            st.error("Pinecone index is not available")
            return False
            
        # Create a flattened version of the metadata for Pinecone
        # Pinecone only accepts string, number, boolean or list of strings as values
        flattened_data = {}
        
        # Personal info
        for key, value in data.get('personal', {}).items():
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                flattened_data[f"personal_{key}"] = value
            else:
                flattened_data[f"personal_{key}"] = str(value)
        
        # Academic info
        for key, value in data.get('academic', {}).items():
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                flattened_data[f"academic_{key}"] = value
            else:
                flattened_data[f"academic_{key}"] = str(value)
        
        # Loan info
        flattened_data["loan_required"] = data.get('loan', {}).get('required', False)
        if data.get('loan', {}).get('required'):
            loan_amount = data.get('loan', {}).get('loan_amount')
            family_income = data.get('loan', {}).get('family_income')
            flattened_data["loan_amount"] = loan_amount if isinstance(loan_amount, (int, float)) else 0
            flattened_data["family_income"] = family_income if isinstance(family_income, (int, float)) else 0
            flattened_data["guarantor_name"] = data.get('loan', {}).get('guarantor_name', "")
            flattened_data["guarantor_relation"] = data.get('loan', {}).get('guarantor_relation', "")
        
        # Status info
        for key, value in data.get('status', {}).items():
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                flattened_data[f"status_{key}"] = value
            else:
                flattened_data[f"status_{key}"] = str(value)
        
        # Convert any extracted_data to simple strings to avoid nesting issues
        if 'extracted_data' in data:
            for doc_type, doc_data in data['extracted_data'].items():
                if isinstance(doc_data, dict):
                    # Store each field separately with proper prefix
                    for field_key, field_value in doc_data.items():
                        key_name = f"extracted_{doc_type}_{field_key}"
                        if isinstance(field_value, (str, int, float, bool)) or (isinstance(field_value, list) and all(isinstance(x, str) for x in field_value)):
                            flattened_data[key_name] = field_value
                        else:
                            flattened_data[key_name] = str(field_value)
                else:
                    flattened_data[f"extracted_{doc_type}"] = str(doc_data)
        
        # Store the ID separately
        flattened_data["id"] = student_id
        flattened_data["full_name"] = data.get('personal', {}).get('name', "Unknown")
        flattened_data["email"] = data.get('personal', {}).get('email', "")
        flattened_data["exam_rank"] = data.get('academic', {}).get('exam_rank', 0)
        
        # Store serialized full data for later reconstruction
        # Only include basic data that we know will deserialize properly
        storage_data = {
            "id": student_id,
            "personal": data.get('personal', {}),
            "academic": data.get('academic', {}),
            "status": data.get('status', {}),
            "loan": {
                "required": data.get('loan', {}).get('required', False),
                "loan_amount": data.get('loan', {}).get('loan_amount', 0),
                "family_income": data.get('loan', {}).get('family_income', 0),
                "guarantor_name": data.get('loan', {}).get('guarantor_name', ""),
                "guarantor_relation": data.get('loan', {}).get('guarantor_relation', "")
            }
        }
        
        # Store extracted data separately to avoid nesting issues
        extracted_marksheet = data.get('extracted_data', {}).get('marksheet', {})
        extracted_scorecard = data.get('extracted_data', {}).get('scorecard', {})
        
        if isinstance(extracted_marksheet, dict) and not any(isinstance(v, (dict, list)) for v in extracted_marksheet.values()):
            storage_data["extracted_marksheet"] = extracted_marksheet
        else:
            storage_data["extracted_marksheet"] = {"error": "Data structure not compatible with storage"}
            
        if isinstance(extracted_scorecard, dict) and not any(isinstance(v, (dict, list)) for v in extracted_scorecard.values()):
            storage_data["extracted_scorecard"] = extracted_scorecard
        else:
            storage_data["extracted_scorecard"] = {"error": "Data structure not compatible with storage"}
        
        # Add the JSON string of the data for retrieval
        try:
            flattened_data["storage_data"] = json.dumps(storage_data)
        except Exception as json_err:
            st.warning(f"Could not serialize full data: {json_err}")
            flattened_data["storage_data"] = json.dumps({"id": student_id, "error": "Serialization failed"})
            
        index.upsert(vectors=[
            {
                "id": student_id,
                "values": embeddings,
                "metadata": flattened_data
            }
        ])
        return True
    except Exception as e:
        st.error(f"Error storing data in Pinecone: {e}")
        return False

def query_pinecone(student_id):
    """Query student data from Pinecone by ID"""
    try:
        response = index.fetch(ids=[student_id])
        if student_id in response.vectors:
            vector_data = response.vectors[student_id]
            
            # Convert to dictionary for easier handling
            data = {
                "id": vector_data.id,
                "values": vector_data.values,
                "metadata": vector_data.metadata
            }
            
            # Try to reconstruct full data if storage_data exists
            if "storage_data" in data["metadata"]:
                try:
                    data["metadata"] = json.loads(data["metadata"]["storage_data"])
                except json.JSONDecodeError:
                    pass
                    
            return data
        return None
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return None

def get_pinecone_embedding(student_id):
    """Retrieve existing embedding from Pinecone"""
    try:
        response = index.fetch(ids=[student_id])
        if student_id in response.vectors:
            return response.vectors[student_id].values
        return None
    except Exception as e:
        st.error(f"Error retrieving embedding: {e}")
        return None



def upsert_to_pinecone(student_id, data, embedding, namespace="default"):
    """Update or insert student data in Pinecone"""
    try:
        # Check if embedding is valid
        if embedding is None:
            st.error("Cannot upsert with None embedding")
            return False
            
        # Prepare metadata (flatten nested structures)
        metadata = {
            "id": student_id,
            "personal_name": data.get("personal", {}).get("name", ""),
            "personal_email": data.get("personal", {}).get("email", ""),
            "academic_exam_rank": data.get("academic", {}).get("exam_rank", 0),
            "status": json.dumps(data.get("status", {}))
        }
        
        # Add loan info if exists
        if data.get("loan", {}).get("required", False):
            metadata.update({
                "loan_required": True,
                "loan_amount": data["loan"].get("loan_amount", 0),
                "family_income": data["loan"].get("family_income", 0)
            })
        else:
            metadata["loan_required"] = False
        
        # Add extracted data if exists
        if "extracted_data" in data:
            metadata["extracted_data"] = json.dumps(data["extracted_data"])
        
        # Ensure embedding is list-like
        if not isinstance(embedding, (list, np.ndarray)):
            st.error(f"Invalid embedding type: {type(embedding)}")
            return False
            
        index.upsert(
            vectors=[{
                "id": student_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
        return True
    except Exception as e:
        st.error(f"Error upserting to Pinecone: {e}")
        return False

# Eligibility Criteria
ELIGIBILITY_CRITERIA = """
1. JEE Rank should be less than 10,000 OR WBJEE rank should be less than 5,000
2. Student must have passed 12th standard with at least 75% marks
3. Student must have studied Physics, Chemistry, and Mathematics in 12th standard
4. Age should be less than 25 years as of the application date
"""

# University Capacity
UNIVERSITY_CAPACITY = """
1. Maximum intake: 500 students
2. Preference to students with higher JEE/WBJEE rank
3. 10% seats reserved for economically weaker sections
4. 15% seats reserved for SC/ST candidates
"""

# Loan Criteria
LOAN_CRITERIA = """
1. Maximum loan amount: Rs. 5,00,000
2. Family income should be less than Rs. 8,00,000 per annum
3. Academic performance (JEE/WBJEE rank) will be considered
4. Student must have a guarantor
"""

# Main Application
def main():
    st.title("Student Admission AI System")
    
    # Initialize session state for page if not exists
    if 'page' not in st.session_state:
        st.session_state['page'] = "Home"
    
    # Sidebar for navigation - use session state for default value
    # Modify in main() function
    page = st.sidebar.selectbox(
        "Navigation", 
        ["Home", "Application Form", "Application Status", "Director Dashboard", "About"],
        index=["Home", "Application Form", "Application Status", "Director Dashboard", "About"].index(st.session_state['page'])
    )

    # Add the new dashboard function to your conditionals
    if page == "Home":
        display_home_page()
    elif page == "Application Form":
        display_application_form()
    elif page == "Application Status":
        modified_display_application_status()
    elif page == "Director Dashboard":
        display_director_dashboard() 
    else:
        display_about_page()

def display_home_page():
    st.header("Welcome to the Automated Student Admission System")
    st.write("""
    This AI-powered system helps streamline the student admission process using state-of-the-art 
    technology including Retrieval-Augmented Generation (RAG) and multi-agent AI systems.
    """)
    
    st.subheader("How it works")
    st.write("""
    1. **Fill the application form** and upload required documents
    2. **AI validates** your information and documents
    3. **Eligibility is checked** based on your JEE/WBJEE rank and other criteria
    4. **Get shortlisted** if you meet the requirements
    5. **Apply for student loans** if needed
    """)
    
    st.subheader("Features")
    st.write("""
    - **Smart Document Processing**: Our system extracts information from digital PDFs and even scanned documents using OCR technology
    - **AI-powered Validation**: Multiple AI agents work together to validate your application and documents
    - **Automated Eligibility Checking**: Get immediate feedback on your eligibility
    - **Integrated Loan Processing**: Apply for financial assistance in the same application
    """)
    
    st.subheader("Start Your Application")
    if st.button("Apply Now"):
        # Update session state and trigger rerun to navigate to application form
        st.session_state['page'] = "Application Form"
        st.rerun()

def display_application_form():
    st.header("Student Application Form")
    
    # Personal Information
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with col2:
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        category = st.selectbox("Category", ["General", "SC", "ST", "OBC", "EWS"])
    
    # Academic Information
    st.subheader("Academic Information")
    col1, col2 = st.columns(2)
    with col1:
        exam_type = st.selectbox("Entrance Exam", ["JEE Main", "WBJEE"])
        exam_rank = st.number_input("Exam Rank", min_value=1, step=1)
        board_percentage = st.number_input("12th Board Percentage", min_value=0.0, max_value=100.0, step=0.1)
    
    with col2:
        physics_marks = st.number_input("Physics Marks (12th)", min_value=0, max_value=100, step=1)
        chemistry_marks = st.number_input("Chemistry Marks (12th)", min_value=0, max_value=100, step=1)
        maths_marks = st.number_input("Mathematics Marks (12th)", min_value=0, max_value=100, step=1)
    
    # Document Upload
    st.subheader("Document Upload")
    
    st.info("""
    You can upload both digital PDFs and scanned documents (photos converted to PDF).
    Our system will automatically extract information using OCR technology.
    """)
    
    marksheet = st.file_uploader("12th Marksheet (PDF)", type=["pdf"])
    entrance_scorecard = st.file_uploader("Entrance Exam Scorecard (PDF)", type=["pdf"])
    
    # Add a checkbox to preview extracted text
    if marksheet or entrance_scorecard:
        preview_extraction = st.checkbox("Preview text extracted from documents")
        
        if preview_extraction:
            col1, col2 = st.columns(2)
            
            with col1:
                if marksheet:
                    st.subheader("Text from Marksheet")
                    marksheet_text = extract_text_from_pdf(marksheet)
                    st.text_area("Extracted Text", marksheet_text, height=200)
                    
                    # Show structured data extraction
                    st.subheader("Extracted Data")
                    marksheet_data = extract_structured_data_from_document(marksheet_text, "marksheet")
                    st.json(marksheet_data)
            
            with col2:
                if entrance_scorecard:
                    st.subheader("Text from Entrance Exam Scorecard")
                    scorecard_text = extract_text_from_pdf(entrance_scorecard)
                    st.text_area("Extracted Text", scorecard_text, height=200)
                    
                    # Show structured data extraction
                    st.subheader("Extracted Data")
                    scorecard_data = extract_structured_data_from_document(scorecard_text, "scorecard")
                    st.json(scorecard_data)
    
    # Loan Information
    st.subheader("Loan Information")
    loan_required = st.checkbox("I need financial assistance")
    
    if loan_required:
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=500000, step=10000)
            family_income = st.number_input("Annual Family Income (₹)", min_value=0, step=10000)
        
        with col2:
            guarantor_name = st.text_input("Guarantor Name")
            guarantor_relation = st.text_input("Guarantor Relation")
    else:
        loan_amount = 0
        family_income = 0
        guarantor_name = ""
        guarantor_relation = ""
    
    # Submit Button
    submit_button = st.button("Submit Application")
    
    if submit_button:
        # Basic validation
        if not full_name or not email or not phone:
            st.error("Please fill all required personal information fields.")
            return
            
        if exam_rank <= 0:
            st.error("Please enter a valid exam rank.")
            return
            
        if not marksheet or not entrance_scorecard:
            st.error("Please upload both 12th marksheet and entrance exam scorecard.")
            return
            
        if loan_required and (not guarantor_name or not guarantor_relation):
            st.error("Please fill all loan information fields.")
            return
        
        # Process and extract data from documents
        marksheet_text = extract_text_from_pdf(marksheet)
        scorecard_text = extract_text_from_pdf(entrance_scorecard)
        
        marksheet_data = extract_structured_data_from_document(marksheet_text, "marksheet")
        scorecard_data = extract_structured_data_from_document(scorecard_text, "scorecard")
        
        # Generate unique student ID
        student_id = str(uuid.uuid4())
        
        # Create structured data for storage
        student_data = {
            "personal": {
                "name": full_name,
                "dob": str(dob),
                "gender": gender,
                "email": email,
                "phone": phone,
                "category": category
            },
            "academic": {
                "exam_type": exam_type,
                "exam_rank": exam_rank,
                "board_percentage": board_percentage,
                "physics_marks": physics_marks,
                "chemistry_marks": chemistry_marks,
                "maths_marks": maths_marks
            },
            "loan": {
                "required": loan_required,
                "loan_amount": loan_amount,
                "family_income": family_income,
                "guarantor_name": guarantor_name,
                "guarantor_relation": guarantor_relation
            },
            "status": {
                "application_status": "Submitted",
                "document_verification": "Pending",
                "eligibility_check": "Pending",
                "shortlisting": "Pending",
                "loan_status": "Not Applied" if not loan_required else "Pending"
            },
            "extracted_data": {
                "marksheet": marksheet_data,
                "scorecard": scorecard_data
            }
        }
        
        # Generate embeddings for search/retrieval
        student_text = f"""
        Student: {full_name}
        Email: {email}
        Phone: {phone}
        Category: {category}
        Exam: {exam_type}
        Rank: {exam_rank}
        Board Percentage: {board_percentage}
        Physics: {physics_marks}
        Chemistry: {chemistry_marks}
        Mathematics: {maths_marks}
        Loan Required: {loan_required}
        """
        
        embeddings = generate_embeddings(student_text)
        
        if embeddings:
            # Store data in Pinecone
            success = store_in_pinecone(student_id, student_data, embeddings)
            
            if success:
                st.success(f"Application submitted successfully. Your Application ID is: {student_id}")
                st.info("Save your Application ID to check your application status later.")
                
                # Store ID in session state
                st.session_state['current_application_id'] = student_id
                
                # Process application using CrewAI
                with st.spinner("Processing your application..."):
                    process_application(student_id, student_data, marksheet_text, scorecard_text)
            else:
                st.error("Failed to submit application. Please try again.")
        else:
            st.error("Failed to process application. Please try again.")

def create_pdf_generation_agent():
    """Create an agent specifically for generating PDF documents from application data"""
    
    pdf_agent = Agent(
        role="PDF Generation Agent",
        goal="Generate professional PDF documents from application data",
        backstory="Specialized in creating standardized documents from structured data with proper formatting and branding"
    )
    
    return pdf_agent



def generate_application_pdf(student_id):
    """Generate a PDF for a student application using an agent"""
    try:
        # Retrieve application data from Pinecone
        application_data = query_pinecone(student_id)
        
        if not application_data:
            return None
            
        # Extract metadata from application_data
        metadata = application_data.get("metadata", {})
        
        # Check if we need to parse storage_data
        if "storage_data" in metadata and isinstance(metadata["storage_data"], str):
            try:
                parsed_data = json.loads(metadata["storage_data"])
                metadata.update(parsed_data)
            except json.JSONDecodeError:
                st.warning("Could not parse stored application data")
        
        # Create and fill PDF buffer directly since we have all data
        pdf_buffer = create_pdf_document(student_id, metadata)
        return pdf_buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        return None

def safely_get_data(data, path, default="N/A"):
    """Safely navigate nested dictionaries with string or dict handling"""
    if not data:
        return default
    
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, str):
            try:
                parsed = json.loads(current)
                if isinstance(parsed, dict) and key in parsed:
                    current = parsed[key]
                else:
                    return default
            except:
                return default
        else:
            return default
            
    if current is None:
        return default
    return current

def create_pdf_document(student_id, data):
    """Create a well-formatted PDF document from application data"""
    # Create a PDF buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    styles.add(ParagraphStyle(
        name='Header',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name='Subheader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue,
        spaceAfter=8
    ))
    
    styles.add(ParagraphStyle(
        name='NormalBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='Section',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.navy,
        spaceAfter=6
    ))
    
    # Create content elements
    elements = []
    
    # Add university logo/header (placeholder)
    # elements.append(Image("university_logo.png", width=200, height=50))
    elements.append(Paragraph("University Engineering Admission", styles['Header']))
    elements.append(Paragraph("Application Status and Feedback Report", styles['Subheader']))
    elements.append(Spacer(1, 12))
    
    # Add timestamp and ID
    elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Paragraph(f"Application ID: {student_id}", styles['NormalBold']))
    elements.append(Spacer(1, 12))
    
    # Add student information
    elements.append(Paragraph("Student Information", styles['Section']))
    
    # Extract personal data safely
    personal_data = {}
    for key in ['name', 'email', 'phone', 'category', 'dob', 'gender']:
        personal_data[key] = safely_get_data(data, ['personal', key])
    
    student_data = [
        ["Name:", personal_data['name']],
        ["Email:", personal_data['email']],
        ["Phone:", personal_data['phone']],
        ["Category:", personal_data['category']],
        ["Date of Birth:", personal_data['dob']],
        ["Gender:", personal_data['gender']]
    ]
    
    student_table = Table(student_data, colWidths=[120, 300])
    student_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    
    elements.append(student_table)
    elements.append(Spacer(1, 12))
    
    # Add academic information
    elements.append(Paragraph("Academic Information", styles['Section']))
    
    # Extract academic data safely
    academic_data = {}
    for key in ['exam_type', 'exam_rank', 'board_percentage', 'physics_marks', 'chemistry_marks', 'maths_marks']:
        academic_data[key] = safely_get_data(data, ['academic', key])
    
    academic_table_data = [
        ["Exam Type:", academic_data['exam_type']],
        ["Exam Rank:", academic_data['exam_rank']],
        ["Board Percentage:", f"{academic_data['board_percentage']}%"],
        ["Physics Marks:", academic_data['physics_marks']],
        ["Chemistry Marks:", academic_data['chemistry_marks']],
        ["Mathematics Marks:", academic_data['maths_marks']]
    ]
    
    academic_table = Table(academic_table_data, colWidths=[120, 300])
    academic_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    
    elements.append(academic_table)
    elements.append(Spacer(1, 12))
    
    # Add application status
    elements.append(Paragraph("Application Status", styles['Section']))
    
    # Extract status data safely
    application_status = safely_get_data(data, ['status', 'application_status'], "Submitted")
    elements.append(Paragraph(f"Overall Status: {application_status}", styles['NormalBold']))
    elements.append(Spacer(1, 6))
    
    # Create status table
    doc_status = safely_get_data(data, ['status', 'document_verification', 'status'], "Pending")
    elig_status = safely_get_data(data, ['status', 'eligibility', 'status'], "Pending")
    short_status = safely_get_data(data, ['status', 'shortlisting', 'status'], "Pending")
    
    # Check if loan was applied for
    loan_required = safely_get_data(data, ['loan', 'required'], False)
    if isinstance(loan_required, str):
        loan_required = loan_required.lower() == "true"
    
    loan_status = "Not Applied"
    if loan_required:
        loan_status = safely_get_data(data, ['loan', 'status', 'status'], "Pending")
    
    status_table_data = [
        ["Process", "Status"],
        ["Document Verification", doc_status],
        ["Eligibility Check", elig_status],
        ["Shortlisting", short_status],
        ["Loan Processing", loan_status]
    ]
    
    status_table = Table(status_table_data, colWidths=[150, 270])
    status_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    
    elements.append(status_table)
    elements.append(Spacer(1, 12))
    
    # Add detailed feedback if available
    feedback_data = safely_get_data(data, ['status', 'feedback'], {})
    
    if feedback_data and feedback_data != "N/A":
        elements.append(Paragraph("Application Feedback", styles['Section']))
        
        detailed_feedback = safely_get_data(data, ['status', 'feedback', 'detailed_feedback'], "")
        if detailed_feedback and detailed_feedback != "N/A":
            elements.append(Paragraph("Detailed Feedback:", styles['NormalBold']))
            elements.append(Paragraph(detailed_feedback, styles['Normal']))
            elements.append(Spacer(1, 8))
        
        # Extract improvement areas
        improvement_areas = safely_get_data(data, ['status', 'feedback', 'improvement_areas'], [])
        if improvement_areas and improvement_areas != "N/A":
            elements.append(Paragraph("Areas for Improvement:", styles['NormalBold']))
            for area in improvement_areas:
                elements.append(Paragraph(f"• {area}", styles['Normal']))
            elements.append(Spacer(1, 8))
        
        # Extract alternative paths
        alternative_paths = safely_get_data(data, ['status', 'feedback', 'alternative_paths'], [])
        if alternative_paths and alternative_paths != "N/A":
            elements.append(Paragraph("Alternative Paths:", styles['NormalBold']))
            for path in alternative_paths:
                elements.append(Paragraph(f"• {path}", styles['Normal']))
            elements.append(Spacer(1, 8))
        
        # Extract reapplication guidance
        reapplication_guidance = safely_get_data(data, ['status', 'feedback', 'reapplication_guidance'], "")
        if reapplication_guidance and reapplication_guidance != "N/A":
            elements.append(Paragraph("Reapplication Guidance:", styles['NormalBold']))
            elements.append(Paragraph(reapplication_guidance, styles['Normal']))
            elements.append(Spacer(1, 8))
    
    # Add eligibility details if available
    eligibility_data = safely_get_data(data, ['status', 'eligibility'], {})
    
    if eligibility_data and eligibility_data != "N/A":
        elements.append(Paragraph("Eligibility Assessment", styles['Section']))
        
        status = safely_get_data(data, ['status', 'eligibility', 'status'], "Pending")
        elements.append(Paragraph(f"Status: {status}", styles['NormalBold']))
        
        reasons = safely_get_data(data, ['status', 'eligibility', 'reasons'], [])
        if reasons and status == "Eligible":
            elements.append(Paragraph("Qualification Factors:", styles['NormalBold']))
            for reason in reasons:
                elements.append(Paragraph(f"• {reason}", styles['Normal']))
            elements.append(Spacer(1, 8))
        
        missing_criteria = safely_get_data(data, ['status', 'eligibility', 'missing_criteria'], [])
        if missing_criteria and status != "Eligible":
            elements.append(Paragraph("Missing Criteria:", styles['NormalBold']))
            for criterion in missing_criteria:
                elements.append(Paragraph(f"• {criterion}", styles['Normal']))
            elements.append(Spacer(1, 8))
    
    # Add shortlisting details if available
    shortlisting_data = safely_get_data(data, ['status', 'shortlisting'], {})
    
    if shortlisting_data and shortlisting_data != "N/A":
        elements.append(Paragraph("Shortlisting Details", styles['Section']))
        
        status = safely_get_data(data, ['status', 'shortlisting', 'status'], "Pending")
        priority_score = safely_get_data(data, ['status', 'shortlisting', 'priority_score'], 0)
        remarks = safely_get_data(data, ['status', 'shortlisting', 'remarks'], "")
        
        if status != "Pending":
            if status == "Shortlisted":
                elements.append(Paragraph(f"Status: {status} (Priority Score: {priority_score})", styles['NormalBold']))
            else:
                elements.append(Paragraph(f"Status: {status}", styles['NormalBold']))
            
            if remarks:
                elements.append(Paragraph("Remarks:", styles['NormalBold']))
                elements.append(Paragraph(remarks, styles['Normal']))
                elements.append(Spacer(1, 8))
    
    # Add loan details if applied
    if loan_required:
        elements.append(Paragraph("Loan Information", styles['Section']))
        
        loan_amount = safely_get_data(data, ['loan', 'loan_amount'], 0)
        family_income = safely_get_data(data, ['loan', 'family_income'], 0)
        guarantor_name = safely_get_data(data, ['loan', 'guarantor_name'], "N/A")
        guarantor_relation = safely_get_data(data, ['loan', 'guarantor_relation'], "N/A")
        
        loan_data = [
            ["Loan Amount Requested:", f"₹{loan_amount:,}"],
            ["Family Income:", f"₹{family_income:,}"],
            ["Guarantor Name:", guarantor_name],
            ["Guarantor Relation:", guarantor_relation]
        ]
        
        loan_table = Table(loan_data, colWidths=[150, 270])
        loan_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        
        elements.append(loan_table)
        elements.append(Spacer(1, 8))
        
        # Loan status details
        loan_status_data = safely_get_data(data, ['loan', 'status'], {})
        
        if loan_status_data and loan_status_data != "N/A":
            status = safely_get_data(data, ['loan', 'status', 'status'], "Pending")
            elements.append(Paragraph(f"Loan Status: {status}", styles['NormalBold']))
            
            if status == "Approved":
                approved_amount = safely_get_data(data, ['loan', 'status', 'approved_amount'], 0)
                elements.append(Paragraph(f"Approved Amount: ₹{approved_amount:,}", styles['Normal']))
            
            if status == "Rejected":
                reasons = safely_get_data(data, ['loan', 'status', 'reasons'], [])
                if reasons:
                    elements.append(Paragraph("Rejection Reasons:", styles['NormalBold']))
                    for reason in reasons:
                        elements.append(Paragraph(f"• {reason}", styles['Normal']))
                
                alt_aid = safely_get_data(data, ['loan', 'status', 'alternative_aid'], [])
                if alt_aid:
                    elements.append(Paragraph("Alternative Financial Aid:", styles['NormalBold']))
                    for aid in alt_aid:
                        elements.append(Paragraph(f"• {aid}", styles['Normal']))
    
    # Add footer
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("This is an automated report generated based on your application data.", styles['Normal']))
    elements.append(Paragraph("For any clarifications, please contact the university admissions office.", styles['Normal']))
    elements.append(Paragraph("Email: admissions@university.edu | Phone: +91-123-456-7890", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    # Return the PDF buffer
    buffer.seek(0)
    return buffer

def get_pdf_download_link(pdf_buffer, filename="application_report.pdf"):
    """Generate a download link for a PDF file"""
    if pdf_buffer is None:
        return None
    
    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" class="btn">Download PDF Report</a>'

def display_pdf(pdf_buffer):
    """Display a PDF in the Streamlit app"""
    if pdf_buffer is None:
        return None
    
    base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    return pdf_display

def safe_json_parse(data_str, default={}):
    """Safely parse JSON string with error handling"""
    if not isinstance(data_str, str):
        return data_str
    
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return default

def process_application(student_id, student_data, marksheet_text, scorecard_text):
    """Process student application through the complete admission pipeline"""
    try:
        # Initialize result variables at the beginning of the function
        # This ensures they exist even if their respective tasks don't run
        validation_result = {"is_valid": False, "rank_valid": False, "verified_rank": None, "issues": []}
        eligibility_result = {"is_eligible": False, "reasons": [], "missing_criteria": []}
        shortlisting_result = {"is_shortlisted": False, "priority_score": 0, "evaluation_criteria": {}, "remarks": ""}
        loan_result = {"loan_approved": False, "loan_amount": 0, "reasons": [], "alternative_aid": []}
        feedback_result = {"improvement_areas": [], "alternative_paths": [], "reapplication_guidance": "", "detailed_feedback": ""}
        
        # Initialize agents and tasks
        validation_agent = Agent(
            role="Document Validation Agent",
            goal=("Validate WBJEE engineering rank or jee mains rank against application data"),
            backstory=("Specializes in verifying WBJEE or jee mains ranks from scorecard documents")
        )
        
        eligibility_agent = Agent(
            role="Eligibility Checking Agent",
            goal=("Verify if applicant meets all criteria based on WBJEE rank or jee rank and class 12 board percentage"),
            backstory=("Expert in evaluating academic qualifications against admission standards")
        )
        
        shortlisting_agent = Agent(
            role="Shortlisting Agent",
            goal=("Prioritize qualified candidates based on merit and available seats"),
            backstory=("Handles merit-based ranking considering institutional constraints and candidate strengths")
        )
        
        loan_agent = Agent(
            role="Loan Processing Agent",
            goal=("Evaluate and approve loan requests efficiently"),
            backstory=("Manages financial aid decisions based on need and merit with quick processing")
        )
        
        feedback_agent = Agent(
            role="Application Feedback Agent",
            goal=("Provide personalized feedback to rejected applicants"),
            backstory=("Specializes in analyzing rejection reasons and providing constructive feedback for future improvement")
        )
        
        pdf_agent = Agent(
            role="PDF Generation Agent",
            goal=("Generate comprehensive PDF reports of application outcomes"),
            backstory=("Expert in creating professional, well-formatted documents with all relevant application information")
        )

        # Create tasks with Pinecone integration
        validation_task = Task(
            description=(
                f"Validate WBJEE engineering or jee mains rank for student {student_id}. "
                f"Extract and verify the rank from the scorecard. "
                f"Student data: {json.dumps(student_data)}, "
                f"Scorecard text sample: {scorecard_text[:200]}..."
            ),
            expected_output=(
                "Validation report with is_valid flag, verified WBJEE engineering or jee main rank and class 12 board percentage, "
                "and any discrepancies identified"
            ),
            agent=validation_agent
        )
        
        eligibility_task = Task(
            description=(
                f"Check eligibility for student {student_id} based on "
                '''verified WBJEE rank or Jee mains rank and class 12 board percentage and admission criteria-
                Admission Criteria
                JEE Rank should be less than 10,000 OR WBJEE rank should be less than 5,000
                Student must have passed 12th standard with at least 75% marks
                Student must have studied Physics, Chemistry, and Mathematics in 12th standard
                Age should be less than 25 years as of the application date
                University Capacity
                Maximum intake: 800 students
                Preference to students with higher JEE/WBJEE rank
                10% seats reserved for economically weaker sections
                15% seats reserved for SC/ST candidates
                Loan Criteria
                Maximum loan amount: Rs. 5,00,000
                Family income should be less than Rs. 8,00,000 per annum
                Academic performance (JEE/WBJEE rank) will be considered
                Student must have a guarantor'''
                            ),
            expected_output=(
                "Eligibility determination with is_eligible flag and detailed reasons"
            ),
            agent=eligibility_agent,
            dependencies=[validation_task]
        )
        
        shortlisting_task = Task(
            description=(
                f"Evaluate shortlisting for student {student_id} considering "
                "WBJEE rank or JEE Mains rank, available seats, and program-specific requirements and also ensure that the board percentage is more than 75%"
            ),
            expected_output=(
                "Shortlisting decision with priority score, detailed evaluation criteria, and remarks"
            ),
            agent=shortlisting_agent,
            dependencies=[eligibility_task]
        )
        
        loan_task = Task(
            description=(
                f"Process loan request for student {student_id} if required if only the loan criteria is true as extracted from the database and the candidate has been shortlisted, "
                "considering merit, financial need, and available scholarship opportunities"
            ),
            expected_output=(
                "Loan decision with approval status, amount if approved, and alternative financial aid suggestions only if criteria is true"
            ),
            agent=loan_agent,
            dependencies=[shortlisting_task],
            enabled=student_data.get('loan', {}).get('required', True) and student_data.get('status', {}).get('shortlisting', {}).get('status', "") == "Shortlisted"
        )
        
        feedback_task = Task(
            description=(
                f"Generate personalized feedback for student {student_id} based on their application data. "
                f"Consider the student's academic performance, entrance exam rank, and category. "
                f"Provide actionable guidance even if the application is successful. "
                f"For rejected applications, analyze rejection reasons and provide detailed improvement suggestions."
                
            ),
            expected_output=(
                "Detailed feedback with specific improvement areas, alternative path suggestions, "
                "and reapplication guidance if applicable and also tell them the reason as to why their application was rejected was it because of eligibility criteria or missing data or any other factor"
            ),
            agent=feedback_agent,
            dependencies=[eligibility_task, shortlisting_task]
        )
        
        # Add PDF generation task
        pdf_generation_task = Task(
            description=(
                f"Generate a comprehensive PDF report for student {student_id} "
                f"containing all application data, status, feedback, and next steps. "
                f"Format the document professionally with university branding. "
                f"Include sections for personal information, academic details, "
                f"application status, eligibility assessment, shortlisting details, "
                f"and personalized feedback."
            ),
            expected_output=(
                "A binary buffer containing the generated PDF document with all "
                "required sections properly formatted and ready for download."
            ),
            agent=pdf_agent,
            dependencies=[feedback_task, eligibility_task, shortlisting_task]
        )

        # Create and run crew
        admission_crew = Crew(
            agents=[validation_agent, eligibility_agent, shortlisting_agent, loan_agent, feedback_agent, pdf_agent],
            tasks=[validation_task, eligibility_task, shortlisting_task, loan_task, feedback_task, pdf_generation_task],
            verbose=True
        )
        
        results = admission_crew.kickoff()

        # Process results
        updated_data = student_data.copy()
        
        # [Rest of the process_application function continues...]
        
        # Initialize status dictionary if not exists
        if 'status' not in updated_data:
            updated_data['status'] = {}
        
        # Update validation status - only WBJEE rank
        if 'validation_task' in results:
            validation_result = results['validation_task']
            
            updated_data['status']['document_verification'] = {
                'status': "Verified" if validation_result.get('is_valid') else "Failed",
                'wbjee': {
                    'engineering_rank_valid': validation_result.get('rank_valid', False),
                    'verified_rank': validation_result.get('verified_rank', None),
                    'issues': validation_result.get('issues', [])
                },
                'timestamp': datetime.now().isoformat()
            }
        
        # Update eligibility status
        if 'eligibility_task' in results:
            eligibility_result = results['eligibility_task']
            updated_data['status']['eligibility'] = {
                'status': "Eligible" if eligibility_result.get('is_eligible') else "Not Eligible",
                'reasons': eligibility_result.get('reasons', []),
                'missing_criteria': eligibility_result.get('missing_criteria', []),
                'timestamp': datetime.now().isoformat()
            }
        
        # Update shortlisting status
        if 'shortlisting_task' in results:
            shortlisting_result = results['shortlisting_task']
            updated_data['status']['shortlisting'] = {
                'status': "Shortlisted" if shortlisting_result.get('is_shortlisted') else "Not Shortlisted",
                'priority_score': shortlisting_result.get('priority_score', 0),
                'evaluation_criteria': shortlisting_result.get('evaluation_criteria', {}),
                'remarks': shortlisting_result.get('remarks', ""),
                'timestamp': datetime.now().isoformat()
            }
        
        # Update loan status if applicable
        if student_data.get('loan', {}).get('required', False) and 'loan_task' in results:
            loan_result = results['loan_task']
            if 'loan' not in updated_data:
                updated_data['loan'] = {}
            updated_data['loan']['status'] = {
                'status': "Approved" if loan_result.get('loan_approved') else "Rejected",
                'approved_amount': loan_result.get('loan_amount', 0),
                'reasons': loan_result.get('reasons', []),
                'alternative_aid': loan_result.get('alternative_aid', []),
                'timestamp': datetime.now().isoformat()
            }
        
        # Always add feedback regardless of application status
        if 'feedback_task' in results:
            feedback_result = results['feedback_task']
            updated_data['status']['feedback'] = {
                'improvement_areas': feedback_result.get('improvement_areas', []),
                'alternative_paths': feedback_result.get('alternative_paths', []),
                'reapplication_guidance': feedback_result.get('reapplication_guidance', ""),
                'detailed_feedback': feedback_result.get('detailed_feedback', ""),
                'timestamp': datetime.now().isoformat()
            }
        
        # Update overall application status based on all results
        # This ensures the main status is always updated
        if eligibility_result.get('is_eligible', False) and shortlisting_result.get('is_shortlisted', False):
            updated_data['status']['application_status'] = "Accepted"
        elif 'eligibility_task' in results and not eligibility_result.get('is_eligible', False):
            updated_data['status']['application_status'] = "Rejected - Not Eligible"
        elif 'shortlisting_task' in results and not shortlisting_result.get('is_shortlisted', False):
            updated_data['status']['application_status'] = "Rejected - Not Shortlisted"
        else:
            updated_data['status']['application_status'] = "Processed"
        
        # Update Pinecone record with error handling
        try:
            # Prepare text for embedding generation or retrieval
            student_text = f"""
            Student: {updated_data.get('personal', {}).get('name', 'Unknown')}
            Email: {updated_data.get('personal', {}).get('email', 'Unknown')}
            Phone: {updated_data.get('personal', {}).get('phone', 'Unknown')}
            Category: {updated_data.get('personal', {}).get('category', 'Unknown')}
            Exam: {updated_data.get('academic', {}).get('exam_type', 'Unknown')}
            Rank: {updated_data.get('academic', {}).get('exam_rank', 0)}
            Status: {updated_data.get('status', {}).get('application_status', 'Unknown')}
            """
            
            # Get current embedding or generate a new one
            current_embedding = get_pinecone_embedding(student_id)
            if current_embedding is None:
                current_embedding = generate_embeddings(student_text)
            
            if current_embedding is not None:
                # Use store_in_pinecone for consistency with original storage
                success = store_in_pinecone(student_id, updated_data, current_embedding)
                
                if not success:
                    # If store_in_pinecone fails, try direct upsert as a fallback
                    st.warning("Primary update method failed. Trying alternative method...")
                    
                    # First, query existing data to preserve any fields we're not updating
                    existing_data = query_pinecone(student_id)
                    if existing_data and 'metadata' in existing_data:
                        # Create a flattened version of the metadata for Pinecone
                        flattened_data = {}
                        
                        # Personal info
                        for key, value in updated_data.get('personal', {}).items():
                            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                                flattened_data[f"personal_{key}"] = value
                            else:
                                flattened_data[f"personal_{key}"] = str(value)
                        
                        # Academic info
                        for key, value in updated_data.get('academic', {}).items():
                            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                                flattened_data[f"academic_{key}"] = value
                            else:
                                flattened_data[f"academic_{key}"] = str(value)
                        
                        # Status info - this is the crucial part that needs updating
                        for key, value in updated_data.get('status', {}).items():
                            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                                flattened_data[f"status_{key}"] = value
                            else:
                                flattened_data[f"status_{key}"] = json.dumps(value)
                        
                        # Loan info 
                        flattened_data["loan_required"] = updated_data.get('loan', {}).get('required', False)
                        if updated_data.get('loan', {}).get('required'):
                            loan_amount = updated_data.get('loan', {}).get('loan_amount')
                            family_income = updated_data.get('loan', {}).get('family_income')
                            flattened_data["loan_amount"] = loan_amount if isinstance(loan_amount, (int, float)) else 0
                            flattened_data["family_income"] = family_income if isinstance(family_income, (int, float)) else 0
                            
                            # Add loan status if exists
                            if 'status' in updated_data.get('loan', {}):
                                flattened_data["loan_status"] = json.dumps(updated_data['loan']['status'])
                        
                        # Store the ID separately
                        flattened_data["id"] = student_id
                        flattened_data["full_name"] = updated_data.get('personal', {}).get('name', "Unknown")
                        
                        # Store serialized full data for later reconstruction
                        try:
                            flattened_data["storage_data"] = json.dumps(updated_data)
                        except Exception as json_err:
                            st.warning(f"Could not serialize full data: {json_err}")
                            
                        # Direct upsert to Pinecone
                        index.upsert(vectors=[
                            {
                                "id": student_id,
                                "values": current_embedding,
                                "metadata": flattened_data
                            }
                        ])
                        st.success("Application data updated successfully using alternative method.")
                    else:
                        st.error("Both update methods failed. Data processed but not saved.")
                else:
                    st.success("Application data updated successfully.")
            else:
                st.warning("Could not update database due to embedding issues. Data processed but not saved.")
                
            return updated_data
        except Exception as pinecone_error:
            st.error(f"Pinecone update failed: {str(pinecone_error)}")
            # Log detailed error for debugging
            st.error(f"Error details: {traceback.format_exc()}")
            return updated_data
            
    except Exception as e:
        st.error(f"Application processing failed: {str(e)}")
        # Log detailed error for debugging
        st.error(f"Error details: {traceback.format_exc()}")
        # Return partial data if available
        if 'status' not in student_data:
            student_data['status'] = {}
        student_data['status']['processing_error'] = str(e)
        student_data['status']['timestamp'] = datetime.now().isoformat()
        return student_data
    
def display_director_dashboard():
    st.header("University Director Dashboard")
    
    # Simple authentication
    with st.expander("Director Authentication", expanded=True):
        director_password = st.text_input("Enter Director Access Code", type="password")
        authenticate = st.button("Login")
        
        if authenticate:
            if director_password == "admin123":  # In production, use a more secure method
                st.session_state['director_authenticated'] = True
            else:
                st.error("Invalid access code")
    
    # Only show dashboard if authenticated
    if st.session_state.get('director_authenticated', False):
        # Show dashboard content
        display_dashboard_content()
    else:
        st.info("Please authenticate to view the director dashboard")

def display_dashboard_content():
    """Display dashboard content after authentication"""
    
    # Get metrics from database
    metrics = get_admission_metrics()
    
    # Overview section
    st.subheader("Admission Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", metrics["total_applications"])
    with col2:
        st.metric("Applications Processed", metrics["processed_applications"], 
                delta=f"{metrics['processing_percentage']}%")
    with col3:
        st.metric("Eligible Candidates", metrics["eligible_candidates"])
    with col4:
        st.metric("Shortlisted", metrics["shortlisted_candidates"])
    
    # Application status breakdown
    st.subheader("Application Status Breakdown")
    
    # Pie chart for status distribution
    fig_status = create_status_chart(metrics["status_distribution"])
    st.plotly_chart(fig_status)
    
    # Applications by department/category
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Applications by Category")
        fig_category = create_category_chart(metrics["category_distribution"])
        st.plotly_chart(fig_category)
    
    with col2:
        st.subheader("Exam Rank Distribution")
        fig_rank = create_rank_distribution(metrics["rank_distribution"])
        st.plotly_chart(fig_rank)
    
    # Loan processing summary
    st.subheader("Loan Processing Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Loan Applications", metrics["loan_applications"])
        st.metric("Approved Loans", metrics["approved_loans"])
        st.metric("Total Loan Amount (₹)", f"{metrics['total_loan_amount']:,}")
    
    with col2:
        fig_loan = create_loan_chart(metrics["loan_status_distribution"])
        st.plotly_chart(fig_loan)
    
    # Recent applications
    st.subheader("Recent Applications")
    recent_applications = get_recent_applications(10)
    
    # Display recent applications in a table
    if recent_applications:
        st.dataframe(recent_applications)
    else:
        st.info("No recent applications to display")
    
    # Action items section
    st.subheader("Action Items")
    pending_actions = get_pending_actions()
    
    if pending_actions:
        for action in pending_actions:
            st.warning(f"**{action['title']}**: {action['description']} - {action['due_date']}")
    else:
        st.success("No pending action items")

def get_admission_metrics():
    """Get admission metrics from Pinecone database"""
    try:
        # Initialize metrics dictionary
        metrics = {
            "total_applications": 0,
            "processed_applications": 0,
            "processing_percentage": 0,
            "eligible_candidates": 0,
            "shortlisted_candidates": 0,
            "status_distribution": {},
            "category_distribution": {},
            "rank_distribution": {"ranges": [], "counts": []},
            "loan_applications": 0,
            "approved_loans": 0,
            "total_loan_amount": 0,
            "loan_status_distribution": {}
        }
        
        # Check if index is available
        if index is None:
            return metrics
        
        # Query all vectors with pagination (adjust as needed for large datasets)
        try:
            # Use fetch to get all vectors
            # In a real implementation with many records, you'd use a different approach
            query_result = index.fetch(ids=[], namespace="default")
            vectors = query_result.get("vectors", {})
            
            # Calculate metrics
            metrics["total_applications"] = len(vectors)
            
            # Process each application
            status_distribution = {}
            category_distribution = {}
            loan_status_distribution = {}
            rank_ranges = {
                "1-1000": 0,
                "1001-5000": 0,
                "5001-10000": 0,
                "10001+": 0
            }
            
            for vector_id, vector_data in vectors.items():
                metadata = vector_data.get("metadata", {})
                
                # Try to parse the storage_data if available
                try:
                    if "storage_data" in metadata:
                        data = json.loads(metadata["storage_data"])
                        metadata = data  # Replace with parsed data
                except:
                    pass
                
                # Count processed applications
                if metadata.get("status", {}).get("document_verification", "Pending") != "Pending":
                    metrics["processed_applications"] += 1
                
                # Status distribution
                status = metadata.get("status", {}).get("application_status", "Submitted")
                status_distribution[status] = status_distribution.get(status, 0) + 1
                
                # Category distribution
                category = metadata.get("personal", {}).get("category", "Unknown")
                category_distribution[category] = category_distribution.get(category, 0) + 1
                
                # Eligible candidates
                if metadata.get("status", {}).get("eligibility", {}).get("status", "") == "Eligible":
                    metrics["eligible_candidates"] += 1
                
                # Shortlisted candidates
                if metadata.get("status", {}).get("shortlisting", {}).get("status", "") == "Shortlisted":
                    metrics["shortlisted_candidates"] += 1
                
                # Rank distribution
                exam_rank = metadata.get("academic", {}).get("exam_rank", 0)
                if 1 <= exam_rank <= 1000:
                    rank_ranges["1-1000"] += 1
                elif 1001 <= exam_rank <= 5000:
                    rank_ranges["1001-5000"] += 1
                elif 5001 <= exam_rank <= 10000:
                    rank_ranges["5001-10000"] += 1
                else:
                    rank_ranges["10001+"] += 1
                
                # Loan metrics
                if metadata.get("loan", {}).get("required", False):
                    metrics["loan_applications"] += 1
                    
                    loan_status = metadata.get("loan", {}).get("status", {}).get("status", "Pending")
                    loan_status_distribution[loan_status] = loan_status_distribution.get(loan_status, 0) + 1
                    
                    if loan_status == "Approved":
                        metrics["approved_loans"] += 1
                        metrics["total_loan_amount"] += metadata.get("loan", {}).get("status", {}).get("approved_amount", 0)
            
            # Calculate processing percentage
            if metrics["total_applications"] > 0:
                metrics["processing_percentage"] = round((metrics["processed_applications"] / metrics["total_applications"]) * 100, 1)
            
            # Update distributions
            metrics["status_distribution"] = status_distribution
            metrics["category_distribution"] = category_distribution
            metrics["loan_status_distribution"] = loan_status_distribution
            
            # Update rank distribution
            metrics["rank_distribution"]["ranges"] = list(rank_ranges.keys())
            metrics["rank_distribution"]["counts"] = list(rank_ranges.values())
            
            return metrics
            
        except Exception as e:
            st.error(f"Error fetching metrics: {e}")
            return metrics
    except Exception as e:
        st.error(f"Error in get_admission_metrics: {e}")
        return metrics

def get_recent_applications(limit=10):
    """Get recent applications from Pinecone database"""
    try:
        if index is None:
            return None
        
        # Query all vectors
        query_result = index.fetch(ids=[], namespace="default")
        vectors = query_result.get("vectors", {})
        
        # Create a list of applications with relevant fields
        applications = []
        
        for vector_id, vector_data in vectors.items():
            metadata = vector_data.get("metadata", {})
            
            # Try to parse storage_data if available
            try:
                if "storage_data" in metadata:
                    data = json.loads(metadata["storage_data"])
                    metadata = data  # Replace with parsed data
            except:
                pass
            
            # Extract relevant fields
            application = {
                "id": vector_id,
                "name": metadata.get("personal", {}).get("name", "Unknown"),
                "exam_type": metadata.get("academic", {}).get("exam_type", "Unknown"),
                "exam_rank": metadata.get("academic", {}).get("exam_rank", 0),
                "category": metadata.get("personal", {}).get("category", "Unknown"),
                "status": metadata.get("status", {}).get("application_status", "Submitted"),
                "document_verification": metadata.get("status", {}).get("document_verification", "Pending"),
                "eligibility": metadata.get("status", {}).get("eligibility", {}).get("status", "Pending"),
                "shortlisting": metadata.get("status", {}).get("shortlisting", {}).get("status", "Pending"),
            }
            
            applications.append(application)
        
        # Sort by exam rank (assuming lower is better)
        applications.sort(key=lambda x: x["exam_rank"])
        
        # Return only the requested number of applications
        return pd.DataFrame(applications[:limit])
        
    except Exception as e:
        st.error(f"Error fetching recent applications: {e}")
        return None

def get_pending_actions():
    """Get pending action items for the director"""
    # This would typically come from a database, but for demo purposes, we'll return static data
    return [
        {
            "title": "Approve Final Shortlist",
            "description": "Review and approve the final shortlist of candidates",
            "due_date": "April 15, 2023"
        },
        {
            "title": "Loan Budget Review",
            "description": "Review the loan budget allocation for the academic year",
            "due_date": "April 20, 2023"
        }
    ]

def create_status_chart(status_distribution):
    """Create a pie chart of application status distribution"""
    import plotly.express as px
    
    # Create a dataframe from the status distribution
    df = pd.DataFrame({
        "Status": list(status_distribution.keys()),
        "Count": list(status_distribution.values())
    })
    
    # Create the pie chart
    fig = px.pie(df, values="Count", names="Status", title="Application Status Distribution")
    return fig

def create_category_chart(category_distribution):
    """Create a bar chart of applications by category"""
    import plotly.express as px
    
    # Create a dataframe from the category distribution
    df = pd.DataFrame({
        "Category": list(category_distribution.keys()),
        "Count": list(category_distribution.values())
    })
    
    # Create the bar chart
    fig = px.bar(df, x="Category", y="Count", title="Applications by Category")
    return fig

def create_rank_distribution(rank_distribution):
    """Create a bar chart of exam rank distribution"""
    import plotly.express as px
    
    # Create a dataframe from the rank distribution
    df = pd.DataFrame({
        "Rank Range": rank_distribution["ranges"],
        "Count": rank_distribution["counts"]
    })
    
    # Create the bar chart
    fig = px.bar(df, x="Rank Range", y="Count", title="Exam Rank Distribution")
    return fig

def create_loan_chart(loan_status_distribution):
    """Create a pie chart of loan status distribution"""
    import plotly.express as px
    
    # Create a dataframe from the loan status distribution
    df = pd.DataFrame({
        "Status": list(loan_status_distribution.keys()),
        "Count": list(loan_status_distribution.values())
    })
    
    # Create the pie chart
    fig = px.pie(df, values="Count", names="Status", title="Loan Status Distribution")
    return fig



def modified_display_application_status():
    """Modified application status display function that uses the new PDF generation"""
    st.header("Check Application Status")
    
    # Option to use current application ID if available
    if 'current_application_id' in st.session_state:
        use_current_id = st.checkbox("Use current application ID", value=True)
        if use_current_id:
            application_id = st.session_state['current_application_id']
            st.info(f"Using application ID: {application_id}")
        else:
            application_id = st.text_input("Enter your Application ID")
    else:
        application_id = st.text_input("Enter your Application ID")
    
    check_status = st.button("Check Status")
    
    if check_status or ('last_checked_id' in st.session_state and st.session_state['last_checked_id'] == application_id):
        if not application_id:
            st.error("Please enter an Application ID")
            return
        
        # Store the ID being checked to maintain display after page reload
        st.session_state['last_checked_id'] = application_id
            
        # Query Pinecone for the application data
        application_data = query_pinecone(application_id)
        
        if application_data:
            try:
                st.success(f"Application found for ID: {application_id}")
                
                # Extract metadata from application_data
                metadata = application_data.get("metadata", {})
                
                # Check if we need to parse storage_data
                if "storage_data" in metadata and isinstance(metadata["storage_data"], str):
                    try:
                        parsed_data = json.loads(metadata["storage_data"])
                        metadata.update(parsed_data)
                    except json.JSONDecodeError:
                        st.warning("Could not parse stored application data")
                
                # Display personal information
                st.subheader("Personal Information")
                personal_data = safely_get_data(metadata, ['personal'], {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Name: {safely_get_data(personal_data, ['name'])}")
                    st.text(f"Email: {safely_get_data(personal_data, ['email'])}")
                    st.text(f"Phone: {safely_get_data(personal_data, ['phone'])}")
                
                with col2:
                    st.text(f"Category: {safely_get_data(personal_data, ['category'])}")
                    st.text(f"Date of Birth: {safely_get_data(personal_data, ['dob'])}")
                    st.text(f"Gender: {safely_get_data(personal_data, ['gender'])}")
                
                # Display academic information
                st.subheader("Academic Information")
                academic_data = safely_get_data(metadata, ['academic'], {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Exam Type: {safely_get_data(academic_data, ['exam_type'])}")
                    st.text(f"Exam Rank: {safely_get_data(academic_data, ['exam_rank'])}")
                    st.text(f"Board Percentage: {safely_get_data(academic_data, ['board_percentage'])}%")
                
                with col2:
                    st.text(f"Physics Marks: {safely_get_data(academic_data, ['physics_marks'])}")
                    st.text(f"Chemistry Marks: {safely_get_data(academic_data, ['chemistry_marks'])}")
                    st.text(f"Mathematics Marks: {safely_get_data(academic_data, ['maths_marks'])}")
                
                # Display application status
                st.subheader("Application Status")
                status_data = safely_get_data(metadata, ['status'], {})
                
                # Show overall status
                overall_status = safely_get_data(status_data, ['application_status'], "Submitted")
                
                # Create status indicator
                if overall_status == "Accepted":
                    st.success(f"Overall Status: {overall_status}")
                elif overall_status.startswith("Rejected"):
                    st.error(f"Overall Status: {overall_status}")
                elif overall_status == "Processed":
                    st.info(f"Overall Status: {overall_status}")
                else:
                    st.warning(f"Overall Status: {overall_status}")
                
                # Show detailed status in columns
                status_cols = st.columns(4)
                
                with status_cols[0]:
                    doc_status = safely_get_data(status_data, ['document_verification', 'status'], "Pending")
                    st.text(f"Documents: {doc_status}")
                
                with status_cols[1]:
                    elig_status = safely_get_data(status_data, ['eligibility', 'status'], "Pending")
                    st.text(f"Eligibility: {elig_status}")
                
                with status_cols[2]:
                    short_status = safely_get_data(status_data, ['shortlisting', 'status'], "Pending")
                    st.text(f"Shortlisting: {short_status}")
                
                with status_cols[3]:
                    loan_required = safely_get_data(metadata, ['loan', 'required'], False)
                    if isinstance(loan_required, str):
                        loan_required = loan_required.lower() == "true"
                    
                    if loan_required:
                        loan_status = safely_get_data(metadata, ['loan', 'status', 'status'], "Pending")
                        st.text(f"Loan: {loan_status}")
                    else:
                        st.text("Loan: Not Applied")
                
                # Generate PDF for application report
                pdf_buffer = generate_application_pdf(application_id)
                
                # Prominently display feedback from the feedback agent
                feedback_data = safely_get_data(status_data, ['feedback'], {})
                
                if feedback_data and feedback_data != "N/A":
                    st.subheader("📋 Application Feedback")
                    
                    detailed_feedback = safely_get_data(feedback_data, ['detailed_feedback'], "")
                    
                    if detailed_feedback:
                        with st.expander("Detailed Feedback", expanded=True):
                            st.markdown(f"**{detailed_feedback}**")
                    
                    # Display structured feedback information
                    improvement_areas = safely_get_data(feedback_data, ['improvement_areas'], [])
                    alternative_paths = safely_get_data(feedback_data, ['alternative_paths'], [])
                    reapplication_guidance = safely_get_data(feedback_data, ['reapplication_guidance'], "")
                    
                    if improvement_areas:
                        with st.expander("Areas for Improvement", expanded=True):
                            for area in improvement_areas:
                                st.markdown(f"- {area}")
                    
                    if alternative_paths:
                        with st.expander("Alternative Paths", expanded=True):
                            for path in alternative_paths:
                                st.markdown(f"- {path}")
                    
                    if reapplication_guidance:
                        with st.expander("Reapplication Guidance", expanded=True):
                            st.markdown(reapplication_guidance)
                else:
                    st.info("No feedback available yet. Feedback will be provided once your application is processed.")
                
                # Display PDF view and download options
                if pdf_buffer:
                    st.subheader("Application Report PDF")
                    
                    # Create tabs for viewing and downloading
                    pdf_tabs = st.tabs(["View PDF", "Download PDF"])
                    
                    with pdf_tabs[0]:
                        # Display the PDF
                        pdf_html = display_pdf(pdf_buffer)
                        if pdf_html:
                            st.markdown(pdf_html, unsafe_allow_html=True)
                        else:
                            st.error("Unable to display PDF. Try downloading instead.")
                    
                    with pdf_tabs[1]:
                        # Provide download link with custom styling
                        st.markdown("""
                        <style>
                        .btn {
                            display: inline-block;
                            padding: 10px 20px;
                            background-color: #3366cc;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            margin: 10px 0;
                            font-weight: bold;
                        }
                        .btn:hover {
                            background-color: #254b99;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        download_link = get_pdf_download_link(pdf_buffer, f"application_report_{application_id}.pdf")
                        if download_link:
                            st.markdown(download_link, unsafe_allow_html=True)
                            st.info("Click the button above to download your application report as a PDF file.")
                        else:
                            st.error("Unable to generate download link.")
                else:
                    st.error("Could not generate PDF report. Please try again later.")
                
                # Display eligibility details if available
                eligibility_data = status_data.get("eligibility", {})
                if isinstance(eligibility_data, str):
                    try:
                        eligibility_data = json.loads(eligibility_data)
                    except:
                        eligibility_data = {}
                
                if eligibility_data:
                    status = eligibility_data.get("status", "Pending")
                    reasons = eligibility_data.get("reasons", [])
                    missing_criteria = eligibility_data.get("missing_criteria", [])
                    
                    if reasons or missing_criteria:
                        st.subheader("Eligibility Details")
                        
                        if status == "Eligible":
                            st.success("You meet all eligibility criteria")
                            if reasons:
                                st.subheader("Qualification Factors")
                                for reason in reasons:
                                    st.markdown(f"- {reason}")
                        else:
                            st.error("You do not meet all eligibility criteria")
                            if missing_criteria:
                                st.subheader("Missing Criteria")
                                for criterion in missing_criteria:
                                    st.markdown(f"- {criterion}")
                
                # Display shortlisting details if available
                shortlisting_data = status_data.get("shortlisting", {})
                if isinstance(shortlisting_data, str):
                    try:
                        shortlisting_data = json.loads(shortlisting_data)
                    except:
                        shortlisting_data = {}
                
                if shortlisting_data:
                    status = shortlisting_data.get("status", "Pending")
                    priority_score = shortlisting_data.get("priority_score", 0)
                    remarks = shortlisting_data.get("remarks", "")
                    
                    if status != "Pending":
                        st.subheader("Shortlisting Details")
                        
                        if status == "Shortlisted":
                            st.success(f"You have been shortlisted with a priority score of {priority_score}")
                        else:
                            st.error("You were not shortlisted")
                        
                        if remarks:
                            st.markdown(f"**Remarks**: {remarks}")
                
                # Display loan details if applied
                loan_data = metadata.get("loan", {})
                if isinstance(loan_data, str):
                    try:
                        loan_data = json.loads(loan_data)
                    except:
                        loan_data = {}
                
                loan_required = loan_data.get("required", False)
                if isinstance(loan_required, str):
                    loan_required = loan_required.lower() == "true"
                
                if loan_required:
                    st.subheader("Loan Information")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Loan Amount Requested: ₹{loan_data.get('loan_amount', 0):,}")
                        st.text(f"Family Income: ₹{loan_data.get('family_income', 0):,}")
                    
                    with col2:
                        st.text(f"Guarantor: {loan_data.get('guarantor_name', 'N/A')}")
                        st.text(f"Guarantor Relation: {loan_data.get('guarantor_relation', 'N/A')}")
                    
                    # Display loan status if available
                    loan_status_data = loan_data.get("status", {})
                    if isinstance(loan_status_data, str):
                        try:
                            loan_status_data = json.loads(loan_status_data)
                        except:
                            loan_status_data = {}
                    
                    if loan_status_data:
                        status = loan_status_data.get("status", "Pending")
                        if status == "Approved":
                            st.success(f"Loan Status: {status}")
                            st.info(f"Approved Amount: ₹{loan_status_data.get('approved_amount', 0):,}")
                        elif status == "Rejected":
                            st.error(f"Loan Status: {status}")
                            
                            # Display rejection reasons if available
                            reasons = loan_status_data.get("reasons", [])
                            if reasons:
                                st.subheader("Rejection Reasons")
                                for reason in reasons:
                                    st.markdown(f"- {reason}")
                            
                            # Display alternative aid suggestions if available
                            alt_aid = loan_status_data.get("alternative_aid", [])
                            if alt_aid:
                                st.subheader("Alternative Financial Aid")
                                for aid in alt_aid:
                                    st.markdown(f"- {aid}")
                        else:
                            st.warning(f"Loan Status: {status}")
                            
            except Exception as e:
                st.error(f"Error processing application data: {str(e)}")
                st.error(f"Error details: {traceback.format_exc()}")
        else:
            st.error(f"No application found for ID: {application_id}")
def display_about_page():
    st.header("About the Student Admission AI System")
    
    st.subheader("Admission Criteria")
    st.markdown(ELIGIBILITY_CRITERIA)
    
    st.subheader("University Capacity")
    st.markdown(UNIVERSITY_CAPACITY)
    
    st.subheader("Loan Criteria")
    st.markdown(LOAN_CRITERIA)
    
if __name__ == "__main__":
    main()
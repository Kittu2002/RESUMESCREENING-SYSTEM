import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader
from docx import Document
import re
import cv2

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

image = cv2.imread("D:/final year project/Resume-Screening-App-main/WhatsApp Image 2024-03-25 at 4.22.49 PM (1).jpeg")

# Display the image as the main heading
st.image(image,use_column_width=True,channels='BGR')#use_column_width=True


page_bg_img=f"""
<style>

[data-testid="stAppViewContainer"] {{
    background-color:#ADD8E6;
    background-size:cover;
    }}



</style>
"""

st.markdown(page_bg_img,unsafe_allow_html=True)

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text



def calculate_ats_score(resume_text, job_description):
    # Preprocess text: convert to lowercase and remove non-alphanumeric characters
    
    resume_text = re.sub(r'\W+', ' ', resume_text.lower())
    job_description = re.sub(r'\W+', ' ', job_description.lower())
    
    # Split text into individual words
    resume_words = set(resume_text.split())
    job_words = set(job_description.split())
    
    # Calculate the number of matching words
    matching_words = resume_words.intersection(job_words)

    if len(job_words) == 0:
        return "Enter Correct Job Description"
    # Calculate ATS score: percentage of matching words in the job description
    ats_score = (len(matching_words) / len(job_words)) * 100
    
    return ats_score
# web app

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>RESUME SCREENING SYSTEM</h1>", unsafe_allow_html=True)

    job_description=st.text_input("**Job Description**")
    
    uploaded_file = st.file_uploader('**Upload Resume**', type=['pdf', 'docx'])

    resume_text=None
    

    if uploaded_file is not None:
        name = uploaded_file.name
        if('.pdf' in name):
            reader = PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            resume_text = ''
            for i in range(num_pages):
                page = reader.pages[i]
                resume_text += page.extract_text()
        else:
            resume_text = extract_text_from_docx(uploaded_file)
        #print(resume_text)
        ats_score=calculate_ats_score(resume_text,job_description)
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume]) 

 
        
        #st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        

    if st.button("**SCAN**"):

        try:

            ats_score=calculate_ats_score(resume_text,job_description)

            prediction_id = clf.predict(input_features)[0]

            category_name = category_mapping.get(prediction_id, "Unknown")
            

            if int(ats_score) > 55:
                result = "ATS score of this resume for this job is :" + str(ats_score)
                st.success(result)
                st.write("**Congrats to you to successfully pass the first stage of Recruitment process**")
                st.write("**Our Recruitment Team will contact you soon,please stay tune for further updates**")

            else:

                result = "ATS score of given resume for given job is :" + str(ats_score)
                st.error(result)
                st.write("**Sorry to say this,We cannot proceed with your candidature**")

        except:

            st.error("**BOTH JOB DESCRIPTION AND RESUME ARE REQUIRED FOR ATS SCORE**")

        if resume_text is not None:

            st.write("**Predicted Job Category that suitable to corresponding Resume candidate is:** <span style='color:Green'>{}</span>".format(category_name), unsafe_allow_html=True)


# python main
if __name__ == "__main__":
    main()

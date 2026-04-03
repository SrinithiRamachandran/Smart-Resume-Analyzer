import streamlit as st
import nltk
from nltk.util import ngrams
import spacy
import demo
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import base64, random
import time, datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import (
    ds_course, web_course, android_course, ios_course, uiux_course,
    cloud_course, cybersecurity_course, data_engineering_course, devops_course,
    ai_course, db_admin_course, networking_course, bi_course, game_dev_course,
    resume_videos, interview_videos
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.cluster import KMeans
import pafy
import plotly.express as px
import yt_dlp
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial']
# or you can use 'DejaVu Sans' as a fallback
 # or another font that supports emojis
import re
# Load the job data from CSV file
@st.cache_data
def load_job_data_from_csv():
    df = pd.read_csv('merged_dataset.csv')  # Make sure your CSV file is in the correct location
    return df

# 2. Job Distribution Analysis
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data(data_path="job_descriptions.csv"):
    df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]
    return df

def get_resume_skills():
    return st.session_state.get('matched_technical_skills', [])

def display_top_matching_jobs(df):
    st.subheader("\U0001F50D Top 10 Matching Jobs Based on Your Skills")
    resume_skills = get_resume_skills()

    if not resume_skills:
        st.warning("No skills found from your resume.")
        return

    if 'job_skills' not in df.columns:
        st.error("'job_skills' column not found!")
        return

    df = df.dropna(subset=['job_skills'])
    df['job_skills'] = df['job_skills'].astype(str).str.lower()
    resume_skills_text = " ".join(resume_skills).lower()

    vectorizer = TfidfVectorizer()
    job_skill_vectors = vectorizer.fit_transform(df['job_skills'])
    resume_vector = vectorizer.transform([resume_skills_text])
    similarities = cosine_similarity(resume_vector, job_skill_vectors).flatten()
    df['similarity_score'] = similarities
    top_matches = df.sort_values(by='similarity_score', ascending=False).head(10)

    for _, row in top_matches.iterrows():
        st.markdown(f"### {row.get('job_title', 'Unknown')} at {row.get('company', 'Unknown')}")
        st.markdown(f"**Location:** {row.get('job_location', 'N/A')} | **Type:** {row.get('job_type', 'N/A')} | **Level:** {row.get('job_level', 'N/A')}")
        st.markdown(f"**Required Skills:** {row.get('job_skills', 'N/A')}")
        st.markdown(f"**Match Score:** {round(row['similarity_score'] * 100, 2)}%")
        st.markdown("---")

def skill_demand_analysis(df):
    st.subheader("\U0001F4CA Most In-Demand Skills")
    skills = df['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
    top_skills = skills.value_counts().head(20)
    st.bar_chart(top_skills)

def job_role_insights(df):
    st.subheader("\U0001F4D1 Job Role vs Skills")
    job_roles = df['job_title'].str.lower().value_counts().head(5).index.tolist()
    for role in job_roles:
        st.markdown(f"### {role.title()}")
        role_df = df[df['job_title'].str.lower().str.contains(role)]
        skills = role_df['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
        st.bar_chart(skills.value_counts().head(10))

def location_based_analysis(df):
    st.subheader("\U0001F4CD Location-Based Job Trends")
    top_locations = df['job_location'].value_counts().head(10)
    st.bar_chart(top_locations)

    selected_city = st.selectbox("Select a city to explore skills:", top_locations.index)
    city_df = df[df['job_location'] == selected_city]
    city_skills = city_df['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
    st.bar_chart(city_skills.value_counts().head(10))

def company_demand_analysis(df):
    st.subheader("\U0001F3E2 Companies Hiring & Their Skill Demands")
    top_companies = df['company'].value_counts().head(5)
    for company in top_companies.index:
        st.markdown(f"### {company}")
        company_df = df[df['company'] == company]
        skills = company_df['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
        st.bar_chart(skills.value_counts().head(10))

def time_based_trend(df):
    st.subheader("\U0001F4C5 Job Posting Trends Over Time")
    if 'first_seen' in df.columns:
        df['first_seen'] = pd.to_datetime(df['first_seen'], errors='coerce')
        timeline = df.set_index('first_seen').resample('M').size()
        st.line_chart(timeline)

def skill_gap_analysis(df):
    st.subheader("🧠 Skill Gap Analysis")
    resume_skills = set(get_resume_skills())
    
    target_role = st.text_input("Enter a job role to compare against (e.g., data scientist):")
    if not target_role:
        return

    role_df = df[df['job_title'].str.lower().str.contains(target_role.lower())]
    if role_df.empty:
        st.warning(f"No job postings found for the role: {target_role}")
        return

    role_skills = role_df['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
    top_role_skills = set(role_skills.value_counts().head(20).index)

    missing_skills = top_role_skills - resume_skills
    matching_skills = top_role_skills & resume_skills

    st.markdown(f"### 🔧 Required Skills for '{target_role.title()}'")
    st.write(top_role_skills)
    
    st.markdown("### ✅ Skills You Have")
    st.write(matching_skills if matching_skills else "No matching skills found.")
    
    st.markdown("### ❌ Skills You’re Missing")
    st.write(missing_skills if missing_skills else "You have all the top required skills!")


def work_type_analysis(df):
    st.subheader("\U0001F3E0 Work Type Insights")
    if 'job_type' in df.columns:
        df['job_type'] = df['job_type'].str.lower()
        st.bar_chart(df['job_type'].value_counts())

        if 'remote' in df['job_type'].unique():
            remote_jobs = df[df['job_type'].str.contains("remote")]
            remote_skills = remote_jobs['job_skills'].dropna().str.lower().str.split(',').explode().str.strip()
            st.write("### Top Skills in Remote Jobs")
            st.bar_chart(remote_skills.value_counts().head(10))


def skill_gap_analysis1(df, user_skills, top_n=5):
    # Clean and prepare skills
    user_skills_str = ', '.join(user_skills).lower()

    # Combine job_skills and job_summary into a text corpus
    df['combined_text'] = (df['job_skills'].fillna('') + ' ' + df['job_summary'].fillna('')).str.lower()

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [user_skills_str] + df['combined_text'].tolist()
    X = vectorizer.fit_transform(corpus)

    # Cosine similarity
    cosine_sim = cosine_similarity(X[0:1], X[1:])
    top_matches = np.argsort(cosine_sim[0])[::-1][:top_n]
    top_jobs = df.iloc[top_matches].copy()

    # Identify missing skills
    results = []
    for idx, row in top_jobs.iterrows():
        required_skills = set(map(str.strip, str(row['job_skills']).lower().split(',')))
        user_skills_set = set(skill.strip().lower() for skill in user_skills)
        missing = required_skills - user_skills_set
        results.append({
            'Job Title': row['job_title'],
            'Company': row['company'],
            'Location': row['job_location'],
            'Required Skills': ', '.join(required_skills),
            'Missing Skills (Skill Gap)': ', '.join(missing) if missing else 'None'
        })

    return pd.DataFrame(results)

def job_distribution_analysis(df):
    # ----------------------------
    # Job Titles Distribution
    # ----------------------------
    st.write("### 📊 Job Titles Distribution")
    title_counts = df['job_title'].value_counts().head(10)  # Top 10 only

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(title_counts.index, title_counts.values, color='skyblue')
    ax.set_title("Job Titles Distribution")
    ax.set_xlabel("Job Titles")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------
    # Job Locations Distribution
    # ----------------------------
    st.write("### 📍 Job Locations Distribution")
    location_counts = df['job_location'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(location_counts.index, location_counts.values, color='salmon')
    ax.set_title("Top 10 Job Locations")
    ax.set_xlabel("Locations")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------
    # Job Types Distribution
    # ----------------------------
    st.write("### 🧾 Job Types Distribution")
    job_type_counts = df['job_type'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(job_type_counts.index, job_type_counts.values, color='limegreen')
    ax.set_title("Job Types Distribution")
    ax.set_xlabel("Job Types")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------
    # Experience Levels
    # ----------------------------
    st.write("### 📈 Experience Levels Distribution")
    level_counts = df['job_level'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(level_counts.index, level_counts.values, color='orange')
    ax.set_title("Experience Levels Distribution")
    ax.set_xlabel("Experience Levels")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)


def skills_analysis(df):
    st.write("### 🛠️ Top 10 Skills Distribution")

    def extract_skills(job_skills):
        return [skill.strip().lower() for skill in job_skills.split(',')]

    df['skills_list'] = df['job_skills'].apply(lambda x: extract_skills(x) if pd.notnull(x) else [])
    all_skills = [skill for sublist in df['skills_list'] for skill in sublist]
    skill_counts = Counter(all_skills)
    top_skills = skill_counts.most_common(10)
    top_skills_df = pd.DataFrame(top_skills, columns=["Skill", "Frequency"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_skills_df['Skill'], top_skills_df['Frequency'], color='mediumslateblue')
    ax.set_title("Top 10 Skills from Job Listings")
    ax.set_xlabel("Skills")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)
def job_clustering(df):

    def vectorize_jobs(job_data):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        return vectorizer.fit_transform(job_data)

    # Combine text for clustering (including job skills, summary, company, and location)
    df['combined_text'] = df['job_skills'].fillna('') + " " + df['job_summary'].fillna('') + " " + df['company'].fillna('') + " " + df['job_location'].fillna('')

    if df['combined_text'].str.strip().eq('').all():
        st.warning("Job data has no skills or summaries to cluster.")
        return

    X = vectorize_jobs(df['combined_text'])

    # Clustering
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Plot cluster distribution
    st.write("### Job Clusters Based on Skills, Descriptions, Company, and Location")
    st.markdown("We group jobs into 5 clusters using KMeans based on TF-IDF of their skills, summaries, companies, and locations.")

    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_df = pd.DataFrame({'Cluster': cluster_counts.index, 'Job Count': cluster_counts.values})

    if cluster_df.empty:
        st.warning("No cluster results available.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cluster_df['Cluster'], cluster_df['Job Count'], color='skyblue')
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Number of Jobs per Cluster")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    # Top job titles per cluster (with full details)
    for cluster_id in range(num_clusters):
        st.markdown(f"#### Cluster {cluster_id} - Top Job Titles")
        
        # Filter jobs in the current cluster
        cluster_jobs = df[df['cluster'] == cluster_id]
        
        # Remove duplicate job titles (including company and location)
        cluster_jobs_unique = cluster_jobs.drop_duplicates(subset=['job_title', 'company', 'job_location'])
        
        # Display unique job titles (avoid repeating the same job title)
        top_jobs = cluster_jobs_unique.head(5)  # Limit to top 5 unique jobs

        if top_jobs.empty:
            st.write(f"No unique jobs found in Cluster {cluster_id}.")
        else:
            st.dataframe(top_jobs[['job_title', 'company', 'job_location', 'job_type', 'job_level', 'job_summary']])


# 5. Resume Skill Matching to Jobs
def match_jobs_to_skills(df, resume_skills):
    # Vectorize the resume skills and job descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    job_descriptions = df['combined_text'].fillna('')
    corpus = [resume_skills] + job_descriptions.tolist()
    X = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between resume and job descriptions
    cosine_sim = cosine_similarity(X[0:1], X[1:])

    # Get top 5 job matches
    top_matches = np.argsort(cosine_sim[0])[::-1][:5]  # Top 5 matches
    matched_jobs = df.iloc[top_matches]

    return matched_jobs

# 6. Save Analysis Results to CSV
def save_to_csv(df):
    df.to_csv('job_analysis_results.csv', index=False)

def fetch_yt_video(url):
    ydl_opts = {}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)
            return video_title
    except Exception as e:
        st.error(f"Error fetching video details: {e}")
        return "Unknown Title"

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4, key=f"slider_{course_list[0][0]}")

    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

connection = pymysql.connect(host='localhost', user='root', password='srinithi2005')
cursor = connection.cursor()

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills,
                courses):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (
    name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills,
    courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='./Logo/SRA_Logo.ico',
)
technical_skills = [
    # Programming Languages
    'python', 'java', 'c++', 'c', 'r', 'sql', 'javascript', 'typescript', 'bash', 'perl', 'go', 'rust',

    # Web Development
    'html', 'css', 'sass', 'bootstrap', 'jquery', 'react', 'vue.js', 'angular', 'next.js', 'express.js', 
    'node.js', 'php', 'django', 'flask', 'wordpress', 'graphql', 'webpack', 'redux', 'tailwind css', 'nestjs', 
    'ruby on rails', 'spring boot',

    # Data Science & Machine Learning
    'machine learning', 'deep learning', 'data science', 'data analysis', 'tensorflow', 'keras', 'pytorch', 
    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'nlp', 'computer vision', 

    # Data Engineering & Big Data
    'etl', 'apache airflow', 'apache kafka', 'apache spark', 'hadoop', 'sqoop', 'flume', 'apache hive', 'impala', 
    'cassandra', 'databricks', 'data lakes', 'data warehousing', 'dbt', 'google dataflow', 'aws glue', 'airbyte',

    # Databases
    'mysql', 'postgresql', 'mongodb', 'elasticsearch', 'oracle', 'sqlite', 'redis', 'couchdb',

    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'gitlab ci', 'ci/cd', 
    'vagrant', 'aws lambda', 'aws ec2', 'aws s3', 'prometheus', 'grafana', 'helm', 'logstash', 'kibana', 'openShift', 
    'rancher', 'firebase', 'digital ocean', 'heroku',

    # MLOps
    'mlops', 'model deployment', 'tensorflow serving', 'flask api', 'django api', 'kubeflow', 'azure ml', 'gcp ai', 
    'model monitoring', 'model versioning', 'model tracking',

    # Mobile Development
    'android', 'ios', 'flutter', 'kotlin', 'swift', 'react native', 'xamarin', 'ionic',

    # Visualization / BI Tools
    'tableau', 'power bi', 'data visualization', 'matplotlib', 'seaborn',

    # Cybersecurity & Networking
    'cybersecurity', 'ethical hacking', 'penetration testing', 'network security', 'cloud security', 'firewall', 
    'vpn', 'incident response', 'burp suite', 'nmap', 'wireshark',

    # Project & Team Tools
    'git', 'jira', 'trello', 'asana', 'slack', 'microsoft project', 'agile', 'scrum', 'kanban', 'devops',

    # Design / UI-UX
    'figma', 'adobe xd', 'photoshop', 'illustrator', 'sketch', 'balsamiq', 'zeplin',

    # QA / Testing
    'selenium', 'pytest', 'junit', 'mocha', 'cypress', 'karma',

    # Enterprise Tools (optional for enterprise roles)
    'sap', 'salesforce', 'zoho', 'oracle ebs', 'service now', 'bmc remedy'
]


def extract_skills_section(text):
    skill_section = ''
    patterns = [r'(skills|technical skills|core competencies|key skills|expertise)\s*\n*(.*?)(?:\n\s*\n|\n[A-Z])', 
                r'(skills|technical skills|expertise)\s*:\s*(.*?)\n']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            skill_section += match.group(2)
    return [s.strip() for s in re.split(r',|;|\n', skill_section) if s.strip()]

# --- Step 2: Tokenization ---
def tokenize_resume(resume_text):
    resume_text = resume_text.lower()
    resume_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', resume_text)
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = word_tokenize(resume_text)
    return [word for word in tokens if word not in stopwords]

# --- Step 3: Extract n-grams ---
def extract_ngrams(tokens, max_n=3):
    ngram_list = []
    for n in range(1, max_n + 1):
        ngram_list += [' '.join(gram) for gram in ngrams(tokens, n)]
    return ngram_list

# --- Step 4: Extract noun phrases ---
def extract_noun_phrases(text):
    doc = nlp(text)
    return [chunk.text.lower().strip() for chunk in doc.noun_chunks]

# --- Step 5: Fuzzy match phrases to skill list ---
def fuzzy_match_skills(phrases, skill_list, threshold=80):
    matched = set()
    skill_map = {skill.lower(): skill for skill in skill_list}
    for phrase in phrases:
        cleaned = phrase.lower().strip()
        for skill_key, original_skill in skill_map.items():
            if fuzz.token_set_ratio(cleaned, skill_key) >= threshold:
                matched.add(original_skill)
                break
    return list(matched)

# --- Main Function: Extract technical skills ---
def extract_technical_skills_from_resume(resume_text):
    # Extract phrases from "Skills" section (if exists)
    skills_section_phrases = extract_skills_section(resume_text)

    # Extract all possible text-based phrases
    tokens = tokenize_resume(resume_text)
    ngram_phrases = extract_ngrams(tokens)
    noun_phrases = extract_noun_phrases(resume_text)
    combined_phrases = list(set(ngram_phrases + noun_phrases))

    # Fuzzy match against technical skills
    matched_from_section = fuzzy_match_skills(skills_section_phrases, technical_skills)
    matched_from_text = fuzzy_match_skills(combined_phrases, technical_skills)

    # Merge both results
    all_matched = set(matched_from_section + matched_from_text)
    
    return sorted(all_matched)



def match_domains(resume_skills):
    # Define all domains with their keywords
    domain_keywords = {
    'Data Science': [
        'python', 'r', 'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning',
        'flask', 'streamlit', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'data preprocessing', 'data analysis', 'statistics', 'predictive modeling',
        'data mining', 'jupyter notebook', 'big data'
    ],
    'Web Development': [
        'html', 'css', 'javascript', 'typescript', 'react', 'vue', 'angular', 'node js', 'express js',
        'django', 'flask', 'php', 'laravel', 'mysql', 'mongodb', 'postgresql', 'graphql',
        'rest api', 'soap', 'bootstrap', 'sass', 'webpack', 'frontend', 'backend', 'full stack',
        'wordpress', 'magento', 'web development'
    ],
    'Android Development': [
        'android', 'android development', 'java', 'kotlin', 'flutter', 'dart', 'xml',
        'android studio', 'jetpack compose', 'android sdk', 'material design',
        'firebase', 'mvvm', 'mvp', 'retrofit', 'room database'
    ],
    'IOS Development': [
        'ios', 'ios development', 'swift', 'swiftui', 'objective-c', 'xcode', 'cocoa', 'cocoa touch',
        'core data', 'storyboard', 'auto layout', 'mvvm', 'cloudkit', 'firebase'
    ],
    'UI-UX Development': [
        'ux', 'ui', 'figma', 'adobe xd', 'zeplin', 'balsamiq', 'wireframes', 'prototyping',
        'adobe photoshop', 'adobe illustrator', 'sketch', 'user research', 'user testing',
        'design systems', 'interaction design', 'storyframes', 'design thinking', 'affinity mapping'
    ],
    'Cloud Computing': [
        'aws', 'azure', 'google cloud', 'gcp', 'cloud computing', 'cloud services', 'cloud infrastructure',
        'ec2', 's3', 'lambda', 'cloud functions', 'iam', 'terraform', 'cloudformation',
        'docker', 'kubernetes', 'devops', 'vpc', 'load balancer', 'cloud security'
    ],
    'Cybersecurity': [
        'firewall', 'encryption', 'penetration testing', 'ethical hacking', 'network security',
        'malware analysis', 'incident response', 'siem', 'vulnerability assessment', 'nmap',
        'burpsuite', 'wireshark', 'kali linux', 'snort', 'forensics', 'access control',
        'security compliance', 'cybersecurity'
    ],
    'Data Engineering': [
        'python', 'sql', 'scala', 'java', 'data pipeline', 'etl', 'apache spark', 'hadoop',
        'big data', 'data warehouse', 'data lake', 'airflow', 'dbt', 'kafka',
        'redshift', 'snowflake', 'databricks', 'cloud storage', 'parquet', 'avro'
    ],
    'DevOps': [
        'devops', 'docker', 'kubernetes', 'jenkins', 'git', 'github actions', 'ci/cd',
        'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'scripting',
        'bash', 'shell scripting', 'linux', 'monitoring', 'deployment automation'
    ],
    'Artificial Intelligence': [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
        'nlp', 'cv', 'computer vision', 'reinforcement learning', 'scikit-learn',
        'tensorflow', 'keras', 'pytorch', 'openai', 'huggingface', 'transformers', 'gpt', 'llm'
    ],
    'Database Administration': [
        'mysql', 'postgresql', 'sql server', 'oracle', 'mongodb', 'redis', 'database administration',
        'database management', 'backup and recovery', 'database security', 'query optimization',
        'performance tuning', 'pl/sql', 't-sql', 'nosql', 'database design', 'dba'
    ],
    'Networking': [
        'ccna', 'networking', 'routing', 'switching', 'tcp/ip', 'udp', 'dns', 'dhcp',
        'firewalls', 'load balancing', 'vpn', 'network configuration', 'network security',
        'wireshark', 'osi model', 'bgp', 'ospf', 'network troubleshooting', 'lan', 'wan'
    ],
    'Business Intelligence': [
        'bi', 'business intelligence', 'tableau', 'power bi', 'lookml', 'data studio',
        'data visualization', 'dashboarding', 'reporting', 'sql', 'etl', 'data modeling',
        'kpi', 'dax', 'excel', 'pivot tables', 'data analysis expressions'
    ],
    'Game Development': [
        'unity', 'unreal engine', 'c#', 'c++', 'game development', 'game design', 'game programming',
        '2d', '3d', 'gamification', 'blender', 'animation', 'physics engine', 'game loop',
        'level design', 'multiplayer', 'mobile game development', 'shader programming'
    ]
    
    }


    # Normalize resume skills
    skills_lower = [skill.lower() for skill in resume_skills if isinstance(skill, str)]

    domain_scores = {}
    domain_matches = {}

    # Calculate match scores
    for domain, keywords in domain_keywords.items():
        matched = set(skills_lower).intersection(set(keywords))
        domain_scores[domain] = len(matched)
        domain_matches[domain] = matched

    # Sort domains by match score
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_domains, domain_matches, domain_keywords

def display_top_domains(sorted_domains, domain_matches, domain_keywords, course_recommender):
    reco_field = None
    recommended_skills = []

    for i, (domain, score) in enumerate(sorted_domains[:2]):  # Top 2 matching domains
        if score == 0:
            continue
        
        fit_position = "Best Fit" if i == 0 else "Second Best Fit"
        st.subheader(f"{fit_position}: {domain}")

        matched_skills = list(domain_matches[domain])
        all_skills = set(domain_keywords[domain])
        missing_skills = list(all_skills - domain_matches[domain])

        st.markdown(f"**✅ Matched Skills for {domain}:** {', '.join(matched_skills)}")
        st.markdown(f"**🚀 Recommended to Learn for {domain}:** {', '.join(missing_skills)}")

        st.markdown(
            '''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost💼 your job chances!</h4>''',
            unsafe_allow_html=True
        )

        # Course recommendation
        if domain == "Data Science":
            rec_course = course_recommender(ds_course)
        elif domain == "Web Development":
            rec_course = course_recommender(web_course)
        elif domain == "Android Development":
            rec_course = course_recommender(android_course)
        elif domain == "IOS Development":
            rec_course = course_recommender(ios_course)
        elif domain == "UI-UX Development":
            rec_course = course_recommender(uiux_course)
        elif domain == "Cloud Computing":
            rec_course = course_recommender(cloud_course)
        elif domain == "Cybersecurity":
            rec_course = course_recommender(cybersecurity_course)
        elif domain == "Data Engineering":
            rec_course = course_recommender(data_engineering_course)
        elif domain == "DevOps":
            rec_course = course_recommender(devops_course)
        elif domain == "Artificial Intelligence":
            rec_course = course_recommender(ai_course)
        elif domain == "Database Administration":
            rec_course = course_recommender(db_admin_course)
        elif domain == "Networking":
            rec_course = course_recommender(networking_course)
        elif domain == "Business Intelligence":
            rec_course = course_recommender(bi_course)
        elif domain == "Game Development":
            rec_course = course_recommender(game_dev_course)

        st.markdown(f"**💡 Recommended Courses for {domain}:** {rec_course}")
        
        # Capture the best-fit values to return for DB
        if i == 0:
            reco_field = domain
            recommended_skills = missing_skills
            r_course=rec_course

    return reco_field, recommended_skills,r_course
def section_present(section_keywords, resume_text):
    # Check if any keyword in section_keywords matches the resume_text
    return any(re.search(rf"\b{re.escape(keyword)}\b", resume_text, re.IGNORECASE) for keyword in section_keywords)



def run():
    st.title("Smart Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["resume parsing", "Dataset Analysis","bigdataset analysis","Salary Prediction","analysis2","Admin"] 
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    img = Image.open('./Logo/SRA_Logo.jpg')
    img = img.resize((250, 250))
    st.image(img)

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS SRA;"""
    cursor.execute(db_sql)
    connection.select_db("sra")

    # Create table
    DB_table_name = 'resume parsing'
    table_sql = "CREATE TABLE IF NOT EXISTS `" + DB_table_name + """` (
        ID INT NOT NULL AUTO_INCREMENT,
        Name VARCHAR(100) NOT NULL,
        Email_ID VARCHAR(50) NOT NULL,
        resume_score VARCHAR(8) NOT NULL,
        Timestamp VARCHAR(50) NOT NULL,
        Page_no VARCHAR(5) NOT NULL,
        Predicted_Field VARCHAR(25) NOT NULL,
        User_level VARCHAR(30) NOT NULL,
        Actual_skills VARCHAR(300) NOT NULL,
        Recommended_skills VARCHAR(300) NOT NULL,
        Recommended_courses VARCHAR(600) NOT NULL,
        PRIMARY KEY (ID)
    );
    """
    cursor.execute(table_sql)


    if choice == "resume parsing":
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            show_pdf(save_image_path)

            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.success("Hello " + resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: ' + resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: ' + str(resume_data['no_of_pages']))
                except:
                    pass

                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''', unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''', unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''', unsafe_allow_html=True)

            
                st.subheader("**Skills Recommendation💡**")

                # 🔁 STEP 2: Extract cleaned technical skills from resume text
                matched_technical_skills = extract_technical_skills_from_resume(resume_text)
                st.session_state['matched_technical_skills'] = matched_technical_skills
                # 🔁 STEP 3: Show cleaned skills
                keywords = st_tags(
                    label='### Filtered Technical Skills',
                    text='These are your extracted technical skills (auto-filtered)',
                    value=matched_technical_skills,
                    key='1'
                )

                
                
                #domain finding
                sorted_domains, domain_matches, domain_keywords = match_domains(resume_data['skills'])
                reco_field,recommended_skills,rec_course=display_top_domains(sorted_domains, domain_matches, domain_keywords, course_recommender)
                
                #
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date + '_' + cur_time)

                def section_present(section_keywords, resume_text):
                    import re
                    return any(re.search(rf"\b{keyword}\b", resume_text, re.IGNORECASE) for keyword in section_keywords)

                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0

                if section_present(['objective', 'career objective'], resume_text):
                    resume_score += 10
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please add a Career Objective to give clear intent to recruiters.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['experience', 'work experience', 'professional experience'], resume_text):
                    resume_score += 15
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Experience section. Recruiters value hands-on work!</h4>''',
                        unsafe_allow_html=True)

                if section_present(['education', 'academic background', 'qualifications'], resume_text):
                    resume_score += 15
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please include your Education details.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['projects', 'academic projects', 'personal projects'], resume_text):
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Projects👨‍💻</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Projects to show your technical capability.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['skills', 'technical skills', 'core skills'], resume_text):
                    resume_score += 15
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please highlight your Skills to match the job profile.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['achievements', 'awards', 'recognition'], resume_text):
                    resume_score += 15
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Achievements🏅</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Consider adding Achievements. It strengthens your resume.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['certifications', 'certificates'], resume_text):
                    resume_score += 5
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Certifications</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Please include Certifications if available.</h4>''',
                        unsafe_allow_html=True)

                if section_present(['hobbies', 'interests'], resume_text):
                    resume_score += 5
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Hobbies⚽</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] Consider adding Hobbies/Interests to show personality.</h4>''',
                        unsafe_allow_html=True)

                st.subheader("**Resume Score📝**")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )

                my_bar = st.progress(0)
                for percent_complete in range(resume_score):
                    time.sleep(0.03)
                    my_bar.progress(percent_complete + 1)

                st.success('** Your Resume Writing Score: ' + str(resume_score) + ' / 100**')
                st.warning(
                    "**Note: This score is calculated based on the content that you have added in your Resume.**")
                st.balloons()

                insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                            str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                            str(recommended_skills), str(rec_course))

                ## Resume writing video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ **" + res_vid_title + "**")
                st.video(resume_vid)

                ## Interview Preparation Video
                st.header("**Bonus Video for Interview👨‍💼 Tips💡**")
                interview_vid = random.choice(interview_videos)
                int_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ **" + int_vid_title + "**")
                st.video(interview_vid)

                connection.commit()
            else:
                st.error('Something went wrong..')
    elif choice == "Dataset Analysis":
        st.subheader("📊 Dataset Analysis")

        df = load_job_data_from_csv()

        st.markdown("### 🔍 Job Distribution Overview")
        job_distribution_analysis(df)

        st.markdown("### 💡 Skills Analysis")
        skills_analysis(df)

        st.markdown("### 🔗 Job Clustering")
        job_clustering(df)

        st.markdown("---")
       
        st.markdown("### 🎯 Job Matching Based on Skills")

        # Get skills from session_state if present
        resume_skills_from_file = st.session_state.get('matched_technical_skills', [])

        col1, col2 = st.columns(2)

        with col1:
            use_resume_skills = False
            if resume_skills_from_file:
                use_resume_skills = st.checkbox("✅ Use skills extracted from uploaded resume")
                if use_resume_skills:
                    st.success("Using resume skills:")
                    st.write(resume_skills_from_file)

        with col2:
            resume_skills_input = st.text_area("✍️ Or enter your skills manually (comma-separated)")
            manual_skills = [skill.strip().lower() for skill in resume_skills_input.split(',') if skill.strip()]

        # Combine selected skills
        combined_skills = set()
        if use_resume_skills:
            combined_skills.update(skill.lower() for skill in resume_skills_from_file)
        if manual_skills:
            combined_skills.update(manual_skills)

        if combined_skills:
            combined_skills_str = ', '.join(sorted(combined_skills))

            matched_jobs = match_jobs_to_skills(df, combined_skills_str)

            st.markdown(f"### 🏆 Top 5 Job Matches for Skills: `{combined_skills_str}`")

            if not matched_jobs.empty:
                st.dataframe(
                    matched_jobs[['job_title', 'company', 'job_location', 'job_level']].reset_index(drop=True),
                    use_container_width=True,
                    height=300
                )
            else:
                st.warning("❌ No matching jobs found. Try entering different or broader skills.")
        else:
            st.info("ℹ️ Please upload a resume or enter skills manually to see job matches.")
        if combined_skills:
           st.markdown("### 🔍 Skill Gap Analysis for Top Matched Jobs")
           skill_gap_df = skill_gap_analysis1(df, list(combined_skills), top_n=5)
           st.dataframe(skill_gap_df, use_container_width=True)
    
    elif choice == "analysis2":
             demo.run(load_data())
    elif choice == "Salary Prediction":
             demo.predict_salary(load_data())       
    elif choice == "bigdataset analysis":
       st.title("\U0001F4C8 Smart Job Market Analyzer")
       df = load_job_data_from_csv()

       st.sidebar.title("Analysis Menu")
       options = [
        "Top Matching Jobs", "Skill Demand", "Job Role Insights",
        "Location Trends", "Company Hiring", "Time-Based Trends",
        "Skill Gap Analysis", "Work Type Analysis"
       ]
       choice = st.sidebar.radio("Select an analysis:", options)

       if choice == "Top Matching Jobs":
        display_top_matching_jobs(df)
       elif choice == "Skill Demand":
        skill_demand_analysis(df)
       elif choice == "Job Role Insights":
        job_role_insights(df)
       elif choice == "Location Trends":
        location_based_analysis(df)
       elif choice == "Company Hiring":
        company_demand_analysis(df)
       elif choice == "Time-Based Trends":
        time_based_trend(df)
       elif choice == "Skill Gap Analysis":
        skill_gap_analysis(df)
       elif choice == "Work Type Analysis":
        work_type_analysis(df)

    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            if ad_user == 'user' and ad_password == 'srinithi2005':
                st.success("Welcome admin")
                # Display Data
                cursor.execute('''SELECT*FROM user_data''')
                data = cursor.fetchall()
                st.header("**User's👨‍💻 Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                 'Recommended Course'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)

                ## Pie chart for predicted field recommendations
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()
                st.subheader("📈 **Pie-Chart for Predicted Field Recommendations**")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                ### Pie chart for User's👨‍💻 Experienced Level
                labels = plot_data.User_level.unique()
                values = plot_data.User_level.value_counts()
                st.subheader("📈 ** Pie-Chart for User's👨‍💻 Experienced Level**")
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                st.plotly_chart(fig)


            else:
                st.error("Wrong ID & Password Provided")

run()

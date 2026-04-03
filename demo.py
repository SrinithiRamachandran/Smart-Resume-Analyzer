import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# This line must be the very first Streamlit command

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd
import streamlit as st
import numpy as np
# Function to train and predict salary

from xgboost import XGBRegressor
def predict_salary(df):
    # Sample 20,000 rows to speed up training (can be adjusted)
    df_sample = df.sample(n=20000, random_state=42)

    # Function to extract average salary from salary range
    def extract_avg_salary(s):
        try:
            nums = [float(x.strip().replace('k','')) for x in s.lower().replace("lpa", "").replace("–", "-").replace("$", "").split("-")]
            return sum(nums) / len(nums)
        except:
            return None

    # Extract average salary from the salary range
    df_sample['avg_salary'] = df_sample['salary range'].apply(extract_avg_salary)

    # Check if 'avg_salary' is correctly created
    if 'avg_salary' not in df_sample.columns:
        print("Error: 'avg_salary' column not found.")
        return

    # Drop rows where 'avg_salary' is NaN
    df_sample = df_sample.dropna(subset=['avg_salary'])

    # Additional feature engineering (e.g., extracting number of skills)
    df_sample['num_skills'] = df_sample['skills'].str.split(',').apply(len)

    # Select features and target
    X = df_sample[['skills', 'job title', 'experience', 'num_skills']]  # Add 'num_skills' as an additional feature
    y = df_sample['avg_salary']

    # Preprocessing: Vectorize skills using TF-IDF and One-Hot Encode job title and experience
    preprocessor = ColumnTransformer(
        transformers=[
            ('skills', TfidfVectorizer(max_features=100), 'skills'),
            ('job_title', OneHotEncoder(handle_unknown='ignore'), ['job title']),
            ('experience', OneHotEncoder(handle_unknown='ignore'), ['experience']),
            ('num_skills', 'passthrough', ['num_skills'])  # Directly pass through num_skills
        ])

    # Create the XGBoost model inside a pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'xgb__n_estimators': [100, 200, 500],
        'xgb__learning_rate': [0.01, 0.1, 0.3],
        'xgb__max_depth': [3, 5, 7],
        'xgb__subsample': [0.7, 0.8, 1.0],
        'xgb__colsample_bytree': [0.7, 0.8, 1.0]
    }

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Performance metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Output Model Performance
    print(f"Best Model Parameters: {grid_search.best_params_}")
    print(f"R² Score: {r2:.3f} ({round(r2 * 100)}%)")
    print(f"MAE: ₹ {mae:.2f} K")
    print(f"RMSE: ₹ {rmse:.2f} K")

    # Return the trained model and the evaluation metrics
    return best_model, r2, mae, rmse
def salary_vs_skills_analysis(df):
    st.subheader("💰 Salary vs. Skills Analysis")

    if 'salary range' not in df.columns or 'skills' not in df.columns or 'company' not in df.columns:
        st.error("Required columns ('salary range', 'skills', 'company') not found!")
        return

    # Drop rows where salary, skills, or company are missing
    df = df.dropna(subset=['salary range', 'skills', 'company'])

    # Select companies from the sidebar
    sectors = df['company'].unique()
    selected_sectors = st.sidebar.multiselect("Select companies for salary analysis", sectors, default=sectors[:1])
    df = df[df['company'].isin(selected_sectors)]

    # Extract salary ranges and clean them
    salary_range = df['salary range'].str.extract(r'\$(\d{1,3}(?:,\d{3})*)K-\$(\d{1,3}(?:,\d{3})*)K')

    # Clean salary range by removing commas and 'K', convert to integer
    df['min_salary'] = salary_range[0].str.replace(',', '').astype(float) * 1000
    df['max_salary'] = salary_range[1].str.replace(',', '').astype(float) * 1000

    # Calculate the average salary
    df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

    # Extract number of skills
    df['num_skills'] = df['skills'].str.split(',').apply(len)

    # Show the chart for salary vs number of skills
    st.write("### Salary vs Number of Skills")
    st.line_chart(df[['num_skills', 'avg_salary']].groupby('num_skills').mean())

    # Show the DataFrame for debugging if salaries are 0 or incorrect
    if df['avg_salary'].mean() == 0:
        st.write("### Debugging: Check rows with 0 salary")
        st.write(df[['salary range', 'min_salary', 'max_salary', 'avg_salary']])

    st.write(df[['num_skills', 'avg_salary']].head())


def experience_vs_skills(df):
    st.subheader("🧑‍💻 Experience Level vs. Skill Demand")

    if 'experience' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('experience', 'skills') not found!")
        return

    df = df.dropna(subset=['experience', 'skills'])
    sectors = df['company'].unique()
    selected_sectors = st.sidebar.multiselect("Select companies for experience vs skills", sectors, default=sectors[:1])
    df = df[df['company'].isin(selected_sectors)]

    df['experience'] = df['experience'].str.lower()
    experience_levels = df['experience'].unique()

    for level in experience_levels:
        level_df = df[df['experience'] == level]
        skill_count = level_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Top Skills for {level.title()} Level")
        st.bar_chart(skill_count)

def location_skill_gap_analysis(df):
    st.subheader("📍 Location-Based Skill Gap Analysis")

    if 'location' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('location', 'skills') not found!")
        return

    df = df.dropna(subset=['location', 'skills'])
    locations = df['location'].unique()
    selected_locations = st.sidebar.multiselect("Select locations for skill gap analysis", locations, default=locations[:1])
    df = df[df['location'].isin(selected_locations)]

    for location in selected_locations:
        loc_df = df[df['location'] == location]
        skill_count = loc_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Top Skills in {location}")
        st.bar_chart(skill_count)

def industry_skill_analysis(df):
    st.subheader("🏢 Industry-Wise Skill Demand")

    if 'company' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('company', 'skills') not found!")
        return

    df = df.dropna(subset=['company', 'skills'])
    companies = df['company'].unique()
    selected_companies = st.sidebar.multiselect("Select companies for skill analysis", companies, default=companies[:1])
    df = df[df['company'].isin(selected_companies)]

    for company in selected_companies:
        comp_df = df[df['company'] == company]
        skill_count = comp_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Top Skills at {company}")
        st.bar_chart(skill_count)

def proficiency_vs_requirement(df, resume_skills):
    st.subheader("🧑‍🏫 Skill Proficiency vs. Job Requirement")

    if 'skills' not in df.columns:
        st.error("Required column ('skills') not found!")
        return

    # Extract skills from the dataset and resume
    job_skills = set(df['skills'].str.split(',').explode().str.strip())
    resume_skills = set(resume_skills)

    # Find missing skills
    missing_skills = job_skills - resume_skills

    # Limit to top N missing skills
    top_missing_skills = list(missing_skills)[:10]  # Show only top 10

    st.write("### Top 10 Missing Skills (From your Resume):")
    st.write(top_missing_skills)

def company_growth_vs_skills(df):
    st.subheader("📈 Company Growth vs. Skills in Demand")

    if 'company' not in df.columns or 'skills' not in df.columns or 'company size' not in df.columns:
        st.error("Required columns ('company', 'skills', 'company size') not found!")
        return

    df = df.dropna(subset=['company', 'skills', 'company size'])
    companies = df['company'].unique()
    selected_companies = st.sidebar.multiselect("Select companies for growth vs skills", companies, default=companies[:1])
    df = df[df['company'].isin(selected_companies)]

    for company in selected_companies:
        comp_df = df[df['company'] == company]
        skill_count = comp_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Skills at {company} with Company Size {comp_df['company size'].iloc[0]}")
        st.bar_chart(skill_count)

def display_top_matching_jobs(df):
    st.subheader("🎯 Top Matching Jobs")

    if 'job title' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('job title', 'skills') not found!")
        return

    df = df.dropna(subset=['job title', 'skills'])
    jobs = df['job title'].unique()
    selected_jobs = st.sidebar.multiselect("Select job roles to view", jobs, default=jobs[:1])
    df = df[df['job title'].isin(selected_jobs)]

    for job in selected_jobs:
        job_df = df[df['job title'] == job]
        st.write(f"### Skills for {job}")
        skill_count = job_df['skills'].str.split(',').explode().value_counts().head(10)
        st.bar_chart(skill_count)

def skills_trend_analysis(df):
    st.subheader("🔥 Skill Demand Overview")

    if 'skills' not in df.columns:
        st.error("Required column ('skills') not found!")
        return

    df = df.dropna(subset=['skills'])
    all_skills = df['skills'].str.split(',').explode().value_counts().head(20)
    st.write("### Most In-Demand Skills")
    st.bar_chart(all_skills)
def cross_domain_skills(df):
    st.subheader("🔄 Cross-Domain Skill Comparison")

    if 'skills' not in df.columns or 'job title' not in df.columns:
        st.error("Required columns ('skills', 'job title') not found!")
        return

    df = df.dropna(subset=['skills', 'job title'])
    
    # Extract unique job titles and create a selection for the user
    job_titles = df['job title'].unique()
    selected_titles = st.sidebar.multiselect("Select job titles for cross-domain skill comparison", job_titles, default=job_titles[:2])

    # Filter data for selected job titles
    df = df[df['job title'].isin(selected_titles)]

    # Create a skill comparison between the selected domains (job titles)
    skill_sets = {}
    for title in selected_titles:
        title_df = df[df['job title'] == title]
        skills = title_df['skills'].str.split(',').explode().unique()
        skill_sets[title] = set(skills)

    # Show the comparison of skills across the selected job titles
    st.write("### Skills Comparison Across Selected Job Titles")
    for title, skills in skill_sets.items():
        st.write(f"#### {title}")
        st.write(f"Skills: {', '.join(skills)}")

    # Optionally, find skills that are common or unique across the selected job titles
    common_skills = set.intersection(*skill_sets.values()) if len(skill_sets) > 1 else set()
    unique_skills = {title: skill_sets[title] - common_skills for title in skill_sets}

    if common_skills:
        st.write("### Common Skills Across All Selected Job Titles")
        st.write(", ".join(common_skills))

    for title, unique in unique_skills.items():
        if unique:
            st.write(f"### Skills Unique to {title}")
            st.write(", ".join(unique))
def co_occurrence_analysis(df):
    st.subheader("🔍 Skills Co-occurrence Analysis")
    
    if 'skills' not in df.columns:
        st.error("Required column ('skills') not found!")
        return
    
    # Drop rows with missing skills
    df = df.dropna(subset=['skills'])
    
    # Split skills and create a list of lists of skills
    skill_lists = df['skills'].str.split(',')
    
    # Create a list of pairs of skills that co-occur in the same job listing
    skill_pairs = []
    for skills in skill_lists:
        skills = [s.strip().lower() for s in skills]  # Remove spaces and make lowercase
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                skill_pairs.append((skills[i], skills[j]))
    
    # Count the co-occurrences of each skill pair
    skill_pair_counts = Counter(skill_pairs)
    
    # Convert to a DataFrame for easier visualization
    co_occurrence_df = pd.DataFrame(skill_pair_counts.items(), columns=['Skill Pair', 'Co-occurrence Count'])
    co_occurrence_df = co_occurrence_df.sort_values(by='Co-occurrence Count', ascending=False).head(10)

    st.write("### Top 10 Skill Pairs with Most Co-occurrences:")
    st.write(co_occurrence_df)

    # Optionally, visualize the co-occurrence counts with a bar chart
    st.bar_chart(co_occurrence_df.set_index('Skill Pair')['Co-occurrence Count'])

def skills_by_job_title(df):
    st.subheader("💼 Skills Frequency by Job Title")

    if 'job title' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('job title', 'skills') not found!")
        return

    df = df.dropna(subset=['job title', 'skills'])
    
    # Group by job title and aggregate skill frequency
    job_title_groups = df.groupby('job title')['skills'].apply(lambda x: ','.join(x)).reset_index()
    job_title_groups['skills'] = job_title_groups['skills'].str.split(',')
    
    # Create a dictionary to store frequency of skills for each job title
    skill_frequency_by_title = {}
    for _, row in job_title_groups.iterrows():
        job_title = row['job title']
        skills = row['skills']
        skill_counts = pd.Series(skills).value_counts()
        skill_frequency_by_title[job_title] = skill_counts

    # Display the skill frequency by job title
    for job_title, skill_counts in skill_frequency_by_title.items():
        st.write(f"### Skills for {job_title}")
        st.bar_chart(skill_counts)

def job_role_insights(df):
    st.subheader("📌 Job Role Insights")

    if 'job title' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('job title', 'skills') not found!")
        return

    df = df.dropna(subset=['job title', 'skills'])
    job_titles = df['job title'].unique()
    selected_titles = st.sidebar.multiselect("Select job roles for analysis", job_titles, default=job_titles[:1])
    df = df[df['job title'].isin(selected_titles)]

    for title in selected_titles:
        title_df = df[df['job title'] == title]
        skill_count = title_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Top Skills for {title}")
        st.bar_chart(skill_count)

def location_based_analysis(df):
    st.subheader("🌍 Location Trends")

    if 'location' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('location', 'skills') not found!")
        return

    df = df.dropna(subset=['location', 'skills'])
    locations = df['location'].unique()
    selected_locations = st.sidebar.multiselect("Select locations to view", locations, default=locations[:1])
    df = df[df['location'].isin(selected_locations)]

    for location in selected_locations:
        loc_df = df[df['location'] == location]
        skill_count = loc_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Skills in {location}")
        st.bar_chart(skill_count)

def company_demand_analysis(df):
    st.subheader("🏢 Company Hiring Trends")

    if 'company' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('company', 'skills') not found!")
        return

    df = df.dropna(subset=['company', 'skills'])
    companies = df['company'].unique()
    selected_companies = st.sidebar.multiselect("Select companies to analyze", companies, default=companies[:1])
    df = df[df['company'].isin(selected_companies)]

    for company in selected_companies:
        comp_df = df[df['company'] == company]
        skill_count = comp_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Skills in Demand at {company}")
        st.bar_chart(skill_count)

def time_based_trend(df):
    st.subheader("📅 Time-Based Skill Trends")

    if 'job posting date' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('job posting date', 'skills') not found!")
        return

    df = df.dropna(subset=['job posting date', 'skills'])
    df['posting_month'] = pd.to_datetime(df['job posting date']).dt.to_period('M')
    trend = df.explode('skills').groupby('posting_month')['skills'].count()

    st.write("### Skill Mentions Over Time")
    st.line_chart(trend)

def skill_gap_analysis(df):
    st.subheader("🔍 Skill Gap Analysis")

    if 'skills' not in df.columns:
        st.error("Required column ('skills') not found!")
        return

    df = df.dropna(subset=['skills'])
    skill_count = df['skills'].str.split(',').explode().value_counts().head(20)
    st.write("### Most Common Skill Gaps")
    st.bar_chart(skill_count)

def work_type_analysis(df):
    st.subheader("🏠 Work Type Analysis")

    if 'work type' not in df.columns or 'skills' not in df.columns:
        st.error("Required columns ('work type', 'skills') not found!")
        return

    df = df.dropna(subset=['work type', 'skills'])
    work_types = df['work type'].unique()
    selected_types = st.sidebar.multiselect("Select work types", work_types, default=work_types[:1])
    df = df[df['work type'].isin(selected_types)]

    for wtype in selected_types:
        wtype_df = df[df['work type'] == wtype]
        skill_count = wtype_df['skills'].str.split(',').explode().value_counts().head(10)
        st.write(f"### Top Skills in {wtype} Work Type")
        st.bar_chart(skill_count)


def run(df):
    st.write("Columns available in dataset:", df.columns.tolist())
    st.title("Smart Resume Analyzer - Job Market Insights")
    resume_skills_from_file = st.session_state.get('matched_technical_skills', [])
    analysis_option = st.sidebar.radio(
        "Choose an analysis type:",
        [
            "Salary vs. Skills Analysis",
            "Experience Level vs. Skill Demand",
            "Location-Based Skill Gap Analysis",
            "Industry-Wise Skill Demand",
            "Skills Trend Analysis",
            "Skills Frequency by Job Title",
            "Skills Co-occurrence Analysis",
            "Skill Proficiency vs. Job Requirement",
            "Company Growth vs. Skills in Demand",
            "Cross-Domain Skill Comparison",
            
        ]
    )

    if analysis_option == "Salary vs. Skills Analysis":
        salary_vs_skills_analysis(df)
    elif analysis_option == "Experience Level vs. Skill Demand":
        experience_vs_skills(df)
    elif analysis_option == "Location-Based Skill Gap Analysis":
        location_skill_gap_analysis(df)
    elif analysis_option == "Industry-Wise Skill Demand":
        industry_skill_analysis(df)
    elif analysis_option == "Skills Trend Analysis":
        skills_trend_analysis(df)
    elif analysis_option == "Skills Frequency by Job Title":
        skills_by_job_title(df)
    elif analysis_option == "Skills Co-occurrence Analysis":
        co_occurrence_analysis(df)
    elif analysis_option == "Skill Proficiency vs. Job Requirement":
        proficiency_vs_requirement(df, resume_skills_from_file)
    elif analysis_option == "Company Growth vs. Skills in Demand":
        company_growth_vs_skills(df)
    elif analysis_option == "Cross-Domain Skill Comparison":
        cross_domain_skills(df)
    
"""
This is the main.py file that will be used to run the pipeline and query the SQLite database.
"""

import logging
import os
import sqlite3
import webbrowser
from pathlib import Path
import sys
import time
import hashlib
import pandas as pd
import numpy as np
try:
    import pdfplumber
except ImportError:
    # Handle case where pdfplumber is not installed in environment
    pdfplumber = None
    logging.warning("pdfplumber module not found - PDF resume upload may not work")
# Ensure project root is in sys.path so 'from jobhunter import ...' works when run directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Prevent PyTorch/OpenMP thread collisions and Streamlit websocket crashes on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


import streamlit as st
import streamlit.components.v1 as components
from jobhunter import config
from jobhunter.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, RESUME_PATH
from jobhunter.dataTransformer import DataTransformer
from jobhunter.extract import extract
from jobhunter.FileHandler import FileHandler
from jobhunter.load import load
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from jobhunter.SQLiteHandler import (
    delete_resume_in_db,
    fetch_resumes_from_db,
    get_resume_text,
    save_text_to_db,
    update_resume_in_db,
    update_similarity_in_db,
    check_and_upload_to_db,
)

# Add correct import for textAnalysis functions
from jobhunter.textAnalysis import get_openai_api_key, _is_placeholder_key
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(project_root, ".env"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config at the start
st.set_page_config(
    page_title="JobPulse AI", 
    page_icon="🔍",
    layout="wide",  # Use wide layout for better space utilization
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

def extract_search_params_from_resume(resume_name):
    """Extract job titles and location from resume text."""
    text = get_resume_text(resume_name)
    job_titles = []
    location = None

    job_title_keywords = [
        "Data Scientist", "Software Engineer", "ML Engineer", 
        "AI Developer", "Product Manager", "Full Stack Developer", 
        "Business Analyst", "Project Manager", "System Engineer"
    ]

    for keyword in job_title_keywords:
        if keyword.lower() in text.lower():
            job_titles.append(keyword)

    if "remote" in text.lower():
        location = "Remote"

    return job_titles, location

# Initialize the database
def initialize_database():
    """Create the SQLite database and necessary tables if they don't exist."""
    try:
        conn = sqlite3.connect("all_jobs.db")
        cursor = conn.cursor()
        
        # Create jobs_new table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_key TEXT UNIQUE,
                date TEXT,
                resume_similarity REAL DEFAULT 0,
                title TEXT,
                company TEXT,
                company_url TEXT,
                company_type TEXT,
                job_type TEXT,
                job_is_remote TEXT,
                job_apply_link TEXT,
                job_offer_expiration_date TEXT,
                salary_low REAL,
                salary_high REAL,
                salary_currency TEXT,
                salary_period TEXT,
                job_benefits TEXT,
                city TEXT,
                state TEXT,
                country TEXT,
                apply_options TEXT,
                required_skills TEXT,
                required_experience TEXT,
                required_education TEXT,
                description TEXT,
                highlights TEXT,
                embeddings TEXT
            )
        ''')
        
        # Create resumes table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_name TEXT UNIQUE,
                resume_text TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"Error initializing database: {e}")
        sys.exit(1)

# Initialize database at startup
initialize_database()

def filter_dataframe(df: pd.DataFrame, key_prefix: str = "") -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let users filter columns
    
    Args:
        df (pd.DataFrame): Original dataframe
        key_prefix (str): A prefix to ensure unique widget keys
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters", value=True, key=f"{key_prefix}_add_filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into standard format
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on", 
            df.columns, 
            key=f"{key_prefix}_filter_columns"
        )
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            widget_key_base = f"{key_prefix}_filter_{column}"
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{widget_key_base}_cat"
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    key=f"{widget_key_base}_num"
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{widget_key_base}_date"
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"{widget_key_base}_text"
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def open_next_job_urls(filtered_df: pd.DataFrame, last_opened_index: int, num_jobs: int):
    """
    Opens job URLs in the browser

    Args:
        filtered_df (pd.DataFrame): Dataframe containing job apply links
        last_opened_index (int): Index to start from
        num_jobs (int): Number of job URLs to open
    """
    
    # Extract job apply links from the dataframe
    job_apply_links = filtered_df["job_apply_link"].tolist()
    
    # Get a slice of job links to open based on the last opened index
    job_links_to_open = job_apply_links[last_opened_index:last_opened_index + num_jobs]
    
    # Open each job link in a new browser tab
    for job_link in job_links_to_open:
        if job_link and isinstance(job_link, str) and job_link.startswith("http"):
            try:
                webbrowser.open_new_tab(job_link)
                time.sleep(0.5)  # Small delay to prevent browser throttling
            except Exception as e:
                st.error(f"Failed to open URL: {job_link}. Error: {e}")
        else:
            st.warning(f"Invalid URL: {job_link}")
    
    # Display information about the opened URLs
    st.info(f"Opened {len(job_links_to_open)} job URLs in new browser tabs.")

file_handler = FileHandler(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH)


def run_transform():
    DataTransformer(
        raw_path=RAW_DATA_PATH,
        processed_path=PROCESSED_DATA_PATH,
        resume_path=RESUME_PATH,
        data=file_handler.import_job_data_from_dir(dirpath=RAW_DATA_PATH),
    ).transform()


# Initialize session state variables
if "active_resume" not in st.session_state:
    st.session_state.active_resume = None

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

if "data_queried" not in st.session_state:
    st.session_state["data_queried"] = False # Will be set to True if initial load finds data

if "query_result" not in st.session_state:
    st.session_state["query_result"] = pd.DataFrame()

if "filtered_result" not in st.session_state:
    st.session_state["filtered_result"] = pd.DataFrame()

if "last_opened_index" not in st.session_state:
    st.session_state["last_opened_index"] = 0

# Handle file uploader reset by checking for a rerun trigger flag
if st.session_state.get("trigger_rerun_after_upload", False):
    # Clear the flag
    st.session_state["trigger_rerun_after_upload"] = False
    # Don't need to clear uploader here - just cleared the flag so next rerun will be clean

# --- Function to Load Initial Data --- 
def load_initial_data():
    """Queries the database on app start to load existing jobs and set default resume."""
    if not st.session_state.get("initial_data_loaded", False): # Run only once per session
        logger.info("Attempting initial data load...")
        
        # Get paths from config for loading
        processed_data_dir = config.PROCESSED_DATA_PATH
        logger.info(f"Loading JSON files from {processed_data_dir}")
        
        conn = None
        try:
            conn = sqlite3.connect("all_jobs.db")
            cursor = conn.cursor()

            # --- Load Existing Jobs --- 
            # Check if jobs_new table has data first
            count_query = "SELECT COUNT(*) FROM jobs_new"
            count_result = pd.read_sql(count_query, conn).iloc[0, 0]
            
            query = """
                SELECT 
                    id, primary_key, date, 
                    CAST(resume_similarity AS REAL) AS resume_similarity,
                    title, company, company_url, company_type, job_type, 
                    job_is_remote, job_apply_link, job_offer_expiration_date, 
                    salary_low, salary_high, salary_currency, salary_period, 
                    job_benefits, city, state, country, apply_options, 
                    required_skills, required_experience, required_education, 
                    description, highlights
                FROM jobs_new 
                ORDER BY resume_similarity DESC, date DESC 
            """
            
            if count_result > 0:
                st.session_state["query_result"] = pd.read_sql(query, conn)
                st.session_state["data_queried"] = True 
                st.session_state["last_opened_index"] = 0 
                logger.info(f"Successfully loaded {len(st.session_state['query_result'])} existing jobs on startup.")
            else:
                st.session_state["query_result"] = pd.DataFrame() 
                st.session_state["data_queried"] = False 
                logger.info("Jobs table is empty. No initial jobs loaded.")
                
                # If jobs_new is empty but we have job files, try to load them
                # This helps when Docker container has JSON files but hasn't loaded them to the DB yet
                if processed_data_dir.exists():
                    json_files = list(processed_data_dir.glob("*.json"))
                    if json_files:
                        logger.info(f"Found {len(json_files)} JSON files in {processed_data_dir} but database is empty")
                        try:
                            # Use the file handler to load jobs from JSON files
                            from jobhunter.FileHandler import FileHandler
                            file_handler = FileHandler()
                            json_jobs = file_handler.load_json_files(processed_data_dir)
                            if json_jobs:
                                logger.info(f"Found {len(json_jobs)} jobs in JSON files, adding to database")
                                # Import the check_and_upload function
                                from jobhunter.SQLiteHandler import check_and_upload_to_db
                                check_and_upload_to_db(json_jobs)
                                logger.info("Re-querying database after loading JSON files")
                                # Refresh from database
                                st.session_state["query_result"] = pd.read_sql(query, conn)
                                st.session_state["data_queried"] = True 
                        except Exception as json_e:
                            logger.error(f"Error loading JSON files: {json_e}")

            # --- Set Default Resume --- 
            available_resumes = fetch_resumes_from_db() # Assumes fetch_resumes_from_db uses the same connection if needed or opens its own
            if available_resumes:
                # Sort resumes reverse alphabetically (assuming later names are newer)
                available_resumes.sort(reverse=True)
                latest_resume = available_resumes[0]
                st.session_state.active_resume = latest_resume
                logger.info(f"Set default active resume to: {latest_resume}")
                # Trigger similarity calculation for the default resume
                with st.spinner(f"Analyzing default resume '{latest_resume}' for job matching..."): # Add spinner here
                    update_success = update_similarity_in_db(latest_resume)
                    if update_success:
                        logger.info(f"Successfully updated similarities for default resume: {latest_resume}")
                    else:
                         logger.error(f"Failed to update similarities for default resume: {latest_resume}")
                         # Optionally show an error in the UI if needed
                         # st.error("Failed to analyze default resume. Job matching might be incomplete.")
            else:
                 logger.info("No resumes found in database to set as default.")
                 st.session_state.active_resume = None # Ensure it's None if DB is empty
            
        except Exception as query_e:
            st.error(f"An error occurred during initial data load: {query_e}")
            logger.error(f"DB Query/Resume Error during initial load: {query_e}", exc_info=True)
            st.session_state["query_result"] = pd.DataFrame()
            st.session_state["data_queried"] = False
            st.session_state.active_resume = None
        finally:
            if conn:
                conn.close()
        st.session_state["initial_data_loaded"] = True # Mark initial load as attempted

# --- Load initial data --- 
load_initial_data()

# --- UI Layout Starts Here ---

# Print session state for debugging (optional)
# logger.debug(f"Session State on Rerun: {st.session_state}")

# Main title in sidebar
st.sidebar.title("JobPulse AI")
st.sidebar.write("AI-powered job search assistant")

# --- Sidebar: API Keys Status (.env) ---
st.sidebar.markdown("<h3 style='color: #38bdf8; font-size: 1.1rem; margin-top: 10px;'>🔑 System Architecture</h3>", unsafe_allow_html=True)

st.sidebar.success("🟢 Embeddings Engine: Local\n`all-MiniLM-L6-v2 (100% Free)`")
st.sidebar.success("🟢 Job Scraper Engine: Local\n`python-jobspy (100% Free)`")

st.sidebar.caption("ℹ️ Both vector generation and live scraping run 100% locally on your machine.")

# --- Sidebar: Resume Management ---
st.sidebar.header("Resume Management")
st.sidebar.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 10px 0;'>", unsafe_allow_html=True) # Separator

# Fetch available resumes
available_resumes = fetch_resumes_from_db()
available_resumes.sort(reverse=True) # Sort newest first

# Display active resume info
active_resume_placeholder = st.sidebar.empty()
if st.session_state.active_resume:
    active_resume_placeholder.markdown(f'<div class="active-resume-div">Active Resume: <strong>{st.session_state.active_resume}</strong></div>', unsafe_allow_html=True)
else:
    active_resume_placeholder.info("No active resume selected. Please upload or select one.")

# Resume Actions Container
resume_container = st.sidebar.container() # Remove border parameter
with resume_container:
    st.subheader("Manage Resumes")
    
    # --- Upload Section ---
    st.markdown("**Upload New Resume**")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        key="resume_uploader",
        label_visibility="collapsed"
    )
    # Automatic processing logic
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get("last_uploaded_file_name", None):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                try:
                    text = " "
                    if uploaded_file.type == "application/pdf":
                        logging.info(f"Reading PDF file: {uploaded_file.name} using pdfplumber")
                        text = ""
                        with pdfplumber.open(uploaded_file) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                        logging.info(f"Successfully extracted text from PDF: {uploaded_file.name}")
                    else:
                        logging.info(f"Reading TXT file: {uploaded_file.name}")
                        text = uploaded_file.read().decode("utf-8")
                        logging.info(f"Successfully read text from TXT: {uploaded_file.name}")

                    save_text_to_db(uploaded_file.name, text)
                    st.session_state.active_resume = uploaded_file.name # Set newly uploaded as active
                    st.session_state.last_uploaded_file_name = uploaded_file.name # Track processed file

                    st.success(f"✅ Resume '{uploaded_file.name}' uploaded!")

                    # Analyze the newly uploaded resume immediately
                    with st.spinner("Analyzing resume for job matching..."):
                        update_success = update_similarity_in_db(uploaded_file.name)
                        if update_success:
                            st.success("Resume analyzed and ready!")
                        else:
                            st.error("Failed to analyze resume.")

                    # Use a flag to indicate we need to rerun instead
                    st.session_state["trigger_rerun_after_upload"] = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file '{uploaded_file.name}': {e}")
                    logger.error(f"File processing error: {e}", exc_info=True)
                    st.session_state.last_uploaded_file_name = None

    # --- Selection / Deletion Section ---
    st.markdown("**Select or Delete Existing Resume**")
    if not available_resumes:
        st.info("No existing resumes found. Upload one first.")
    else:
        # Find index of current active resume for dropdown default
        try:
            current_index = available_resumes.index(st.session_state.active_resume) if st.session_state.active_resume in available_resumes else 0
        except ValueError:
            current_index = 0 # Default to first if active_resume is somehow invalid

        selected_resume = st.selectbox(
            "Select Active Resume",
            options=available_resumes,
            index=current_index,
            key="resume_select_dropdown", # Use a distinct key
            label_visibility="collapsed"
        )

        # Update active resume if selection changed
        if selected_resume != st.session_state.active_resume:
            st.session_state.active_resume = selected_resume
            # Update the displayed active resume immediately
            active_resume_placeholder.markdown(f'<div class="active-resume-div">Active Resume: <strong>{st.session_state.active_resume}</strong></div>', unsafe_allow_html=True)
            # Trigger analysis
            with st.spinner(f"Analyzing selected resume '{selected_resume}' and updating job similarity scores..."):
                # Update similarity scores for the selected resume
                update_success = update_similarity_in_db(selected_resume)
                
                if update_success:
                    # Re-query the database to get updated scores
                    try:
                        conn = sqlite3.connect("all_jobs.db")
                        query = """
                            SELECT
                                id, primary_key, date,
                                CAST(resume_similarity AS REAL) AS resume_similarity,
                                title, company, company_url, company_type, job_type,
                                job_is_remote, job_apply_link, job_offer_expiration_date,
                                salary_low, salary_high, salary_currency, salary_period,
                                job_benefits, city, state, country, apply_options,
                                required_skills, required_experience, required_education,
                                description, highlights
                            FROM jobs_new
                            ORDER BY resume_similarity DESC, date DESC
                        """
                        st.session_state["query_result"] = pd.read_sql(query, conn)
                        conn.close()
                        st.session_state["data_queried"] = True
                    except Exception as query_e:
                        logger.error(f"Error re-querying database after resume change: {query_e}")
                    
                    st.success(f"Resume '{selected_resume}' is now active and all jobs have been analyzed.")
                else:
                    st.error(f"Failed to analyze resume '{selected_resume}'.")
                    
            st.rerun() # Use experimental_rerun instead of rerun

        # Delete button (only enabled if a resume is selected/available)
        delete_disabled = not bool(selected_resume)
        if st.button("Delete Selected Resume", key="delete_selected", disabled=delete_disabled):
            if selected_resume:
                st.session_state.confirming_delete = selected_resume
                st.rerun() # Use experimental_rerun instead of rerun


# --- Deletion Confirmation Logic ---
# Moved outside the container, appears only when needed
if st.session_state.get("confirming_delete"):
    resume_to_delete = st.session_state.confirming_delete
    st.sidebar.warning(f"⚠️ Are you sure you want to delete '{resume_to_delete}'? This cannot be undone.")
    del_col1, del_col2 = st.sidebar.columns(2)
    if del_col1.button("Yes, Delete Permanently", key="confirm_delete_yes", type="primary"):
        with st.spinner(f"Deleting '{resume_to_delete}'..."):
            delete_resume_in_db(resume_to_delete)
            # Clear active resume if it was deleted
            if st.session_state.active_resume == resume_to_delete:
                st.session_state.active_resume = None
            # Clear confirmation state
            st.session_state.confirming_delete = None
            st.sidebar.success(f"🗑️ Resume '{resume_to_delete}' deleted.")
            time.sleep(1)
            st.rerun()
    if del_col2.button("Cancel", key="confirm_delete_cancel"):
        st.session_state.confirming_delete = None
        st.rerun()

# --- Sidebar: Job Search Parameters ---
st.sidebar.header("Job Search")

# Check for active resume before allowing search
if not st.session_state.active_resume:
    st.sidebar.warning("Please select or upload a resume above first.")
else:
    # Search Parameter Input
    search_container = st.sidebar.container() # Remove border parameter
    with search_container:
        st.subheader("Search Parameters")
        
        job_titles_input = st.text_input(
            "Job Titles",
            placeholder="Enter job titles separated by commas",
            help="e.g., Data Scientist, ML Engineer",
            key="job_titles_input"
        )
        
        country = st.selectbox(
            "Country", options=["IND","US", "UK", "LON", "AU", "DE", "FRA", "ES", "IT"], index=0, key="country_select"
        )
        
        date_posted = st.selectbox(
            "Time Frame", options=["all", "today", "3days", "week", "month"], index=0, key="date_posted_select"
        )
        
        location = st.text_input(
            "Location", placeholder="City, state, or region (e.g., Chicago, Remote)", key="location_input"
        )

        job_titles = [title.strip() for title in job_titles_input.split(",")] if job_titles_input else []
        if job_titles and job_titles[0]:
            st.markdown("##### Selected Positions:")
            # Use st.container with horizontal scroll if many titles
            with st.container():
                # Display pills horizontally
                pill_html = "".join([f'<span class="job-pill">{title}</span>' for title in job_titles])
                st.markdown(f'<div style="line-height: 2.0;">{pill_html}</div>', unsafe_allow_html=True)

        # Search Button
        search_disabled = not (job_titles and job_titles[0])
        pages_to_scrape = st.slider(
            "Pages per site", min_value=1, max_value=3, value=1,
            help="More pages = more results but slower. 1 page ≈ 10-30 seconds. 3 pages ≈ 1-2 minutes.",
            key="pages_slider"
        )
        search_button = st.button(
            "🔍 Find Jobs", type="primary", disabled=search_disabled, key="find_jobs_button",
            use_container_width=True  # Make button full width in this column
        )
        if search_disabled:
            st.caption("ℹ️ Please enter at least one job title to search.")


    # --- Job Search Execution Section ---
    if search_button:
        # Replace sidebar spinner with a proper placeholder approach
        progress_message = st.sidebar.empty()
        progress_message.info("Searching for jobs...")
        
        if st.session_state.openai_api_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
            logging.info("Set OpenAI API key from session state for job search")

        # Temporarily override PAGES config with user's choice
        import jobhunter.config as _cfg
        _cfg.PAGES = pages_to_scrape

        steps = [
            lambda: extract(job_titles, country=country, date_posted=date_posted, location=location),
            run_transform,
            load,
        ]


        progress_container = st.sidebar.container()
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0)

        try:
            progress_text.text("Step 1/3: Searching for jobs...")
            total_jobs = steps[0]()
            progress_bar.progress(1/3)

            progress_text.text("Step 2/3: Processing job data...")
            steps[1]()
            progress_bar.progress(2/3)

            progress_text.text("Step 3/3: Saving to database...")
            steps[2]()
            progress_bar.progress(1.0)

            progress_text.empty()
            time.sleep(0.5)
            progress_container.empty()
            # Clear the initial progress message
            progress_message.empty()

            if total_jobs > 0:
                st.sidebar.success(f"✅ Search complete! Found {total_jobs} jobs.")
                
                # Add a pause with spinner to allow similarity calculations to complete
                similarity_container = st.empty()
                with similarity_container.container():
                    with st.spinner("Calculating resume similarities for your jobs..."):
                        # Calculate how many jobs need similarity calculation
                        conn = sqlite3.connect("all_jobs.db")
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM jobs_new WHERE resume_similarity = 0 OR resume_similarity IS NULL")
                        zero_similarity_count = cursor.fetchone()[0]
                        conn.close()
                        
                        if zero_similarity_count > 0:
                            st.info(f"Calculating similarity scores for {zero_similarity_count} jobs...")
                            
                            # Wait a bit for calculations to complete
                            # The timeout should be proportional to the number of jobs
                            # but with a reasonable maximum wait time
                            wait_time = min(max(5, zero_similarity_count / 10), 30)  # Between 5-30 seconds
                            
                            # Show a countdown
                            countdown_text = st.empty()
                            for i in range(int(wait_time), 0, -1):
                                countdown_text.text(f"Refreshing in {i} seconds...")
                                time.sleep(1)
                            
                            # Final refresh
                            countdown_text.text("Refreshing now...")
                
                # Clear the spinner container
                similarity_container.empty()
                st.sidebar.success(f"Ready to view your {total_jobs} jobs with similarity scores!")
            else:
                st.sidebar.warning("No new jobs found matching your search criteria.")
                st.sidebar.markdown("""
                <div style="background-color: #3a3a3a; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin: 10px 0;">
                    <h4 style="margin-top: 0;">Suggestions</h4>
                    <ul>
                        <li>Try more general job titles</li>
                        <li>Specify a major tech hub location</li>
                        <li>Broaden your search time frame</li>
                        <li>Check spelling</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # --- Trigger results update AFTER pipeline finishes ---
            try:
                conn = sqlite3.connect("all_jobs.db")
                query = """
                    SELECT
                        id, primary_key, date,
                        CAST(resume_similarity AS REAL) AS resume_similarity,
                        title, company, company_url, company_type, job_type,
                        job_is_remote, job_apply_link, job_offer_expiration_date,
                        salary_low, salary_high, salary_currency, salary_period,
                        job_benefits, city, state, country, apply_options,
                        required_skills, required_experience, required_education,
                        description, highlights
                    FROM jobs_new
                    ORDER BY resume_similarity DESC, date DESC
                """
                st.session_state["query_result"] = pd.read_sql(query, conn)
                conn.close()
                st.session_state["data_queried"] = True
                st.session_state["last_opened_index"] = 0
                logger.info(f"Successfully queried and updated results for {len(st.session_state['query_result'])} jobs.")
                # Force rerun to display results immediately after search
                st.rerun()
            except Exception as query_e:
                st.sidebar.error(f"An error occurred while querying the database after search: {query_e}")
                logger.error(f"DB Query Error after search: {query_e}", exc_info=True)
                st.session_state["data_queried"] = False

        except Exception as pipeline_error:
            st.sidebar.error(f"An error occurred during the job search pipeline: {pipeline_error}")
            logger.error(f"Pipeline Error: {pipeline_error}", exc_info=True)
            progress_text.empty()
            progress_container.empty()
            st.session_state["data_queried"] = False

# --- Main Content Area: Job Search Dashboard ---
st.markdown("""
<div class="header-banner">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span class="premium-badge">✨ V2.0 AI Career Architect</span>
            <h1 style="margin: 0.5rem 0 0 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-primary);">JobPulse AI</h1>
            <p style="margin: 0.3rem 0 0 0; color: var(--text-secondary); font-size: 1.15rem;">Advanced semantic matching, offline vector indexing & live opportunity tracking</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add a premium metrics row
if st.session_state.get("data_queried") and not st.session_state.get("query_result", pd.DataFrame()).empty:
    df_current = st.session_state["query_result"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Opportunities</div>
            <div class="metric-val">{len(df_current)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        high_matches = len(df_current[df_current['resume_similarity'] > 0.7]) if 'resume_similarity' in df_current.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Match (&gt;70%)</div>
            <div class="metric-val" style="color: #10b981;">{high_matches}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        avg_score = df_current['resume_similarity'].mean() * 100 if 'resume_similarity' in df_current.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Match Score</div>
            <div class="metric-val" style="color: #38bdf8;">{avg_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        active_res = st.session_state.active_resume if st.session_state.active_resume else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Resume</div>
            <div class="metric-val" style="font-size: 1.2rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{active_res}</div>
        </div>
        """, unsafe_allow_html=True)

# Add a refresh button to manually update similarity scores
refresh_col1, refresh_col2 = st.columns([1, 5])
if refresh_col1.button("🔄 Refresh Scores", help="Click to refresh similarity scores for all jobs"):
    with st.spinner("Refreshing similarity scores..."):
        try:
            # Add countdown timer for 20 seconds
            countdown_container = st.empty()
            progress_bar = st.progress(0)
            
            # Show countdown
            for i in range(20, 0, -1):
                progress_bar.progress((20 - i) / 20)
                countdown_container.info(f"Waiting {i} seconds for similarity calculations to complete...")
                time.sleep(1)
            
            countdown_container.empty()
            progress_bar.empty()
            
            conn = sqlite3.connect("all_jobs.db")
            query = """
                SELECT
                    id, primary_key, date,
                    CAST(resume_similarity AS REAL) AS resume_similarity,
                    title, company, company_url, company_type, job_type,
                    job_is_remote, job_apply_link, job_offer_expiration_date,
                    salary_low, salary_high, salary_currency, salary_period,
                    job_benefits, city, state, country, apply_options,
                    required_skills, required_experience, required_education,
                    description, highlights
                FROM jobs_new
                ORDER BY resume_similarity DESC, date DESC
            """
            st.session_state["query_result"] = pd.read_sql(query, conn)
            conn.close()
            st.session_state["data_queried"] = True
            
            refresh_col2.success("✅ Scores refreshed successfully!")
            time.sleep(1.5)
            st.rerun()
        except Exception as e:
            refresh_col2.error(f"Error refreshing scores: {e}")
            logger.error(f"Error during refresh: {e}", exc_info=True)

# Create multi-tab layout
tab_jobs, tab_analytics, tab_settings = st.tabs(["💼 Matched Opportunities", "📊 Search Analytics", "⚙️ System & .env Status"])

with tab_jobs:
    # Check if data is ready to be displayed
    if st.session_state["data_queried"] and not st.session_state["query_result"].empty:
        temp_filtered_df = filter_dataframe(st.session_state["query_result"], key_prefix="results_filter")
        filtered_df = temp_filtered_df if temp_filtered_df is not None and not temp_filtered_df.empty else pd.DataFrame()

        st.session_state["filtered_result"] = filtered_df

        if not filtered_df.empty:
            st.write(f"Showing **{len(filtered_df)}** jobs matching your criteria")
            
            if 'date' in filtered_df.columns:
                filtered_df['date'] = pd.to_datetime(filtered_df['date']).dt.date
            
            display_columns = [
                'resume_similarity',
                'title',
                'highlights',
                'salary_low',
                'salary_high',
                'date',
                'company',
                'job_apply_link',
                'job_is_remote',
                'description'
            ]
            
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            display_df = filtered_df[display_columns]

            # 🚀 Apply to Top 10 Jobs Button
            if st.button("🚀 Apply to Top 10 Jobs", key="apply_top_10_button"):
                top_10_jobs = filtered_df.head(10)
                apply_links = top_10_jobs["job_apply_link"].tolist()
                applied_count = 0
                with st.spinner("Opening job application pages..."):
                    for link in apply_links:
                        if link and isinstance(link, str) and link.startswith("http"):
                            try:
                                webbrowser.open_new_tab(link)
                                applied_count += 1
                                time.sleep(0.5)
                            except Exception as e:
                                st.warning(f"Could not open: {link}. Error: {e}")
                st.success(f"✅ Opened {applied_count} job application pages in your browser.")

            st.dataframe(display_df, height=700, use_container_width=True, hide_index=True)
        else:
            st.warning("No jobs match the current filter criteria. Adjust filters above.")

    elif st.session_state["data_queried"] and st.session_state["query_result"].empty:
        st.warning("No job results found in the database. Run a new search to populate it.")
    else:
        st.info("Database is currently empty. Upload a resume and run a job search from the sidebar.")
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 16px; margin-top: 20px; max-width: 800px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <h3 style="margin-top: 0; color: #38bdf8; font-weight: 700;">🚀 Getting Started with JobPulse AI</h3>
            <ol style="color: #e2e8f0; line-height: 1.8; font-size: 1.05rem;">
                <li>Upload or select a resume in the <strong>Resume Management</strong> section (Everything runs 100% locally and free!)</li>
                <li>Enter job titles and search criteria in the <strong>Job Search</strong> section</li>
                <li>Click <strong>Find Jobs</strong> to start scraping live opportunities across top job boards</li>
                <li>Results will appear here sorted by semantic similarity to your qualifications</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

with tab_analytics:
    st.subheader("📊 Data Breakdown & Analytics")
    if st.session_state.get("data_queried") and not st.session_state.get("query_result", pd.DataFrame()).empty:
        df_ana = st.session_state["query_result"]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<h4 style='color: #38bdf8;'>Similarity Score Distribution</h4>", unsafe_allow_html=True)
            if 'resume_similarity' in df_ana.columns:
                hist_data = np.histogram(df_ana['resume_similarity'], bins=10, range=(0.0, 1.0))[0]
                chart_df = pd.DataFrame({"Match Count": hist_data}, index=[f"{i*10}-{(i+1)*10}%" for i in range(10)])
                st.bar_chart(chart_df)
        
        with col_b:
            st.markdown("<h4 style='color: #38bdf8;'>Remote vs Onsite Breakdown</h4>", unsafe_allow_html=True)
            if 'job_is_remote' in df_ana.columns:
                remote_counts = df_ana['job_is_remote'].apply(lambda x: "Remote" if str(x).lower() in ["true", "1", "yes"] else "Onsite/Hybrid").value_counts()
                st.bar_chart(remote_counts)

        st.markdown("<h4 style='color: #38bdf8;'>Top Hiring Companies</h4>", unsafe_allow_html=True)
        if 'company' in df_ana.columns:
            top_companies = df_ana['company'].value_counts().head(10)
            st.bar_chart(top_companies)
    else:
        st.info("No analytics available. Run a job search or load data first.")

with tab_settings:
    st.subheader("⚙️ System Configuration & Environment Status")
    st.markdown("""
    <div style='background: rgba(17, 24, 39, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 10px;'>
        <h4>System Engine Architecture</h4>
        <p>Your application is configured for 100% offline and open-source execution:</p>
    """, unsafe_allow_html=True)
    
    st.write("**Embeddings Engine**: `🟢 Local (all-MiniLM-L6-v2) - 100% Free`")
    st.write("**Job Scraper Engine**: `🟢 Local (python-jobspy) - 100% Free`")
    st.write(f"**Database Path**: `{os.path.abspath('all_jobs.db')}`")
    st.write(f"**Processed Data Directory**: `{config.PROCESSED_DATA_PATH}`")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Apply custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-surface: rgba(15, 23, 42, 0.65);
        --border-color: rgba(148, 163, 184, 0.2);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-glow: #38bdf8;
    }

    /* Overall App & Widget Base */
    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b 0%, #0f172a 40%, #020617 100%) !important;
        color: #f8fafc !important;
    }
    body, input, textarea, select, p, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    /* Ensure Streamlit icons retain their icon font */
    [data-testid="stIcon"], .st-emotion-cache-1if7yv9, [class*="Icon"] {
        font-family: "Source Sans Pro", sans-serif !important; /* Fallback to Streamlit default for icons */
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3.5rem;
        padding-right: 3.5rem;
        max-width: 1600px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f19 0%, #0f172a 100%) !important;
        border-right: 1px solid var(--border-color) !important;
        min-width: 280px !important;
        max-width: 320px !important;
        box-shadow: 5px 0 25px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Selectbox dropdown popup visibility */
    div[role="listbox"], div[role="listbox"] * {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }

    /* Native Tabs Override */
    [data-testid="stTabs"] button {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.8rem 1.6rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: linear-gradient(180deg, transparent 0%, rgba(56, 189, 248, 0.12) 100%) !important;
        border-top-left-radius: 10px !important;
        border-top-right-radius: 10px !important;
    }

    /* Native Text Inputs Override - ONLY target actual input elements, not baseweb wrappers */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        padding: 0.7rem 1.2rem !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }

    /* Style the selectbox outer container but NOT inner divs that control height */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    }

    /* Fix selected value text visibility - target value div specifically */
    [data-testid="stSelectbox"] [data-baseweb="select"] [data-id="select-value-container"] div,
    [data-testid="stSelectbox"] [data-baseweb="select"] div[value],
    [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] div,
    [data-testid="stSelectbox"] [aria-selected="true"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        min-height: unset !important;
        height: auto !important;
        overflow: visible !important;
    }

    /* Ensure no inner div gets height:0 or overflow:hidden from our padding rules */
    [data-testid="stSelectbox"] [data-baseweb="select"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Fix for the hidden text: override any height collapsing on value display */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
        height: auto !important;
        min-height: 1.5rem !important;
        overflow: visible !important;
    }

    /* Ensure multiselect pills text is visible */
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span {
        color: #ffffff !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35) !important;
    }

    /* Native Buttons Override */
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.6rem !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.5) !important;
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    }

    /* Native File Uploader Override */
    [data-testid="stFileUploader"] > section {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] > section:hover {
        border-color: #38bdf8 !important;
        background: rgba(15, 23, 42, 0.8) !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.2) !important;
    }

    .premium-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        color: #c084fc;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid rgba(168, 85, 247, 0.4);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.2);
        display: inline-block;
        margin-bottom: 12px;
    }

    .header-banner {
        background: var(--bg-surface);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.2rem 2.8rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        border-top: 2px solid #6366f1;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .metric-card {
        background: var(--bg-surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.15);
    }
    .metric-card:hover::after {
        opacity: 1;
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }
    .metric-val {
        color: var(--text-primary);
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Container Styling */
    .container-with-border {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Styling for job pills */
    .job-pill {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(99, 102, 241, 0.15));
        color: var(--accent-glow);
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-size: 0.85em;
        font-weight: 700;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.1);
    }

    .active-resume-div {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.15));
        border-left: 4px solid #10b981;
        color: #ecfdf5;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 16px 0;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
</style>
""", unsafe_allow_html=True)

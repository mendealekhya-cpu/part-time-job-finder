import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import ollama
except Exception:
    ollama = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Part-Time Work Matching",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

USERS_FILE = "users.csv"
JOBS_FILE = "jobs.csv"
REQUESTS_FILE = "contact_requests.csv"


# ============================================================
# WORK OPTIONS
# ============================================================

WORK_OPTIONS = [
    "Home Tutor",
    "Shop Assistant",
    "Restaurant / Cafe",
    "Delivery",
    "Data Entry",
    "Office Assistant",
    "Social Media",
    "Customer Support",
    "Event Assistant",
    "Technical Work",
    "Internship / Technical",
    "Other"
]


# ============================================================
# CSV COLUMNS
# ============================================================

USER_COLUMNS = [
    "mobile",
    "name",
    "role",
    "city",
    "salary",
    "availability",
    "skills",
    "qualification",
    "experience",
    "created_at"
]

JOB_COLUMNS = [
    "id",
    "title",
    "company",
    "city",
    "work_type",
    "category",
    "salary",
    "hours",
    "days",
    "work_mode",
    "skills",
    "qualification",
    "experience",
    "description",
    "created_by",
    "created_at"
]

REQUEST_COLUMNS = [
    "id",
    "sender_mobile",
    "sender_name",
    "sender_type",
    "receiver_mobile",
    "receiver_name",
    "receiver_type",
    "job_id",
    "job_title",
    "company",
    "message",
    "status",
    "created_at"
]


# ============================================================
# CREATE REQUIRED FILES
# ============================================================

def create_files():

    if not os.path.exists(USERS_FILE):
        pd.DataFrame(
            columns=USER_COLUMNS
        ).to_csv(
            USERS_FILE,
            index=False
        )

    if not os.path.exists(JOBS_FILE):
        pd.DataFrame(
            columns=JOB_COLUMNS
        ).to_csv(
            JOBS_FILE,
            index=False
        )

    if not os.path.exists(REQUESTS_FILE):
        pd.DataFrame(
            columns=REQUEST_COLUMNS
        ).to_csv(
            REQUESTS_FILE,
            index=False
        )


create_files()


# ============================================================
# LOAD USERS
# ============================================================

def load_users():

    try:

        df = pd.read_csv(
            USERS_FILE,
            dtype=str
        )

        for column in USER_COLUMNS:

            if column not in df.columns:
                df[column] = ""

        return df[USER_COLUMNS].fillna("")

    except Exception:

        return pd.DataFrame(
            columns=USER_COLUMNS
        )


# ============================================================
# LOAD JOBS
# ============================================================

def load_jobs():

    try:

        df = pd.read_csv(
            JOBS_FILE,
            dtype=str
        )

        for column in JOB_COLUMNS:

            if column not in df.columns:
                df[column] = ""

        return df[JOB_COLUMNS].fillna("")

    except Exception:

        return pd.DataFrame(
            columns=JOB_COLUMNS
        )


# ============================================================
# LOAD CONTACT REQUESTS
# ============================================================

def load_requests():

    try:

        df = pd.read_csv(
            REQUESTS_FILE,
            dtype=str
        )

        for column in REQUEST_COLUMNS:

            if column not in df.columns:
                df[column] = ""

        return df[REQUEST_COLUMNS].fillna("")

    except Exception:

        return pd.DataFrame(
            columns=REQUEST_COLUMNS
        )


# ============================================================
# SAVE USERS
# ============================================================

def save_users(df):

    for column in USER_COLUMNS:

        if column not in df.columns:
            df[column] = ""

    df = df[USER_COLUMNS]

    df.to_csv(
        USERS_FILE,
        index=False
    )


# ============================================================
# SAVE JOBS
# ============================================================

def save_jobs(df):

    for column in JOB_COLUMNS:

        if column not in df.columns:
            df[column] = ""

    df = df[JOB_COLUMNS]

    df.to_csv(
        JOBS_FILE,
        index=False
    )


# ============================================================
# SAVE REQUESTS
# ============================================================

def save_requests(df):

    for column in REQUEST_COLUMNS:

        if column not in df.columns:
            df[column] = ""

    df = df[REQUEST_COLUMNS]

    df.to_csv(
        REQUESTS_FILE,
        index=False
    )


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "mobile" not in st.session_state:
    st.session_state.mobile = ""

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "role" not in st.session_state:
    st.session_state.role = ""


# ============================================================
# SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    if SentenceTransformer is None:
        return None

    try:

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    except Exception:

        return None


embedding_model = load_embedding_model()


# ============================================================
# CREATE EMBEDDING
# ============================================================

def get_embedding(text):

    if embedding_model is None:
        return None

    try:

        return embedding_model.encode(
            str(text),
            convert_to_numpy=True
        )

    except Exception:

        return None


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    if a is None or b is None:
        return 0.0

    try:

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )

    except Exception:

        return 0.0


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_jobs(
    query,
    jobs_df,
    top_k=5
):

    if jobs_df.empty:
        return jobs_df.copy()

    query_embedding = get_embedding(
        query
    )

    if query_embedding is None:

        return jobs_df.head(
            top_k
        ).copy()

    similarity_scores = []

    for _, job in jobs_df.iterrows():

        job_text = f"""
        Job Title: {job['title']}
        Company: {job['company']}
        City: {job['city']}
        Work Type: {job['work_type']}
        Salary: {job['salary']}
        Skills: {job['skills']}
        Qualification: {job['qualification']}
        Experience: {job['experience']}
        Description: {job['description']}
        """

        job_embedding = get_embedding(
            job_text
        )

        score = cosine_similarity(
            query_embedding,
            job_embedding
        )

        similarity_scores.append(score)

    result = jobs_df.copy()

    result["rag_score"] = similarity_scores

    result = result.sort_values(
        "rag_score",
        ascending=False
    )

    return result.head(top_k)


# ============================================================
# RAG RETRIEVAL FOR WORKERS
# ============================================================

def retrieve_workers(
    query,
    workers_df,
    top_k=10
):

    if workers_df.empty:
        return workers_df.copy()

    query_embedding = get_embedding(
        query
    )

    if query_embedding is None:

        return workers_df.head(
            top_k
        ).copy()

    scores = []

    for _, worker in workers_df.iterrows():

        worker_text = f"""
        Worker Name: {worker['name']}
        City: {worker['city']}
        Expected Salary: {worker['salary']}
        Availability: {worker['availability']}
        Skills: {worker['skills']}
        Qualification: {worker['qualification']}
        Experience: {worker['experience']}
        """

        worker_embedding = get_embedding(
            worker_text
        )

        score = cosine_similarity(
            query_embedding,
            worker_embedding
        )

        scores.append(score)

    result = workers_df.copy()

    result["rag_score"] = scores

    result = result.sort_values(
        "rag_score",
        ascending=False
    )

    return result.head(top_k)


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt):

    if ollama is None:

        return (
            "Ollama is not available."
        )

    try:

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception:

        return (
            "Ollama could not generate a response. "
            "Please make sure Ollama is running."
        )


# ============================================================
# EXTRACT NUMBER FROM SALARY
# ============================================================

def extract_salary(value):

    if pd.isna(value):
        return None

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        str(value)
    )

    if not numbers:
        return None

    try:

        return float(
            numbers[0]
        )

    except Exception:

        return None


# ============================================================
# JOB MATCH SCORE
# ============================================================

def calculate_job_match_score(
    work,
    city,
    salary,
    job
):

    score = 0

    # Work
    if (
        str(work).strip().lower()
        ==
        str(job["work_type"]).strip().lower()
    ):

        score += 40

    # City
    if (
        str(city).strip().lower()
        ==
        str(job["city"]).strip().lower()
    ):

        score += 30

    # Salary
    seeker_salary = extract_salary(
        salary
    )

    job_salary = extract_salary(
        job["salary"]
    )

    if (
        seeker_salary is not None
        and job_salary is not None
    ):

        if job_salary >= seeker_salary:

            score += 20

        elif job_salary >= seeker_salary * 0.8:

            score += 10

    score += 10

    return min(score, 100)


# ============================================================
# WORKER MATCH SCORE FOR EMPLOYER
# ============================================================

def calculate_worker_match_score(
    job,
    worker
):

    score = 0

    # Work
    if (
        str(job["work_type"]).strip().lower()
        in
        str(worker.get("skills", "")).strip().lower()
    ):

        score += 20

    # City
    if (
        str(job["city"]).strip().lower()
        ==
        str(worker["city"]).strip().lower()
    ):

        score += 40

    # Salary
    job_salary = extract_salary(
        job["salary"]
    )

    worker_salary = extract_salary(
        worker["salary"]
    )

    if (
        job_salary is not None
        and worker_salary is not None
    ):

        if job_salary >= worker_salary:

            score += 20

        elif job_salary >= worker_salary * 0.8:

            score += 10

    # Technical skills
    if str(job["skills"]).strip():

        if str(worker["skills"]).strip():

            job_skills = (
                str(job["skills"])
                .lower()
                .split(",")
            )

            worker_skills = (
                str(worker["skills"])
                .lower()
                .split(",")
            )

            common = set(
                skill.strip()
                for skill in job_skills
            ).intersection(
                set(
                    skill.strip()
                    for skill in worker_skills
                )
            )

            if common:
                score += 20

    else:

        score += 20

    return min(score, 100)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):

    if PdfReader is None:
        return ""

    try:

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += (
                    page_text
                    + "\n"
                )

        return text

    except Exception:

        return ""


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title(
        "💼 Part-Time Work Matching Platform"
    )

    st.write(
        "Find part-time work or hire people easily."
    )

    st.divider()

    st.subheader(
        "🔐 Login"
    )

    mobile = st.text_input(
        "📱 Mobile Number",
        placeholder="Enter your 10-digit mobile number"
    )

    if st.button(
        "Continue",
        type="primary",
        use_container_width=True
    ):

        mobile = mobile.strip()

        if not mobile:

            st.error(
                "Please enter your mobile number."
            )

            return

        if (
            not mobile.isdigit()
            or len(mobile) != 10
        ):

            st.error(
                "Please enter a valid 10-digit mobile number."
            )

            return

        st.session_state.logged_in = True
        st.session_state.mobile = mobile
        st.session_state.page = "Home"

        st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.title(
        "🏠 Part-Time Work Matching"
    )

    st.write(
        "What would you like to do?"
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # WORK
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "👨‍🎓 I Want to Work"
        )

        st.write(
            "Find suitable part-time jobs."
        )

        if st.button(
            "🔍 I Want to Work",
            use_container_width=True,
            key="home_work"
        ):

            st.session_state.role = "seeker"
            st.session_state.page = "Find Work"

            st.rerun()

    # --------------------------------------------------------
    # HIRE
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "💼 I Want to Hire"
        )

        st.write(
            "Post jobs and find suitable workers."
        )

        if st.button(
            "📢 I Want to Hire",
            use_container_width=True,
            key="home_hire"
        ):

            st.session_state.role = "employer"
            st.session_state.page = "Post Job"

            st.rerun()


# ============================================================
# WORKER PAGE
# ============================================================

def work_seeker_page():

    st.title(
        "👨‍🎓 I Want to Work"
    )

    st.write(
        "Enter your details to find suitable jobs."
    )

    with st.form(
        "worker_form"
    ):

        name = st.text_input(
            "👤 Your Name",
            placeholder="Enter your full name"
        )

        st.text_input(
            "📱 Mobile Number",
            value=st.session_state.mobile,
            disabled=True
        )

        # ONLY WORK = SELECT BOX
        work = st.selectbox(
            "💼 Work",
            WORK_OPTIONS
        )

        # CITY = NORMAL INPUT
        city = st.text_input(
            "📍 City",
            placeholder="Example: Hyderabad"
        )

        # SALARY = NORMAL INPUT
        salary = st.text_input(
            "💰 Expected Salary",
            placeholder="Example: 10000 per month"
        )

        availability = st.text_input(
            "⏰ Availability",
            placeholder="Example: 5 PM - 9 PM"
        )

        skills = ""
        qualification = ""
        experience = ""

        if work == "Internship / Technical":

            st.subheader(
                "🛠️ Technical Details"
            )

            skills = st.text_input(
                "Technical Skills",
                placeholder="Example: Python, C, Verilog"
            )

            qualification = st.text_input(
                "Qualification",
                placeholder="Example: B.Tech ECE"
            )

            experience = st.text_input(
                "Experience",
                placeholder="Example: Fresher"
            )

        submitted = st.form_submit_button(
            "🔍 Find Matching Jobs",
            use_container_width=True
        )

    if not submitted:
        return

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not name.strip():

        st.error(
            "Please enter your name."
        )

        return

    if not city.strip():

        st.error(
            "Please enter your city."
        )

        return

    if not salary.strip():

        st.error(
            "Please enter your expected salary."
        )

        return

    # --------------------------------------------------------
    # SAVE / UPDATE WORKER
    # --------------------------------------------------------

    users_df = load_users()

    new_user = {
        "mobile": st.session_state.mobile,
        "name": name,
        "role": "seeker",
        "city": city,
        "salary": salary,
        "availability": availability,
        "skills": skills,
        "qualification": qualification,
        "experience": experience,
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    users_df = users_df[
        users_df["mobile"].astype(str)
        !=
        str(st.session_state.mobile)
    ]

    users_df = pd.concat(
        [
            users_df,
            pd.DataFrame([new_user])
        ],
        ignore_index=True
    )

    save_users(users_df)

    # --------------------------------------------------------
    # LOAD JOBS
    # --------------------------------------------------------

    jobs_df = load_jobs()

    if jobs_df.empty:

        st.warning(
            "No jobs have been posted yet."
        )

        return

    # --------------------------------------------------------
    # BASIC FILTER
    # --------------------------------------------------------

    work_matches = jobs_df[
        jobs_df["work_type"]
        .astype(str)
        .str.lower()
        ==
        work.lower()
    ]

    if work_matches.empty:

        matching_jobs = jobs_df.copy()

        st.info(
            "No exact work match found. "
            "Showing available jobs."
        )

    else:

        matching_jobs = work_matches.copy()

    # --------------------------------------------------------
    # MATCH SCORE
    # --------------------------------------------------------

    scores = []

    for _, job in matching_jobs.iterrows():

        score = calculate_job_match_score(
            work,
            city,
            salary,
            job
        )

        scores.append(score)

    matching_jobs["match_score"] = scores

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    query = f"""
    Student or worker looking for part-time work.

    Name: {name}
    Work: {work}
    City: {city}
    Expected Salary: {salary}
    Availability: {availability}
    Skills: {skills}
    Qualification: {qualification}
    Experience: {experience}
    """

    retrieved_jobs = retrieve_jobs(
        query,
        matching_jobs,
        top_k=5
    )

    st.success(
        f"Found {len(retrieved_jobs)} recommended job(s)."
    )

    # --------------------------------------------------------
    # DISPLAY JOBS
    # --------------------------------------------------------

    st.subheader(
        "✨ Recommended Jobs"
    )

    requests_df = load_requests()

    for index, job in retrieved_jobs.iterrows():

        score = calculate_job_match_score(
            work,
            city,
            salary,
            job
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### 💼 {job['title']}"
            )

            st.write(
                f"🏢 **Company:** {job['company']}"
            )

            st.write(
                f"📍 **City:** {job['city']}"
            )

            st.write(
                f"💼 **Work:** {job['work_type']}"
            )

            st.write(
                f"💰 **Salary:** {job['salary']}"
            )

            if str(job["hours"]).strip():

                st.write(
                    f"⏰ **Hours:** {job['hours']}"
                )

            if str(job["days"]).strip():

                st.write(
                    f"📅 **Days:** {job['days']}"
                )

            if str(job["work_mode"]).strip():

                st.write(
                    f"🏠 **Mode:** {job['work_mode']}"
                )

            if str(job["description"]).strip():

                st.write(
                    f"📝 **Description:** "
                    f"{job['description'][:500]}"
                )

            st.progress(
                score / 100
            )

            st.write(
                f"🎯 **Match Score: {score}%**"
            )

            # ------------------------------------------------
            # CONTACT EMPLOYER
            # ------------------------------------------------

            job_id = str(job["id"])

            already_requested = requests_df[
                (
                    requests_df["sender_mobile"].astype(str)
                    ==
                    str(st.session_state.mobile)
                )
                &
                (
                    requests_df["receiver_mobile"].astype(str)
                    ==
                    str(job["created_by"])
                )
                &
                (
                    requests_df["job_id"].astype(str)
                    ==
                    job_id
                )
            ]

            if already_requested.empty:

                if st.button(
                    "📩 Contact Employer",
                    key=f"contact_employer_{job_id}_{index}",
                    use_container_width=True
                ):

                    request_id = (
                        len(requests_df) + 1
                    )

                    new_request = {
                        "id": request_id,
                        "sender_mobile": st.session_state.mobile,
                        "sender_name": name,
                        "sender_type": "Worker",
                        "receiver_mobile": job["created_by"],
                        "receiver_name": job["company"],
                        "receiver_type": "Employer",
                        "job_id": job_id,
                        "job_title": job["title"],
                        "company": job["company"],
                        "message": (
                            f"{name} is interested in "
                            f"your job: {job['title']}."
                        ),
                        "status": "Pending",
                        "created_at": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }

                    requests_df = pd.concat(
                        [
                            requests_df,
                            pd.DataFrame([new_request])
                        ],
                        ignore_index=True
                    )

                    save_requests(
                        requests_df
                    )

                    st.success(
                        "📩 Contact request sent to the employer!"
                    )

                    st.rerun()

            else:

                st.info(
                    "📩 Contact request already sent."
                )


# ============================================================
# EMPLOYER POST JOB PAGE
# ============================================================

def employer_page():

    st.title(
        "💼 I Want to Hire"
    )

    st.write(
        "Post a job and find suitable workers."
    )

    with st.form(
        "employer_form"
    ):

        job_title = st.text_input(
            "📌 Job Title",
            placeholder="Example: Part-Time Tutor"
        )

        company = st.text_input(
            "🏢 Company / Organization",
            placeholder="Example: ABC Academy"
        )

        # ONLY WORK = SELECT BOX
        work = st.selectbox(
            "💼 Work",
            WORK_OPTIONS
        )

        # CITY = NORMAL INPUT
        city = st.text_input(
            "📍 City",
            placeholder="Example: Hyderabad"
        )

        # SALARY = NORMAL INPUT
        salary = st.text_input(
            "💰 Salary",
            placeholder="Example: 10000 per month"
        )

        hours = st.text_input(
            "⏰ Working Hours",
            placeholder="Example: 5 PM - 9 PM"
        )

        days = st.text_input(
            "📅 Working Days",
            placeholder="Example: Monday to Friday"
        )

        work_mode = st.selectbox(
            "🏠 Work Mode",
            [
                "On-site",
                "Remote",
                "Hybrid"
            ]
        )

        description = st.text_area(
            "📝 Job Description",
            placeholder="Describe the work..."
        )

        skills = ""
        qualification = ""
        experience = ""

        if work == "Internship / Technical":

            st.subheader(
                "🛠️ Technical Requirements"
            )

            skills = st.text_input(
                "Required Technical Skills",
                placeholder="Example: Python, Verilog, C"
            )

            qualification = st.text_input(
                "Required Qualification",
                placeholder="Example: B.Tech ECE"
            )

            experience = st.text_input(
                "Required Experience",
                placeholder="Example: Fresher"
            )

        st.subheader(
            "📄 Optional Job PDF"
        )

        uploaded_pdf = st.file_uploader(
            "Upload Job / Internship PDF",
            type=["pdf"]
        )

        submitted = st.form_submit_button(
            "📢 Post Job",
            use_container_width=True
        )

    if not submitted:
        return

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not job_title.strip():

        st.error(
            "Please enter the job title."
        )

        return

    if not company.strip():

        st.error(
            "Please enter the company name."
        )

        return

    if not city.strip():

        st.error(
            "Please enter the city."
        )

        return

    if not salary.strip():

        st.error(
            "Please enter the salary."
        )

        return

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_text = ""

    if uploaded_pdf is not None:

        pdf_text = extract_pdf_text(
            uploaded_pdf
        )

    final_description = description

    if pdf_text:

        final_description += (
            "\n\nPDF Information:\n"
            + pdf_text[:5000]
        )

    # --------------------------------------------------------
    # JOB ID
    # --------------------------------------------------------

    jobs_df = load_jobs()

    if jobs_df.empty:

        new_id = 1

    else:

        ids = pd.to_numeric(
            jobs_df["id"],
            errors="coerce"
        ).dropna()

        if ids.empty:

            new_id = 1

        else:

            new_id = int(
                ids.max()
            ) + 1

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    new_job = {
        "id": new_id,
        "title": job_title,
        "company": company,
        "city": city,
        "work_type": work,
        "category": (
            "Technical"
            if work == "Internship / Technical"
            else "General"
        ),
        "salary": salary,
        "hours": hours,
        "days": days,
        "work_mode": work_mode,
        "skills": skills,
        "qualification": qualification,
        "experience": experience,
        "description": final_description,
        "created_by": st.session_state.mobile,
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    jobs_df = pd.concat(
        [
            jobs_df,
            pd.DataFrame([new_job])
        ],
        ignore_index=True
    )

    save_jobs(
        jobs_df
    )

    st.success(
        "✅ Job posted successfully!"
    )

    st.balloons()


# ============================================================
# FIND WORKERS PAGE
# ============================================================

def find_workers_page():

    st.title(
        "👥 Find Suitable Workers"
    )

    st.write(
        "Select one of your posted jobs to find matching workers."
    )

    jobs_df = load_jobs()

    if jobs_df.empty:

        st.info(
            "You have not posted any jobs yet."
        )

        return

    my_jobs = jobs_df[
        jobs_df["created_by"].astype(str)
        ==
        str(st.session_state.mobile)
    ]

    if my_jobs.empty:

        st.info(
            "You have not posted any jobs yet."
        )

        return

    job_options = {}

    for _, job in my_jobs.iterrows():

        label = (
            f"{job['title']} - "
            f"{job['company']} - "
            f"{job['city']}"
        )

        job_options[label] = job["id"]

    selected_label = st.selectbox(
        "💼 Select Job",
        list(job_options.keys())
    )

    selected_job_id = str(
        job_options[selected_label]
    )

    selected_rows = my_jobs[
        my_jobs["id"].astype(str)
        ==
        selected_job_id
    ]

    if selected_rows.empty:
        return

    job = selected_rows.iloc[0]

    users_df = load_users()

    workers = users_df[
        users_df["role"].astype(str)
        ==
        "seeker"
    ].copy()

    if workers.empty:

        st.info(
            "No workers have created profiles yet."
        )

        return

    # --------------------------------------------------------
    # BASIC CITY FILTER
    # --------------------------------------------------------

    city_workers = workers[
        workers["city"]
        .astype(str)
        .str.lower()
        ==
        str(job["city"]).lower()
    ]

    if city_workers.empty:

        matching_workers = workers

        st.info(
            "No exact city match found. "
            "Showing other available workers."
        )

    else:

        matching_workers = city_workers

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    worker_query = f"""
    Employer looking for a worker.

    Job: {job['title']}
    Work Type: {job['work_type']}
    City: {job['city']}
    Salary: {job['salary']}
    Skills: {job['skills']}
    Qualification: {job['qualification']}
    Experience: {job['experience']}
    Description: {job['description']}
    """

    retrieved_workers = retrieve_workers(
        worker_query,
        matching_workers,
        top_k=10
    )

    st.subheader(
        "✨ Matching Workers"
    )

    requests_df = load_requests()

    for index, worker in retrieved_workers.iterrows():

        score = calculate_worker_match_score(
            job,
            worker
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### 👤 {worker['name']}"
            )

            st.write(
                f"📍 **City:** {worker['city']}"
            )

            st.write(
                f"💼 **Interested Work:** "
                f"{job['work_type']}"
            )

            st.write(
                f"💰 **Expected Salary:** "
                f"{worker['salary']}"
            )

            if str(worker["availability"]).strip():

                st.write(
                    f"⏰ **Availability:** "
                    f"{worker['availability']}"
                )

            if str(worker["skills"]).strip():

                st.write(
                    f"🛠️ **Skills:** "
                    f"{worker['skills']}"
                )

            if str(worker["qualification"]).strip():

                st.write(
                    f"🎓 **Qualification:** "
                    f"{worker['qualification']}"
                )

            if str(worker["experience"]).strip():

                st.write(
                    f"📚 **Experience:** "
                    f"{worker['experience']}"
                )

            st.progress(
                score / 100
            )

            st.write(
                f"🎯 **Match Score: {score}%**"
            )

            # ------------------------------------------------
            # CONTACT WORKER
            # ------------------------------------------------

            already_requested = requests_df[
                (
                    requests_df["sender_mobile"].astype(str)
                    ==
                    str(st.session_state.mobile)
                )
                &
                (
                    requests_df["receiver_mobile"].astype(str)
                    ==
                    str(worker["mobile"])
                )
                &
                (
                    requests_df["job_id"].astype(str)
                    ==
                    selected_job_id
                )
            ]

            if already_requested.empty:

                if st.button(
                    "📩 Contact Worker",
                    key=f"contact_worker_{worker['mobile']}_{index}",
                    use_container_width=True
                ):

                    request_id = (
                        len(requests_df) + 1
                    )

                    new_request = {
                        "id": request_id,
                        "sender_mobile": st.session_state.mobile,
                        "sender_name": job["company"],
                        "sender_type": "Employer",
                        "receiver_mobile": worker["mobile"],
                        "receiver_name": worker["name"],
                        "receiver_type": "Worker",
                        "job_id": selected_job_id,
                        "job_title": job["title"],
                        "company": job["company"],
                        "message": (
                            f"{job['company']} is interested "
                            f"in contacting you about "
                            f"{job['title']}."
                        ),
                        "status": "Pending",
                        "created_at": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }

                    requests_df = pd.concat(
                        [
                            requests_df,
                            pd.DataFrame([new_request])
                        ],
                        ignore_index=True
                    )

                    save_requests(
                        requests_df
                    )

                    st.success(
                        "📩 Contact request sent to the worker!"
                    )

                    st.rerun()

            else:

                st.info(
                    "📩 Contact request already sent."
                )


# ============================================================
# MY JOBS PAGE
# ============================================================

def my_jobs_page():

    st.title(
        "📋 My Posted Jobs"
    )

    jobs_df = load_jobs()

    if jobs_df.empty:

        st.info(
            "No jobs have been posted yet."
        )

        return

    my_jobs = jobs_df[
        jobs_df["created_by"].astype(str)
        ==
        str(st.session_state.mobile)
    ]

    if my_jobs.empty:

        st.info(
            "You have not posted any jobs yet."
        )

        return

    for _, job in my_jobs.iterrows():

        with st.container(
            border=True
        ):

            st.subheader(
                f"💼 {job['title']}"
            )

            st.write(
                f"🏢 **Company:** {job['company']}"
            )

            st.write(
                f"💼 **Work:** {job['work_type']}"
            )

            st.write(
                f"📍 **City:** {job['city']}"
            )

            st.write(
                f"💰 **Salary:** {job['salary']}"
            )

            st.write(
                f"📝 **Description:** "
                f"{job['description'][:500]}"
            )


# ============================================================
# CONTACT REQUESTS PAGE
# ============================================================

def contact_requests_page():

    st.title(
        "📩 Contact Requests"
    )

    requests_df = load_requests()

    if requests_df.empty:

        st.info(
            "You don't have any contact requests yet."
        )

        return

    # --------------------------------------------------------
    # RECEIVED REQUESTS
    # --------------------------------------------------------

    received = requests_df[
        requests_df["receiver_mobile"].astype(str)
        ==
        str(st.session_state.mobile)
    ].copy()

    # --------------------------------------------------------
    # SENT REQUESTS
    # --------------------------------------------------------

    sent = requests_df[
        requests_df["sender_mobile"].astype(str)
        ==
        str(st.session_state.mobile)
    ].copy()

    # --------------------------------------------------------
    # RECEIVED
    # --------------------------------------------------------

    st.subheader(
        "📥 Received Requests"
    )

    if received.empty:

        st.info(
            "No received requests."
        )

    else:

        for index, request in received.iterrows():

            with st.container(
                border=True
            ):

                st.write(
                    f"👤 **From:** "
                    f"{request['sender_name']}"
                )

                st.write(
                    f"💼 **Job:** "
                    f"{request['job_title']}"
                )

                if str(request["company"]).strip():

                    st.write(
                        f"🏢 **Company:** "
                        f"{request['company']}"
                    )

                st.write(
                    f"📩 **Message:** "
                    f"{request['message']}"
                )

                st.write(
                    f"📌 **Status:** "
                    f"{request['status']}"
                )

                st.caption(
                    f"Request received: "
                    f"{request['created_at']}"
                )

                # ACCEPT
                if request["status"] == "Pending":

                    if st.button(
                        "✅ Accept Request",
                        key=f"accept_{request['id']}_{index}",
                        use_container_width=True
                    ):

                        requests_df.loc[
                            requests_df["id"].astype(str)
                            ==
                            str(request["id"]),
                            "status"
                        ] = "Accepted"

                        save_requests(
                            requests_df
                        )

                        st.success(
                            "Request accepted!"
                        )

                        st.rerun()

                    if st.button(
                        "❌ Decline Request",
                        key=f"decline_{request['id']}_{index}",
                        use_container_width=True
                    ):

                        requests_df.loc[
                            requests_df["id"].astype(str)
                            ==
                            str(request["id"]),
                            "status"
                        ] = "Declined"

                        save_requests(
                            requests_df
                        )

                        st.info(
                            "Request declined."
                        )

                        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SENT
    # --------------------------------------------------------

    st.subheader(
        "📤 Sent Requests"
    )

    if sent.empty:

        st.info(
            "No requests sent yet."
        )

    else:

        for index, request in sent.iterrows():

            with st.container(
                border=True
            ):

                st.write(
                    f"👤 **To:** "
                    f"{request['receiver_name']}"
                )

                st.write(
                    f"💼 **Job:** "
                    f"{request['job_title']}"
                )

                st.write(
                    f"📌 **Status:** "
                    f"{request['status']}"
                )

                st.caption(
                    f"Sent: "
                    f"{request['created_at']}"
                )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    with st.sidebar:

        st.title(
            "💼 Menu"
        )

        st.write(
            f"📱 Logged in"
        )

        st.divider()

        # HOME
        if st.button(
            "🏠 Home",
            use_container_width=True,
            key="menu_home"
        ):

            st.session_state.page = "Home"

            st.rerun()

        # FIND WORK
        if st.button(
            "🔍 Find Work",
            use_container_width=True,
            key="menu_find_work"
        ):

            st.session_state.role = "seeker"
            st.session_state.page = "Find Work"

            st.rerun()

        # POST JOB
        if st.button(
            "📢 Post Job",
            use_container_width=True,
            key="menu_post_job"
        ):

            st.session_state.role = "employer"
            st.session_state.page = "Post Job"

            st.rerun()

        # FIND WORKERS
        if st.button(
            "👥 Find Workers",
            use_container_width=True,
            key="menu_find_workers"
        ):

            st.session_state.role = "employer"
            st.session_state.page = "Find Workers"

            st.rerun()

        # MY JOBS
        if st.button(
            "📋 My Jobs",
            use_container_width=True,
            key="menu_my_jobs"
        ):

            st.session_state.role = "employer"
            st.session_state.page = "My Jobs"

            st.rerun()

        # CONTACT REQUESTS
        if st.button(
            "📩 Contact Requests",
            use_container_width=True,
            key="menu_requests"
        ):

            st.session_state.page = "Requests"

            st.rerun()

        st.divider()

        # LOGOUT
        if st.button(
            "🚪 Logout",
            use_container_width=True,
            key="menu_logout"
        ):

            st.session_state.logged_in = False
            st.session_state.mobile = ""
            st.session_state.role = ""
            st.session_state.page = "Home"

            st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    sidebar()

    if st.session_state.page == "Home":

        home_page()

    elif st.session_state.page == "Find Work":

        work_seeker_page()

    elif st.session_state.page == "Post Job":

        employer_page()

    elif st.session_state.page == "Find Workers":

        find_workers_page()

    elif st.session_state.page == "My Jobs":

        my_jobs_page()

    elif st.session_state.page == "Requests":

        contact_requests_page()

    else:

        home_page()
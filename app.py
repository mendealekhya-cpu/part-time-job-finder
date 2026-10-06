import os
import re
import pandas as pd
import streamlit as st
import chromadb
import ollama

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Part-Time Connect",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .job-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .match-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f7ff;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "mobile" not in st.session_state:
    st.session_state.mobile = ""

if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False

if "user_type" not in st.session_state:
    st.session_state.user_type = ""

if "jobs" not in st.session_state:
    st.session_state.jobs = None


# ============================================================
# LOAD JOB DATA
# ============================================================

def load_jobs():
    if os.path.exists("jobs.csv"):
        return pd.read_csv("jobs.csv")

    columns = [
        "id",
        "title",
        "company",
        "location",
        "work_type",
        "category",
        "salary",
        "hours",
        "days",
        "work_mode",
        "skills",
        "description"
    ]

    return pd.DataFrame(columns=columns)


jobs = load_jobs()


# ============================================================
# SAVE JOB DATA
# ============================================================

def save_jobs(data):
    data.to_csv("jobs.csv", index=False)


# ============================================================
# VALIDATE MOBILE NUMBER
# ============================================================

def valid_mobile(number):
    return bool(re.fullmatch(r"[0-9]{10}", number))


# ============================================================
# SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CHROMA DATABASE
# ============================================================

@st.cache_resource
def get_chroma_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="part_time_jobs"
    )

    return collection


# ============================================================
# INDEX JOBS INTO RAG DATABASE
# ============================================================

def index_jobs():

    if jobs.empty:
        return

    model = load_embedding_model()
    collection = get_chroma_collection()

    documents = []
    ids = []
    metadatas = []

    for _, job in jobs.iterrows():

        text = f"""
        Job title: {job['title']}
        Company: {job['company']}
        Location: {job['location']}
        Work type: {job['work_type']}
        Category: {job['category']}
        Salary: {job['salary']}
        Working hours: {job['hours']}
        Working days: {job['days']}
        Work mode: {job['work_mode']}
        Skills: {job['skills']}
        Description: {job['description']}
        """

        documents.append(text)
        ids.append(str(job["id"]))

        metadatas.append(
            {
                "title": str(job["title"]),
                "company": str(job["company"]),
                "location": str(job["location"]),
                "work_type": str(job["work_type"]),
                "category": str(job["category"]),
                "salary": str(job["salary"]),
                "hours": str(job["hours"]),
                "days": str(job["days"]),
                "work_mode": str(job["work_mode"])
            }
        )

    embeddings = model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


# ============================================================
# RAG SEARCH
# ============================================================

def rag_search(query, number_of_results=5):

    if jobs.empty:
        return []

    model = load_embedding_model()
    collection = get_chroma_collection()

    query_embedding = model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=number_of_results
    )

    return results.get("documents", [[]])[0]


# ============================================================
# OLLAMA AI
# ============================================================

def ask_ollama(prompt):

    try:

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful part-time job "
                        "matching assistant. "
                        "Only use the job information provided "
                        "in the prompt."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as error:

        return (
            "Ollama could not be reached.\n\n"
            f"Error: {error}"
        )


# ============================================================
# MATCHING FUNCTION
# ============================================================

def calculate_match(
    user_location,
    user_work,
    user_salary,
    user_hours,
    user_days
):

    if jobs.empty:
        return pd.DataFrame()

    result = jobs.copy()

    result["match_score"] = 0

    # Location matching
    result.loc[
        result["location"].str.lower().str.contains(
            user_location.lower(),
            na=False
        ),
        "match_score"
    ] += 30

    # Work type matching
    result.loc[
        result["work_type"].str.lower().str.contains(
            user_work.lower(),
            na=False
        ),
        "match_score"
    ] += 25

    # Hours matching
    result.loc[
        result["hours"].str.lower().str.contains(
            user_hours.lower(),
            na=False
        ),
        "match_score"
    ] += 15

    # Days matching
    result.loc[
        result["days"].str.lower().str.contains(
            user_days.lower(),
            na=False
        ),
        "match_score"
    ] += 15

    # Salary matching
    try:

        expected_salary = float(user_salary)

        result["salary_number"] = (
            result["salary"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

        result.loc[
            result["salary_number"] >= expected_salary,
            "match_score"
        ] += 15

    except ValueError:

        pass

    return result.sort_values(
        by="match_score",
        ascending=False
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception:

        return ""


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        '<div class="main-title">💼 Part-Time Connect</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Find part-time work or hire people nearby"
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📱 Mobile Number Login")

    mobile = st.text_input(
        "Enter your 10-digit mobile number",
        max_chars=10,
        placeholder="9876543210"
    )

    if st.button(
        "Send OTP",
        use_container_width=True
    ):

        if valid_mobile(mobile):

            st.session_state.mobile = mobile
            st.session_state.otp_sent = True

            st.success("OTP sent successfully!")

        else:

            st.error(
                "Please enter a valid 10-digit mobile number."
            )

    if st.session_state.otp_sent:

        st.info(
            "Demo OTP: 123456\n\n"
            "Real SMS OTP can be connected later."
        )

        otp = st.text_input(
            "Enter OTP",
            max_chars=6
        )

        if st.button(
            "Verify OTP",
            use_container_width=True
        ):

            if otp == "123456":

                st.session_state.logged_in = True

                st.success("Login successful!")

                st.rerun()

            else:

                st.error("Incorrect OTP.")


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        '<div class="main-title">'
        "🏠 Part-Time Connect"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Connect people looking for work with people offering work."
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("What do you want to do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 👨‍🎓 I Want to Work")

        st.write(
            "Find suitable part-time jobs near your location."
        )

        if st.button(
            "Find Part-Time Work",
            use_container_width=True
        ):

            st.session_state.user_type = "worker"
            st.rerun()

    with col2:

        st.markdown("### 💼 I Want to Hire")

        st.write(
            "Post a part-time job and find suitable workers."
        )

        if st.button(
            "Post a Job",
            use_container_width=True
        ):

            st.session_state.user_type = "employer"
            st.rerun()

    st.divider()

    st.subheader("✨ What our platform provides")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Available Jobs",
            len(jobs)
        )

    with col2:

        st.metric(
            "AI Assistant",
            "Ollama"
        )

    with col3:

        st.metric(
            "Matching",
            "RAG + AI"
        )


# ============================================================
# WORKER PAGE
# ============================================================

def worker_page():

    st.title("👨‍🎓 Find Part-Time Work")

    st.write(
        "Enter your preferences and we will find suitable work."
    )

    location = st.text_input(
        "📍 Location / Area",
        placeholder="Example: Hyderabad"
    )

    work_type = st.text_input(
        "💼 What type of work do you want?",
        placeholder="Example: shop assistant"
    )

    salary = st.text_input(
        "💰 Expected monthly salary",
        placeholder="Example: 7000"
    )

    hours = st.text_input(
        "⏰ Available working hours",
        placeholder="Example: 5-8 PM"
    )

    days = st.text_input(
        "🗓️ Available days",
        placeholder="Example: Monday to Friday"
    )

    work_category = st.selectbox(
        "Select work category",
        [
            "Normal Part-Time",
            "Internship / Technical"
        ]
    )

    technical_skills = ""

    if work_category == "Internship / Technical":

        technical_skills = st.text_input(
            "🛠️ Technical skills",
            placeholder="Example: Python, HTML, C"
        )

        qualification = st.text_input(
            "🎓 Qualification",
            placeholder="Example: B.Tech ECE"
        )

    else:

        qualification = ""

    if st.button(
        "🔍 Find Matching Jobs",
        use_container_width=True
    ):

        if not location or not work_type:

            st.warning(
                "Please enter your location and work type."
            )

        else:

            matches = calculate_match(
                location,
                work_type,
                salary if salary else "0",
                hours,
                days
            )

            st.session_state["matches"] = matches
            st.session_state["worker_location"] = location
            st.session_state["worker_work"] = work_type
            st.session_state["worker_salary"] = salary
            st.session_state["worker_hours"] = hours
            st.session_state["worker_days"] = days
            st.session_state["worker_category"] = work_category
            st.session_state["technical_skills"] = technical_skills
            st.session_state["qualification"] = qualification

    if "matches" in st.session_state:

        matches = st.session_state["matches"]

        st.divider()

        st.subheader("🎯 Recommended Jobs")

        if matches.empty:

            st.info("No jobs found.")

        else:

            for _, job in matches.head(10).iterrows():

                score = int(job["match_score"])

                st.markdown(
                    f"""
                    <div class="job-card">
                    <h3>💼 {job['title']}</h3>
                    <b>Company:</b> {job['company']}<br>
                    <b>📍 Location:</b> {job['location']}<br>
                    <b>💰 Salary:</b> ₹{job['salary']}<br>
                    <b>⏰ Hours:</b> {job['hours']}<br>
                    <b>🗓️ Days:</b> {job['days']}<br>
                    <b>🏠 Work Mode:</b> {job['work_mode']}<br>
                    <b>🛠️ Skills:</b> {job['skills']}<br>
                    <br>
                    <b>Match Score: {score}%</b>
                    <br>
                    {job['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# EMPLOYER PAGE
# ============================================================

def employer_page():

    st.title("💼 Post a Part-Time Job")

    st.write(
        "Enter the details of the work you want to offer."
    )

    title = st.text_input(
        "Job title",
        placeholder="Example: Shop Assistant"
    )

    company = st.text_input(
        "Company / Employer name"
    )

    location = st.text_input(
        "📍 Work location",
        placeholder="Example: Hyderabad"
    )

    work_type = st.text_input(
        "💼 Type of work",
        placeholder="Example: Shop Assistant"
    )

    category = st.selectbox(
        "Work category",
        [
            "Normal Part-Time",
            "Internship / Technical"
        ]
    )

    salary = st.text_input(
        "💰 Salary per month",
        placeholder="Example: 8000"
    )

    hours = st.text_input(
        "⏰ Working hours",
        placeholder="Example: 5-8 PM"
    )

    days = st.text_input(
        "🗓️ Working days",
        placeholder="Example: Monday to Friday"
    )

    work_mode = st.selectbox(
        "🏠 Work mode",
        [
            "Offline",
            "Remote",
            "Hybrid"
        ]
    )

    skills = ""

    if category == "Internship / Technical":

        skills = st.text_input(
            "🛠️ Required technical skills",
            placeholder="Example: Python, C, HTML"
        )

        qualification = st.text_input(
            "🎓 Required qualification",
            placeholder="Example: B.Tech"
        )

    else:

        qualification = ""

    description = st.text_area(
        "📝 Job description",
        placeholder="Describe the work..."
    )

    if st.button(
        "📢 Post Job",
        use_container_width=True
    ):

        if not title or not location or not work_type:

            st.warning(
                "Please fill Job title, Location and Work type."
            )

        else:

            new_id = 1

            if not jobs.empty:

                new_id = int(
                    jobs["id"].max()
                ) + 1

            new_job = {
                "id": new_id,
                "title": title,
                "company": company,
                "location": location,
                "work_type": work_type,
                "category": category,
                "salary": salary,
                "hours": hours,
                "days": days,
                "work_mode": work_mode,
                "skills": skills,
                "description": description
            }

            updated_jobs = pd.concat(
                [
                    jobs,
                    pd.DataFrame([new_job])
                ],
                ignore_index=True
            )

            save_jobs(updated_jobs)

            st.session_state.jobs = updated_jobs

            st.success(
                "Job posted successfully! 🎉"
            )

            st.info(
                "The new job can now be included in matching."
            )


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

def ai_assistant_page():

    st.title("🤖 AI Job Assistant")

    st.write(
        "Describe what kind of part-time work you are looking for."
    )

    query = st.text_area(
        "Your requirements",
        placeholder=(
            "Example: I am looking for evening shop work "
            "in Hyderabad. I can work Monday to Friday "
            "and expect around ₹7000."
        )
    )

    if st.button(
        "🤖 Find Jobs Using AI",
        use_container_width=True
    ):

        if not query:

            st.warning(
                "Please describe your requirements."
            )

        else:

            with st.spinner(
                "Searching the job knowledge base..."
            ):

                documents = rag_search(
                    query,
                    5
                )

            if not documents:

                st.info(
                    "No matching information found."
                )

            else:

                context = "\n\n".join(
                    documents
                )

                prompt = f"""
Student requirements:

{query}

Relevant jobs retrieved from the database:

{context}

Recommend the most suitable jobs.

For each suitable job explain:

1. Job name
2. Company
3. Location
4. Salary
5. Working hours
6. Why it matches

Only use information from the retrieved jobs.
Do not invent job information.
"""

                with st.spinner(
                    "Ollama is generating recommendations..."
                ):

                    answer = ask_ollama(
                        prompt
                    )

                st.subheader(
                    "✨ AI Recommendation"
                )

                st.write(answer)


# ============================================================
# PDF / KNOWLEDGE PAGE
# ============================================================

def knowledge_page():

    st.title("📄 Job Knowledge Base")

    st.write(
        "Upload a PDF containing job or internship information."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        text = extract_pdf_text(
            uploaded_file
        )

        if text:

            st.success(
                "PDF text extracted successfully!"
            )

            st.text_area(
                "Extracted information",
                text[:5000],
                height=250
            )

        else:

            st.warning(
                "Could not extract text from this PDF."
            )


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.logged_in = False
    st.session_state.mobile = ""
    st.session_state.otp_sent = False
    st.session_state.user_type = ""

    st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    with st.sidebar:

        st.title("💼 Part-Time Connect")

        st.write(
            f"📱 {st.session_state.mobile}"
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "👨‍🎓 Find Work",
                "💼 Hire",
                "🤖 AI Assistant",
                "📄 Knowledge Base"
            ]
        )

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):

            logout()

    # --------------------------------------------------------
    # PAGE ROUTING
    # --------------------------------------------------------

    if page == "🏠 Home":

        home_page()

    elif page == "👨‍🎓 Find Work":

        worker_page()

    elif page == "💼 Hire":

        employer_page()

    elif page == "🤖 AI Assistant":

        # Make sure current jobs are indexed
        index_jobs()

        ai_assistant_page()

    elif page == "📄 Knowledge Base":

        knowledge_page()
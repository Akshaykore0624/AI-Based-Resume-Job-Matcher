# fronted.py
# Cleaned final frontend with dark-card style, auth (backend) support and all existing features.
import streamlit as st
import requests
import json
import os
from typing import Optional, Any

# ---------------------------
# CONFIG
# ---------------------------
BACKEND_URL_DEFAULT = "http://localhost:8001"
REQUEST_TIMEOUT = 30  # seconds
LOGO_PATH = "/mnt/data/3d92379d-5d81-462d-920b-ebfba512e13a.png"

# Developer-provided sample file path (used for quick demo / preview)
SAMPLE_UPLOADED_FILE_PATH = "/mnt/data/7.webp"

# ---------------------------
# HELPERS / RENDERERS
# ---------------------------
def scroll_text(text, height=280):
    safe_text = text if text is not None else ""
    st.markdown(f"""
    <div style="
        background:rgba(0,0,0,0.45);
        border:1px solid rgba(255,255,255,0.04);
        padding:12px;
        height:{height}px;
        overflow-y:auto;
        border-radius:8px;
        white-space:pre-wrap;
        font-size:14px;
        color: #e6eef8;">
        {safe_text}
    </div>
    """, unsafe_allow_html=True)

def bullets_from_json(obj: Any, depth: int = 0):
    pad = "  " * depth
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}**:")
                lines.extend(bullets_from_json(v, depth + 1))
            else:
                lines.append(f"{pad}- **{k}**: {v}")
        return lines
    elif isinstance(obj, list):
        lines = []
        for i, item in enumerate(obj, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- Item {i}:")
                lines.extend(bullets_from_json(item, depth + 1))
            else:
                lines.append(f"{pad}- {item}")
        return lines
    else:
        return [f"{pad}- {obj}"]

def pretty_display(obj: Any, title: Optional[str] = None):
    if title:
        st.markdown(f"### <span style='color:#e6eef8'>{title}</span>", unsafe_allow_html=True)
    if isinstance(obj, (dict, list)):
        for line in bullets_from_json(obj):
            st.markdown(f"<div style='color:#e6eef8'>{line}</div>", unsafe_allow_html=True)
    elif isinstance(obj, str):
        st.markdown(f"<div style='white-space:pre-wrap; color:#e6eef8'>{obj}</div>", unsafe_allow_html=True)
    else:
        st.write(obj)

def render_summary(text):
    if not text:
        st.info("No summary available.")
        return
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>📌 Summary</div>", unsafe_allow_html=True)
    scroll_text(text, height=320)

def render_entities(entities):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🧩 Entities</div>", unsafe_allow_html=True)
    if not entities:
        st.info("No entities found.")
        return
    if isinstance(entities, dict):
        for label, items in entities.items():
            st.markdown(f"**{label}**")
            if isinstance(items, list):
                for it in items:
                    st.markdown(f"- {it}")
            else:
                st.markdown(f"- {items}")
    elif isinstance(entities, list):
        if all(isinstance(x, str) for x in entities):
            for it in entities:
                st.markdown(f"- {it}")
            return
        grouped = {}
        for e in entities:
            if isinstance(e, dict):
                label = e.get("label") or e.get("type") or "Entity"
                value = e.get("text") or e.get("value") or json.dumps(e)
            else:
                label = "Entity"
                value = str(e)
            grouped.setdefault(label, []).append(value)
        for label, items in grouped.items():
            st.markdown(f"**{label}**")
            for it in items:
                st.markdown(f"- {it}")
    else:
        scroll_text(str(entities), height=240)

def render_key_elements(elements):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🔑 Key Elements</div>", unsafe_allow_html=True)
    if not elements:
        st.info("No key elements found.")
        return
    if isinstance(elements, dict):
        for k, v in elements.items():
            st.markdown(f"**{k.replace('_',' ').title()}**")
            if isinstance(v, list):
                for it in v:
                    st.markdown(f"- {it}")
            else:
                st.markdown(f"- {v}")
    elif isinstance(elements, list):
        for it in elements:
            st.markdown(f"- {it}")
    else:
        scroll_text(str(elements), height=260)

def render_answer(answer_text):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>❓ Answer</div>", unsafe_allow_html=True)
    if not answer_text:
        st.info("No answer returned.")
        return
    scroll_text(answer_text, height=260)

def render_comparison(comp):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>📊 Comparison Result</div>", unsafe_allow_html=True)
    if not comp:
        st.info("No comparison result.")
        return
    if isinstance(comp, dict):
        for k, v in comp.items():
            if isinstance(v, (list, dict)):
                st.markdown(f"**{k.replace('_',' ').title()}**")
                if isinstance(v, list):
                    for it in v:
                        st.markdown(f"- {it}")
                else:
                    try:
                        scroll_text(json.dumps(v, indent=2), height=220)
                    except Exception:
                        scroll_text(str(v), height=220)
            else:
                st.markdown(f"- **{k}**: {v}")
    elif isinstance(comp, list):
        for it in comp:
            st.markdown(f"- {it}")
    else:
        scroll_text(str(comp), height=260)

def render_job_suggestions(jobs):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>💼 Job Suggestions</div>", unsafe_allow_html=True)
    if not jobs:
        st.info("No job suggestions yet.")
        return
    if isinstance(jobs, str):
        st.markdown(f"- {jobs}")
        return
    if isinstance(jobs, dict):
        if "raw" in jobs and isinstance(jobs["raw"], str):
            st.error(jobs["raw"])
            return
        if "jobs" in jobs and isinstance(jobs["jobs"], list):
            jobs = jobs["jobs"]
        else:
            flat = []
            for k, v in jobs.items():
                if isinstance(v, (list, dict)):
                    flat.append({k: v})
                else:
                    flat.append(f"{k}: {v}")
            jobs = flat
    if isinstance(jobs, list):
        for i, j in enumerate(jobs, start=1):
            if isinstance(j, str):
                st.markdown(f"- {j}")
                continue
            if not isinstance(j, dict):
                st.markdown(f"- {str(j)}")
                continue
            title = j.get("title") or j.get("job_title") or j.get("role") or j.get("position") or j.get("name") or f"Job {i}"
            company = j.get("company") or j.get("employer") or j.get("organization", "")
            st.markdown(f"**{i}. {title}** {'— ' + company if company else ''}")
            desc = j.get("description") or j.get("summary") or j.get("details")
            if desc:
                if isinstance(desc, list):
                    for d in desc:
                        st.markdown(f"- {d}")
                else:
                    st.markdown(desc)
            skills_field = j.get("skills") or j.get("required_skills") or j.get("skillset")
            if skills_field:
                st.markdown("**Skills Required:**")
                if isinstance(skills_field, list):
                    for s in skills_field:
                        st.markdown(f"- {s}")
                else:
                    st.markdown(f"- {skills_field}")
            # links (Apply / Company) rendered as anchors opening in a new tab
            job_link = j.get("job_link") or j.get("apply_link") or j.get("url")
            company_link = j.get("company_link") or j.get("employer_link")
            link_html_parts = []
            if job_link:
                safe_job = str(job_link)
                link_html_parts.append(f"<a href=\"{safe_job}\" target=\"_blank\" rel=\"noopener noreferrer\">Apply</a>")
            if company_link:
                safe_comp = str(company_link)
                link_html_parts.append(f"<a href=\"{safe_comp}\" target=\"_blank\" rel=\"noopener noreferrer\">Company</a>")
            if link_html_parts:
                st.markdown(" | ".join(link_html_parts), unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"- {str(jobs)}")

def render_parsed_resume(parsed):
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🧾 Parsed Resume</div>", unsafe_allow_html=True)
    if not parsed:
        st.info("No parsed resume data.")
        return
    if isinstance(parsed, str):
        scroll_text(parsed, height=300)
        return
    name = parsed.get("name") or parsed.get("full_name") or parsed.get("candidate") or parsed.get("applicant") or ""
    if name:
        st.markdown(f"### 👤 {name}")
    contact_lines = []
    if parsed.get("email"):
        contact_lines.append(f"**Email:** {parsed.get('email')}")
    if parsed.get("phone"):
        contact_lines.append(f"**Phone:** {parsed.get('phone')}")
    loc = parsed.get("location") or parsed.get("address") or parsed.get("city")
    if loc:
        contact_lines.append(f"**Location:** {loc}")
    if contact_lines:
        st.markdown("#### 📞 Contact Details")
        for cl in contact_lines:
            st.markdown(f"- {cl}")
    if parsed.get("summary"):
        st.markdown("#### 📝 Summary")
        scroll_text(parsed.get("summary"), height=160)
    skills = parsed.get("skills") or parsed.get("skill") or parsed.get("skillset") or []
    if isinstance(skills, dict):
        skills = list(skills.values())
    elif isinstance(skills, str):
        skills = [skills]
    if skills:
        st.markdown("#### 🧠 Skills")
        for s in skills:
            if isinstance(s, dict):
                val = s.get("name") or s.get("skill") or s.get("title") or json.dumps(s)
                st.markdown(f"- {val}")
            else:
                st.markdown(f"- {s}")
    experience = parsed.get("experience") or parsed.get("work_experience") or parsed.get("jobs") or []
    if isinstance(experience, str):
        experience = [experience]
    if isinstance(experience, dict):
        experience = list(experience.values())
    if experience:
        st.markdown("#### 💼 Experience")
        for exp in experience:
            if isinstance(exp, str):
                st.markdown(f"- {exp}")
                continue
            if not isinstance(exp, dict):
                st.markdown(f"- {json.dumps(exp)}")
                continue
            role = exp.get("role") or exp.get("title") or exp.get("position") or ""
            company = exp.get("company") or exp.get("employer") or exp.get("organization") or ""
            start = exp.get("start") or exp.get("from") or ""
            end = exp.get("end") or exp.get("to") or ""
            desc = exp.get("description") or exp.get("details") or exp.get("summary") or ""
            hdr = f"- **{role}**" if role else ""
            if company:
                if hdr:
                    hdr += f" at **{company}**"
                else:
                    hdr = f"- **{company}**"
            if start or end:
                hdr += f" ({start} - {end})"
            if hdr:
                st.markdown(hdr)
            if desc:
                if isinstance(desc, list):
                    for d in desc:
                        st.markdown(f"  - {d}")
                else:
                    st.markdown(f"  - {desc}")
    education = parsed.get("education") or parsed.get("academics") or parsed.get("schools") or []
    if isinstance(education, str):
        education = [education]
    if isinstance(education, dict):
        education = list(education.values())
    if education:
        st.markdown("#### 🎓 Education")
        for edu in education:
            if isinstance(edu, str):
                st.markdown(f"- {edu}")
                continue
            if not isinstance(edu, dict):
                st.markdown(f"- {json.dumps(edu)}")
                continue
            degree = edu.get("degree") or edu.get("qualification") or edu.get("course") or ""
            inst = edu.get("institution") or edu.get("college") or edu.get("school") or ""
            start = edu.get("start") or edu.get("from") or ""
            end = edu.get("end") or edu.get("to") or ""
            line = f"- **{degree}** - {inst}" if degree or inst else ""
            if start or end:
                line += f" ({start} - {end})"
            if line:
                st.markdown(line)
    extras = ["certifications", "projects", "languages", "hobbies", "achievements"]
    for e in extras:
        if parsed.get(e):
            st.markdown(f"#### {e.title()}")
            data = parsed.get(e)
            if isinstance(data, list):
                for it in data:
                    st.markdown(f"- {it}")
            else:
                st.markdown(f"- {data}")

# ---------------------------
# API helpers (with auth support)
# ---------------------------
def get_auth_headers():
    token = st.session_state.get("auth_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def api_post(endpoint: str, data=None, files=None, json_payload=None, timeout=REQUEST_TIMEOUT, include_auth: bool = False):
    base = st.session_state.backend_url.rstrip("/") if st.session_state.get("backend_url") else BACKEND_URL_DEFAULT
    url = base + endpoint
    headers = {}
    if include_auth:
        headers.update(get_auth_headers())
    try:
        if files:
            return requests.post(url, files=files, data=data or {}, headers=headers, timeout=timeout)
        if json_payload:
            return requests.post(url, json=json_payload, headers=headers, timeout=timeout)
        return requests.post(url, data=data or {}, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

def api_get(endpoint: str, params=None, timeout=REQUEST_TIMEOUT, include_auth: bool = False):
    base = st.session_state.backend_url.rstrip("/") if st.session_state.get("backend_url") else BACKEND_URL_DEFAULT
    url = base + endpoint
    headers = {}
    if include_auth:
        headers.update(get_auth_headers())
    try:
        return requests.get(url, params=params or {}, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

# ---------------------------
# Auth UI & Dashboard
# ---------------------------
def do_login():
    st.markdown("<div style='padding:10px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🔐 Login</div>", unsafe_allow_html=True)
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        with st.spinner("Logging in..."):
            resp = api_post("/auth/login", data={"username": username, "password": password}, include_auth=False)
        if resp is None:
            st.error("Network error.")
        elif resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                st.error("Unexpected server response.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            token = data.get("access_token") or data.get("token") or data.get("auth_token")
            user = data.get("user") or data.get("profile") or {"username": username}
            if token:
                st.session_state.auth_token = token
                st.session_state.auth_user = user
                st.success("Logged in.")
                st.experimental_rerun()
            else:
                msg = data.get("message") or data.get("detail") or resp.text
                st.error(f"Login failed: {msg}")
        else:
            try:
                data = resp.json()
                msg = data.get("message") or data.get("detail") or resp.text
            except Exception:
                msg = resp.text
            st.error(f"Login failed: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

def do_register():
    st.markdown("<div style='padding:10px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🆕 Register</div>", unsafe_allow_html=True)
    username = st.text_input("Choose username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    password2 = st.text_input("Confirm password", type="password", key="reg_password2")
    if st.button("Register"):
        if not username or not email or not password:
            st.warning("Fill all fields.")
        elif password != password2:
            st.warning("Passwords do not match.")
        else:
            with st.spinner("Registering..."):
                resp = api_post("/auth/register", data={"username": username, "email": email, "password": password}, include_auth=False)
            if resp is None:
                st.error("Network error.")
            elif resp.status_code in (200, 201):
                st.success("Registered successfully. You can now login.")
            else:
                try:
                    data = resp.json()
                    msg = data.get("message") or data.get("detail") or resp.text
                except Exception:
                    msg = resp.text
                st.error(f"Register failed: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

def do_logout():
    st.session_state.auth_token = None
    st.session_state.auth_user = None
    st.success("Logged out.")
    st.experimental_rerun()

def show_dashboard():
    if not st.session_state.get("auth_token"):
        st.info("Please login to view your dashboard.")
        if st.button("Go to Login"):
            st.session_state.page = "Login"
        return
    st.markdown("<div style='padding:10px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px; color:#e6eef8;'>🧑‍💼 Personal Dashboard</div>", unsafe_allow_html=True)
    with st.spinner("Loading dashboard..."):
        resp = api_get("/auth/user-dashboard", include_auth=True)
        if resp is None or resp.status_code != 200:
            resp = api_get("/user/dashboard", include_auth=True)
    if resp is None:
        st.error("Failed to reach backend for dashboard.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        st.error(f"Could not load dashboard: {msg}")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    try:
        data = resp.json()
    except Exception:
        st.error("Invalid dashboard response.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    profile = data.get("profile") or data.get("user") or st.session_state.get("auth_user") or {}
    resumes = data.get("resumes") or data.get("resume_history") or data.get("parsed_resumes") or []
    saved_jobs = data.get("saved_jobs") or data.get("job_suggestions") or []
    recs = data.get("recommendations") or data.get("recommended") or {}
    st.markdown("### Profile")
    if profile:
        pretty_display(profile)
    else:
        st.markdown("_No profile data available._")
    st.markdown("---")
    st.markdown("### Resume History")
    if resumes:
        for i, r in enumerate(resumes, start=1):
            if isinstance(r, str):
                st.markdown(f"- {r}")
            elif isinstance(r, dict):
                title = r.get("filename") or r.get("name") or r.get("id") or f"Resume {i}"
                st.markdown(f"- **{title}**")
                if r.get("parsed") and st.button(f"View parsed #{i}"):
                    render_parsed_resume(r.get("parsed"))
                if r.get("download_url"):
                    st.markdown(f"[Download]({r.get('download_url')})")
            else:
                st.markdown(f"- {json.dumps(r)}")
    else:
        st.markdown("_No saved resumes._")
    st.markdown("---")
    st.markdown("### Saved Jobs")
    if saved_jobs:
        render_job_suggestions(saved_jobs)
    else:
        st.markdown("_No saved jobs._")
    st.markdown("---")
    st.markdown("### Recommendations")
    if recs:
        pretty_display(recs)
    else:
        st.markdown("_No recommendations available._")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# UI STYLES (Dark card theme) - no outer wrappers
# ---------------------------
st.set_page_config(page_title="CareerVision AI Suite", layout="wide", page_icon="💼")

_CUSTOM_CSS = r"""
<style>
:root{--muted:#9fb0d6; --brand-text:#e6eef8; --card-title:#e6eef8;}
body { background: #07101a; color: #e6eef8; }
.brand-title { font-size:36px; font-weight:800; color:#e6eef8; }
.brand-sub { color: #9fb0d6; margin-top:-6px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:16px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
.small-muted { color:#9fb0d6; font-size:13px; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# SESSION INIT
# ---------------------------
if "backend_url" not in st.session_state:
    st.session_state.backend_url = BACKEND_URL_DEFAULT
if "text" not in st.session_state:
    st.session_state.text = ""
if "text2" not in st.session_state:
    st.session_state.text2 = ""
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = {}
if "job_suggestions" not in st.session_state:
    st.session_state.job_suggestions = []
if "job_suggestions_formatted" not in st.session_state:
    st.session_state.job_suggestions_formatted = None
if "page" not in st.session_state:
    st.session_state.page = "Upload / Preview"
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("<div style='display:flex; gap:10px; align-items:center;'>", unsafe_allow_html=True)
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=48)
    except Exception:
        pass
    st.markdown(f"<div><div class='brand-title'>CareerVision</div><div class='brand-sub small-muted'>AI Resume Suite</div></div>", unsafe_allow_html=True)
    st.markdown("</div>")
    st.markdown("---")
    st.text_input("Backend URL", value=st.session_state.backend_url, key="backend_url")
    st.markdown("---")
    # nav options: simplified — remove separate auth pages and entity/key-elements
    nav_options = ["Upload / Preview", "Summary", "Q&A", "Compare", "Resume Parser", "Strengths & Weaknesses"]
    if st.session_state.auth_token:
        nav_options = ["Dashboard"] + nav_options
    st.session_state.page = st.radio("Go to", nav_options, index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0)
    st.markdown("---")
    if st.session_state.auth_token:
        if st.button("Logout"):
            do_logout()
    st.markdown("<div class='small-muted'>Built for speed & clarity</div>", unsafe_allow_html=True)

# ---------------------------
# HEADER (simple, no hero)
# ---------------------------
hcol1, hcol2 = st.columns([1, 4])
with hcol1:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=72)
    except Exception:
        pass
with hcol2:
    st.markdown(f"<div class='brand-title'>Smart Resume Parser</div>", unsafe_allow_html=True)
    if st.session_state.auth_user:
        uname = st.session_state.auth_user.get("username") if isinstance(st.session_state.auth_user, dict) else str(st.session_state.auth_user)
        st.markdown(f"<div class='brand-sub small-muted'>Signed in as <strong style='color:#e6eef8'>{uname}</strong></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='brand-sub small-muted'>Simplify your recruitment process</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------------------
# ROUTER / PAGES
# ---------------------------
page = st.session_state.page

if page == "Login":
    do_login()
elif page == "Register":
    do_register()
elif page == "Dashboard":
    show_dashboard()

# Upload / Preview
elif page == "Upload / Preview":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>📥 Upload & Preview</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2,3])
    with col_l:
        uploaded_file = st.file_uploader("Upload document (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
        if st.button("Use sample uploaded file (local)"):
            if os.path.exists(SAMPLE_UPLOADED_FILE_PATH):
                with open(SAMPLE_UPLOADED_FILE_PATH, "rb") as f:
                    sample_bytes = f.read()
                st.session_state.filename = os.path.basename(SAMPLE_UPLOADED_FILE_PATH)
                files = {"file": (st.session_state.filename, sample_bytes, "application/octet-stream")}
                with st.spinner("Uploading sample to backend..."):
                    resp = api_post("/upload", files=files, include_auth=False)
                if resp and resp.status_code == 200:
                    st.session_state.text = resp.json().get("text", "")
                    st.success("Sample extracted.")
                else:
                    st.session_state.text = ""
                    st.success(f"Sample selected: {st.session_state.filename}")
            else:
                st.error("Sample file not found on disk.")
        if uploaded_file:
            st.session_state.filename = uploaded_file.name
            st.markdown(f"**File selected:** {uploaded_file.name} — {round(len(uploaded_file.getvalue())/1024,1)} KB")
            if st.button("Extract text from file"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                with st.spinner("Uploading & extracting..."):
                    resp = api_post("/upload", files=files, include_auth=bool(st.session_state.auth_token))
                if resp is None:
                    st.error("Network error.")
                elif resp.status_code == 200:
                    st.session_state.text = resp.json().get("text", "")
                    st.success("Text extracted.")
                else:
                    st.error(f"Error: {resp.status_code} — {resp.text}")
        st.markdown("### ✏️ Paste / edit text")
        manual = st.text_area("Paste document text here", value=st.session_state.text, height=220)
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Save text"):
                st.session_state.text = manual
                st.success("Saved.")
        with col2:
            if st.button("Clear"):
                st.session_state.text = ""
                st.session_state.filename = ""
                st.info("Cleared.")
        with col3:
            if st.button("Format JSON -> Bullets"):
                try:
                    parsed = json.loads(st.session_state.text)
                    pretty_display(parsed, "Detected JSON (converted to bullets)")
                except Exception:
                    st.warning("Text isn't valid JSON.")
    with col_r:
        st.markdown("<div style='color:#9fb0d6; margin-bottom:6px;'>Preview (first 1200 chars)</div>", unsafe_allow_html=True)
        if st.session_state.text:
            scroll_text(st.session_state.text[:1200] + ("..." if len(st.session_state.text) > 1200 else ""), height=320)
        else:
            try:
                if os.path.exists(SAMPLE_UPLOADED_FILE_PATH):
                    st.image(SAMPLE_UPLOADED_FILE_PATH, use_column_width=True)
                else:
                    st.info("Upload or paste text first.")
            except Exception:
                st.info("Upload or paste text first.")
    st.markdown("</div>", unsafe_allow_html=True)

# Summary
elif page == "Summary":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>📌 Summary</div>", unsafe_allow_html=True)
    if not st.session_state.text:
        st.info("Upload & extract text first.")
    else:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                resp = api_post("/analyze/summarize", data={"text": st.session_state.text}, include_auth=bool(st.session_state.auth_token))
            if resp and resp.status_code == 200:
                summary = resp.json().get("summary", "") or resp.json().get("result", "") or resp.json().get("text", "")
                render_summary(summary)
            else:
                st.error("Failed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Entities
# (Entities page removed from nav)

# Strengths & Weaknesses
elif page == "Strengths & Weaknesses":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>💪 Strengths & Weaknesses</div>", unsafe_allow_html=True)
    if not st.session_state.text:
        st.info("Upload & extract text first.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Analyze Strengths"):
                with st.spinner("Analyzing strengths..."):
                    # request strengths mode from backend
                    resp = api_post("/analyze/key_elements", data={"text": st.session_state.text, "mode": "strengths"}, include_auth=bool(st.session_state.auth_token))
                if resp and resp.status_code == 200:
                    try:
                        k = resp.json().get("key_elements", resp.text)
                    except Exception:
                        k = resp.text
                    st.markdown("#### Strengths")
                    scroll_text(k, height=260)
                else:
                    st.error("Failed to analyze strengths.")
        with col_b:
            if st.button("Analyze Weaknesses"):
                with st.spinner("Analyzing weaknesses..."):
                    resp = api_post("/analyze/key_elements", data={"text": st.session_state.text, "mode": "weaknesses"}, include_auth=bool(st.session_state.auth_token))
                if resp and resp.status_code == 200:
                    try:
                        k = resp.json().get("key_elements", resp.text)
                    except Exception:
                        k = resp.text
                    st.markdown("#### Weaknesses")
                    scroll_text(k, height=260)
                else:
                    st.error("Failed to analyze weaknesses.")
    st.markdown("</div>", unsafe_allow_html=True)

# Key Elements
elif page == "Key Elements":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>🔑 Key Elements</div>", unsafe_allow_html=True)
    if not st.session_state.text:
        st.info("Upload & extract text first.")
    else:
        if st.button("Extract Key Elements"):
            with st.spinner("Analyzing..."):
                resp = api_post("/analyze/key_elements", data={"text": st.session_state.text}, include_auth=bool(st.session_state.auth_token))
            if resp and resp.status_code == 200:
                elems = resp.json().get("key_elements", resp.json())
                render_key_elements(elems)
            else:
                st.error("Failed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Q&A
elif page == "Q&A":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>❓ Q&A</div>", unsafe_allow_html=True)
    if not st.session_state.text:
        st.info("Upload document first.")
    else:
        question = st.text_input("Ask a question about the document", key="qa_question")
        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    resp = api_post("/analyze/qa", data={"text": st.session_state.text, "question": question}, include_auth=bool(st.session_state.auth_token))
                if resp and resp.status_code == 200:
                    answer = resp.json().get("answer", "") or resp.json().get("result", "") or resp.text
                    render_answer(answer)
                else:
                    st.error("Failed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Compare
elif page == "Compare":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>📊 Compare Documents</div>", unsafe_allow_html=True)
    sec_file = st.file_uploader("Upload second document", type=["pdf", "docx", "txt"], key="compare2")
    if sec_file:
        if st.button("Extract second document"):
            with st.spinner("Extracting..."):
                files = {"file": (sec_file.name, sec_file.getvalue(), sec_file.type)}
                resp = api_post("/upload", files=files, include_auth=bool(st.session_state.auth_token))
            if resp and resp.status_code == 200:
                st.session_state.text2 = resp.json().get("text", "")
                st.success("Second document extracted.")
            else:
                st.error("Failed.")
    if st.session_state.text and st.session_state.text2:
        start_card("Comparison Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Document 1**")
            scroll_text(st.session_state.text, height=300)
        with c2:
            st.markdown("**Document 2**")
            scroll_text(st.session_state.text2, height=300)
        if st.button("Compare now"):
            with st.spinner("Comparing..."):
                resp = api_post("/analyze/compare", data={"text1": st.session_state.text, "text2": st.session_state.text2}, include_auth=bool(st.session_state.auth_token))
            if resp and resp.status_code == 200:
                comp = resp.json().get("comparison", resp.json())
                render_comparison(comp)
            else:
                st.error("Compare failed.")
        end_card()
    else:
        st.info("Provide both documents.")
    st.markdown("</div>", unsafe_allow_html=True)

# Resume Parser
elif page == "Resume Parser":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700; font-size:16px; margin-bottom:8px;color:#e6eef8;'>🧾 Resume Parser</div>", unsafe_allow_html=True)
    parser_file = st.file_uploader("Upload resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], key="resume_up")
    raw_text = st.text_area("Or paste resume text here", value="", height=200)
    if (parser_file or raw_text.strip()) and st.button("Parse Resume"):
        with st.spinner("Parsing resume..."):
            if parser_file:
                files = {"file": (parser_file.name, parser_file.getvalue(), parser_file.type)}
                resp = api_post("/analyze/parse", files=files, include_auth=bool(st.session_state.auth_token))
            else:
                resp = api_post("/analyze/parse", data={"text": raw_text}, include_auth=bool(st.session_state.auth_token))
        if resp is None:
            st.error("Network error.")
        elif resp.status_code == 200:
            parsed = resp.json().get("parsed", resp.json())
            st.session_state.parsed_resume = parsed if parsed else {}
            render_parsed_resume(st.session_state.parsed_resume)
            st.success("Resume parsed.")
            with st.spinner("Generating job suggestions..."):
                try:
                    parsed_json_str = json.dumps(st.session_state.parsed_resume)
                except Exception:
                    parsed_json_str = str(st.session_state.parsed_resume)
                jr = api_post("/analyze/job_suggestions", data={"parsed_resume": parsed_json_str}, include_auth=bool(st.session_state.auth_token))
            if jr and jr.status_code == 200:
                payload = jr.json()
                jobs = payload.get("jobs", payload)
                st.session_state.job_suggestions = jobs
                # store human-readable formatted suggestions too (if present)
                st.session_state.job_suggestions_formatted = payload.get("jobs_formatted")
                render_job_suggestions(st.session_state.job_suggestions)
                if st.session_state.job_suggestions_formatted:
                    st.markdown("---")
                    st.markdown("<div style='color:#e6eef8'>" + st.session_state.job_suggestions_formatted + "</div>", unsafe_allow_html=True)
            else:
                st.error("Job suggestions failed.")
    if st.session_state.job_suggestions:
        st.markdown("---")
        render_job_suggestions(st.session_state.job_suggestions)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#9fb0d6;'>CareerVision AI Suite — built with ❤️</div>", unsafe_allow_html=True)

# backend.py
"""
CareerVision final backend (Final Release):
- Uses a stable, fast instruction model (mistralai/mistral-7b-instruct) to avoid timeouts.
- Async + timeout safe LLM calls with retries and graceful fallbacks.
- Resume parsing, summarize, entities, key_elements, qa, compare endpoints.
- Job suggestions (LLM) + optional live RapidAPI JSearch integration (non-blocking, timed).
- Robust file text extraction (PDF/DOCX/TXT).
- CORS enabled for local dev.
- Requirements:
    pip install fastapi uvicorn python-dotenv openai pdfplumber python-docx httpx python-multipart
Notes:
 - Put your keys in a .env file:
     OPENROUTER_API_KEY=sk-...
     X_RAPIDAPI_KEY=your_rapidapi_key_here  (optional)
"""

import os
import io
import json
import logging
import asyncio
import concurrent.futures
from typing import Optional, Any, List, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import pdfplumber
import docx
import httpx
import re

# OpenRouter/OpenAI-compatible SDK
from openai import OpenAI

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("careervision-backend")

# -------------------------
# Load environment
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RAPIDAPI_KEY = os.getenv("X_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("🚨 OPENROUTER_API_KEY missing. Add OPENROUTER_API_KEY to your .env file.")

# -------------------------
# App init
# -------------------------
app = FastAPI(title="CareerVision Backend — Final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model selection (stable / lower latency)
# -------------------------
# Changed to a stable instruction model to reduce timeouts.
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct")

# -------------------------
# SDK client (OpenRouter)
# -------------------------
# We use the OpenAI-compatible OpenRouter client. Keep keys out of code.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# -------------------------
# Config defaults (tune via env)
# -------------------------
LLM_CALL_TIMEOUT = float(os.getenv("LLM_CALL_TIMEOUT", "18.0"))   # seconds per LLM attempt
LLM_CALL_RETRIES = int(os.getenv("LLM_CALL_RETRIES", "2"))        # attempts (including first)
RAPIDAPI_TIMEOUT = float(os.getenv("RAPIDAPI_TIMEOUT", "8.0"))    # seconds for RapidAPI
MAX_TEXT_SNIPPET = int(os.getenv("MAX_TEXT_SNIPPET", "25000"))    # trim doc text sent to model

# -------------------------
# Text extraction helpers
# -------------------------
def extract_text_from_pdf(content: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                pages = list(executor.map(lambda p: p.extract_text() or "", pdf.pages))
        text = "\n".join(pages).strip()
        return text or "No text extracted from PDF."
    except Exception:
        log.exception("PDF extraction failed")
        return "Error extracting PDF text."

def extract_text_from_docx(content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs).strip()
    except Exception:
        log.exception("DOCX extraction failed")
        return "Error extracting DOCX text."

def extract_text_from_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8", errors="ignore").strip()
    except Exception:
        return content.decode("latin-1", errors="ignore").strip()

def extract_text(file: UploadFile) -> str:
    try:
        content = file.file.read()
    except Exception:
        return "Error reading uploaded file."

    ctype = (file.content_type or "").lower()
    fname = (file.filename or "").lower()

    if "pdf" in ctype or fname.endswith(".pdf"):
        return extract_text_from_pdf(content)
    if "word" in ctype or fname.endswith((".docx", ".doc")):
        return extract_text_from_docx(content)
    if "text" in ctype or fname.endswith(".txt"):
        return extract_text_from_txt(content)
    return "Unsupported file type"

# -------------------------
# Robust JSON extraction helpers
# -------------------------
def extract_json_object(text: str) -> Optional[Any]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_json_array(text: str) -> Optional[List[Any]]:
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None


# -------------------------
# Lightweight fallback helpers
# -------------------------
def _find_emails(text: str) -> List[str]:
    return re.findall(r"[\w\.-]+@[\w\.-]+", text or "")

def _find_phones(text: str) -> List[str]:
    phones = re.findall(r"\+?\d[\d\s\-()]{6,}\d", text or "")
    # normalize
    clean = []
    for p in phones:
        s = re.sub(r"[^0-9+]", "", p)
        if len(s) >= 7:
            clean.append(s)
    return clean

def _sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _fallback_summary(text: str) -> str:
    sents = _sentences(text)
    if len(sents) >= 2:
        return f"{sents[0]} {sents[1]}\n\nKey highlights:\n- {(' | '.join(re.findall(r'\\b\\w{5,}\\b', text)[:3]) or 'Relevant skills') }"
    if text:
        return text[:400] + ("..." if len(text) > 400 else "")
    return "No summary available."

def _fallback_entities(text: str) -> List[Dict[str, str]]:
    ents: List[Dict[str, str]] = []
    emails = _find_emails(text)
    for e in emails:
        ents.append({"entity": e, "type": "EMAIL"})
    phones = _find_phones(text)
    for p in phones:
        ents.append({"entity": p, "type": "PHONE"})
    # simple proper-noun sequences (capitalized words)
    caps = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", text or "")
    for c in caps[:6]:
        ents.append({"entity": c, "type": "PERSON/ORG"})
    return ents

def _fallback_key_elements(text: str, mode: Optional[str] = None) -> str:
    # Provide separate fallbacks for strengths vs weaknesses
    if not text:
        return "No content to analyze."
    if mode == "strengths":
        # attempt to extract top skills and accomplishments
        sents = _sentences(text)
        strengths = []
        # look for lines/phrases with key skill words
        for kw in ["experience","managed","led","developed","designed","achieved","implemented","improved","reduced","increased"]:
            for sent in sents:
                if kw in sent.lower() and len(strengths) < 6:
                    strengths.append(sent.strip())
        if not strengths:
            # fallback: list common positive items
            strengths = ["Strong technical foundation", "Team collaboration", "Problem-solving skills"]
        out_lines = [f"{i+1}. {s}" for i, s in enumerate(strengths)]
        return "\n".join(out_lines)
    else:
        # weaknesses mode or generic
        s = []
        if len(text) < 300:
            s.append("1) Resume length is short — add more details about accomplishments and metrics.")
        else:
            s.append("1) Quantify achievements where possible (use numbers/metrics).")
        s.append("2) Highlight relevant skills near the top.")
        s.append("3) Use active verbs and concise bullet points.")
        s.append("To improve the resume:\n- Add measurable outcomes\n- Include relevant keywords for target roles\n- Tailor the summary to the job")
        return "\n".join(s)

def _fallback_qa(text: str, question: str) -> str:
    # naive keyword match: return sentences containing keywords from question
    if not text or not question:
        return "No content or question provided."
    words = [w.lower() for w in re.findall(r"\w{4,}", question)]
    sents = _sentences(text)
    matches = [sent for sent in sents if any(w in sent.lower() for w in words)]
    if matches:
        return "\n\n".join(matches[:2])
    # fallback to first 2 sentences
    return " ".join(sents[:2]) if sents else "No answer available."

def _fallback_compare(text1: str, text2: str) -> str:
    # compute simple overlap of keywords
    w1 = set(re.findall(r"\w{4,}", (text1 or "").lower()))
    w2 = set(re.findall(r"\w{4,}", (text2 or "").lower()))
    common = list(w1 & w2)[:8]
    only1 = list(w1 - w2)[:6]
    only2 = list(w2 - w1)[:6]
    out = []
    out.append("Similarities:")
    out.extend([f"- {c}" for c in common])
    out.append("\nDifferences:")
    out.extend([f"- {c}" for c in only1[:3]])
    out.extend([f"- {c}" for c in only2[:3]])
    out.append("\nNotable Points:")
    out.append("- Short naive comparison fallback used (use LLM for detailed results).")
    return "\n".join(out)

def _fallback_parsed(resume_text: str) -> Dict[str, Any]:
    # extract name (first non-empty line), email, phone, and a short summary
    parsed = {}
    lines = [l.strip() for l in (resume_text or "").splitlines() if l.strip()]
    if lines:
        parsed["name"] = lines[0][:80]
        parsed["summary"] = " ".join(lines[1:3]) if len(lines) > 1 else ""
    emails = _find_emails(resume_text)
    if emails:
        parsed["email"] = emails[0]
    phones = _find_phones(resume_text)
    if phones:
        parsed["phone"] = phones[0]
    # naive skills: look for 'skills' section or common keywords
    skills = re.findall(r"\b(Python|Java|SQL|AWS|Docker|Kubernetes|Machine Learning|Data Science|React|Node)\b", resume_text or "", flags=re.I)
    parsed["skills"] = list({s for s in skills})[:12]
    return parsed

# -------------------------
# LLM call helper with timeout + retries + safe fallbacks
# -------------------------
async def query_llama_async(prompt: str, text: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    """
    Call the model via OpenRouter SDK inside a thread to avoid blocking the event loop.
    Enforce timeout and retries. Return readable fallback messages on timeout/error.
    """
    if not (text and text.strip()):
        return "Error: No content to process."

    truncated = text[:MAX_TEXT_SNIPPET]

    loop = asyncio.get_event_loop()

    def sync_call():
        try:
            # Use the chat completions endpoint provided by the OpenRouter-compatible SDK.
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert in document analysis, resume parsing, and career coaching. Answer succinctly and format your output clearly."},
                    {"role": "user", "content": f"{prompt}\n\nDocument Text:\n{truncated}"}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # defensive handling of different response shapes
            if hasattr(completion, "choices") and len(completion.choices) > 0:
                choice = completion.choices[0]
                if hasattr(choice, "message"):
                    msg = choice.message
                    if isinstance(msg, dict):
                        return msg.get("content", "").strip()
                    if hasattr(msg, "content"):
                        return msg.content.strip()
                if hasattr(choice, "text"):
                    return getattr(choice, "text").strip()
            return str(completion)
        except Exception as e:
            log.exception("LLM SDK call failed")
            return f"Error: LLM SDK call failed: {str(e)}"

    last_error = None
    for attempt in range(LLM_CALL_RETRIES):
        try:
            fut = loop.run_in_executor(None, sync_call)
            result = await asyncio.wait_for(fut, timeout=LLM_CALL_TIMEOUT)
            if not result:
                last_error = "Error: Model returned empty response."
                continue
            # return result (may still be JSON or plain text)
            return result
        except asyncio.TimeoutError:
            log.warning("LLM call timed out (attempt %d/%d)", attempt + 1, LLM_CALL_RETRIES)
            last_error = "AI is busy right now (timeout). Try again shortly."
            # small backoff before retrying
            await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as e:
            log.exception("Unexpected error on LLM call (attempt %d/%d)", attempt + 1, LLM_CALL_RETRIES)
            last_error = f"Error: Unexpected error when calling LLM: {str(e)}"
            await asyncio.sleep(0.3)

    # after retries, return a friendly fallback message
    return last_error or "Error: LLM call failed."

# -------------------------
# ROUTES
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, extract_text, file)
    if text == "Unsupported file type":
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return JSONResponse({"filename": file.filename, "text": text, "length": len(text)})

@app.post("/analyze/summarize")
async def summarize(text: str = Form(...)):
    prompt = (
        "Produce a clear human-friendly summary suitable for display. "
        "Write 2-4 short sentences and then 'Key highlights:' with 3 bullets."
    )
    resp = await query_llama_async(prompt, text, max_tokens=600, temperature=0.0)
    # Fallback: if LLM timed out or errored, produce a simple fallback summary
    if not resp or resp.startswith("AI is busy") or resp.startswith("Error:"):
        fb = _fallback_summary(text)
        return JSONResponse({"summary": fb})
    return JSONResponse({"summary": resp})

@app.post("/analyze/entities")
async def recognize_entities(text: str = Form(...)):
    prompt = (
        "Extract named entities and return ONLY a JSON array of objects with keys entity and type "
        '(type one of PERSON, ORG, GEO, DATE). Example: [{"entity":"Elon Musk","type":"PERSON"}]'
    )
    resp = await query_llama_async(prompt, text, max_tokens=800, temperature=0.0)
    arr = extract_json_array(resp)
    if arr is not None:
        return JSONResponse({"entities": arr})
    # fallback: try simple regex-based extraction
    fb = _fallback_entities(text)
    if fb:
        return JSONResponse({"entities": fb})
    return JSONResponse({"entities": {"raw": resp}}, status_code=200)

@app.post("/analyze/key_elements")
async def key_elements(text: str = Form(...), mode: str = Form(None)):
    """Analyze resume for either weaknesses or strengths depending on `mode`.
    mode: 'strengths' | 'weaknesses' | None
    """
    if mode == "strengths":
        prompt = (
            "Extract strengths from the resume. Return ONLY a concise numbered list of strengths (1..N), each as 1 short sentence or bullet."
        )
    elif mode == "weaknesses":
        prompt = (
            "Extract weaknesses from the resume. Return ONLY numbered weaknesses (1..N) with a short suggestion after each (format: Weakness — Suggestion)."
        )
    else:
        prompt = (
            "Analyze resume and produce a 'Weakness and Suggestions' section. "
            "Return only formatted plain text with numbered weaknesses and a 'To improve the resume:' bullet list."
        )

    resp = await query_llama_async(prompt, text, max_tokens=1200, temperature=0.0)
    if not resp or resp.startswith("AI is busy") or resp.startswith("Error:"):
        fb = _fallback_key_elements(text, mode=mode)
        return JSONResponse({"key_elements": fb})
    return JSONResponse({"key_elements": resp})

@app.post("/analyze/qa")
async def qa(text: str = Form(...), question: str = Form(...)):
    prompt = f"Answer concisely (1-3 sentences) based only on the document. Question: {question}"
    resp = await query_llama_async(prompt, text, max_tokens=600, temperature=0.0)
    if not resp or resp.startswith("AI is busy") or resp.startswith("Error:"):
        fb = _fallback_qa(text, question)
        return JSONResponse({"answer": fb})
    return JSONResponse({"answer": resp})

@app.post("/analyze/compare")
async def compare_docs(text1: str = Form(...), text2: str = Form(...)):
    prompt = (
        "Compare Doc1 and Doc2. Output three short sections: Similarities (bullets), Differences (bullets), Notable Points (bullets)."
    )
    combined = f"Doc1:\n{text1[:MAX_TEXT_SNIPPET]}\n\nDoc2:\n{text2[:MAX_TEXT_SNIPPET]}"
    resp = await query_llama_async(prompt, combined, max_tokens=1200, temperature=0.0)
    if not resp or resp.startswith("AI is busy") or resp.startswith("Error:"):
        fb = _fallback_compare(text1, text2)
        return JSONResponse({"comparison": fb})
    return JSONResponse({"comparison": resp})

@app.post("/analyze/parse")
async def parse_resume(file: UploadFile = File(None), text: str = Form(None)):
    resume_text = ""
    if file is not None:
        loop = asyncio.get_event_loop()
        resume_text = await loop.run_in_executor(None, extract_text, file)
    elif text:
        resume_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide either a resume file or text.")

    parse_prompt = (
        "Parse the resume and return STRICT JSON (only the JSON object) with fields: "
        '{"name":"","email":"","phone":"","location":"","summary":"","skills":[],"experience":[{"company":"","title":"","start":"","end":"","description":""}],'
        '"education":[{"degree":"","institution":"","start":"","end":""}], "projects":[],"certifications":[]}'
    )
    raw_parse = await query_llama_async(parse_prompt, resume_text, max_tokens=1200, temperature=0.0)
    parsed_obj = extract_json_object(raw_parse)

    if parsed_obj and isinstance(parsed_obj, dict):
        # build display-friendly formatted summary
        name = parsed_obj.get("name", "")
        summary_text = parsed_obj.get("summary", "")
        skills = parsed_obj.get("skills", [])
        experience = parsed_obj.get("experience", [])
        education = parsed_obj.get("education", [])
        projects = parsed_obj.get("projects", [])

        parts = []
        header = f"{name + ' — ' if name else ''}Resume Summary"
        parts.append(header)
        if summary_text:
            parts.append(summary_text)
        if skills:
            parts.append("\nKey skills:\n" + "\n".join(f"- {s}" for s in skills[:20]))
        if experience:
            parts.append("\nExperience (most recent):")
            for exp in experience[:3]:
                title = exp.get("title", "")
                comp = exp.get("company", "")
                period = f"{exp.get('start','')} - {exp.get('end','')}".strip(" -")
                desc = exp.get("description", "")
                parts.append(f"- {title} at {comp} ({period}): {desc}")
        if education:
            parts.append("\nEducation:")
            for edu in education[:3]:
                deg = edu.get("degree","")
                inst = edu.get("institution","")
                parts.append(f"- {deg} — {inst}")
        if projects:
            parts.append("\nProjects:")
            for p in projects[:5]:
                parts.append(f"- {p}")

        formatted_text = "\n\n".join(parts)
        parsed_return = parsed_obj
    else:
        # fallback: try to produce a minimal parsed object from heuristics
        parsed_return = {"raw": raw_parse}
        try:
            fb_parsed = _fallback_parsed(resume_text)
            # merge fb_parsed into parsed_return under keys if missing
            for k, v in fb_parsed.items():
                if k not in parsed_return:
                    parsed_return[k] = v
        except Exception:
            pass
        format_prompt = (
            "Produce a formatted 'Resume Summary' (2-3 sentence summary, key skills bullets, top 3 suggested job titles)."
        )
        formatted_text = await query_llama_async(format_prompt, resume_text, max_tokens=900, temperature=0.0)
        if not formatted_text or formatted_text.startswith("AI is busy") or formatted_text.startswith("Error:"):
            formatted_text = _fallback_summary(resume_text)

    return JSONResponse({"parsed": parsed_return, "formatted": formatted_text})

# -------------------------
# RapidAPI JSearch helper (async)
# -------------------------
async def fetch_jobs_rapidapi(query: str, page: int = 1, num_pages: int = 1) -> Optional[Dict]:
    if not RAPIDAPI_KEY:
        return None
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "jsearch.p.rapidapi.com"}
    params = {"query": query, "page": str(page), "num_pages": str(num_pages)}
    try:
        async with httpx.AsyncClient(timeout=RAPIDAPI_TIMEOUT) as client_http:
            r = await client_http.get(url, headers=headers, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        log.exception("RapidAPI request failed")
        return None

@app.post("/analyze/job_suggestions")
async def job_suggestions(
    text: str = Form(None),
    parsed_resume: str = Form(None),
    industries: str = Form(""),
    experience_level: str = Form("")
):
    payload_text = ""
    parsed_obj = None
    if parsed_resume:
        try:
            parsed_obj = json.loads(parsed_resume)
            payload_text = json.dumps(parsed_obj)
        except Exception:
            payload_text = parsed_resume
    elif text:
        payload_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide resume text or parsed_resume JSON.")

    prompt_struct = (
       """Generate 6 job roles that match the resume. Return ONLY a JSON array (no extra text).

Each item in the array must be an object with the following keys:
- title: string
- seniority: one of Entry, Mid, Senior
- match_score: integer 0-100
- key_skills: array of strings
- short_description: string (1 short sentence)
- recommended_next_steps: array of 1-3 short actionable steps
- job_link: a realistic but FICTIONAL URL beginning with https://jobs.example.com/ and a short id (e.g. https://jobs.example.com/sr-dev-12345)
- company_link: a realistic but FICTIONAL URL beginning with https://company.example.com/ and a short id

Make all company names and links fictional but realistic. Return strictly valid JSON array. Do not output any explanatory text or markdown.
"""
    )
    prompt_textual = (
        "Also produce a human-friendly 'Job Suggestions' section formatted as numbered items (1..6). For each item include Title (Seniority) — match score, a 1-sentence short description, 1-2 recommended next steps, and include Apply and Company links (use the same fictional links as in the structured JSON)."
    )

    # Structured + textual suggestions (use LLM helper with timeouts)
    resp_struct = await query_llama_async(prompt_struct, payload_text, max_tokens=1200, temperature=0.0)
    jobs_array = extract_json_array(resp_struct)

    resp_textual = await query_llama_async(prompt_textual, payload_text, max_tokens=900, temperature=0.0)

    if jobs_array and isinstance(jobs_array, list):
        for j in jobs_array:
            if "match_score" in j:
                try:
                    j["match_score"] = int(float(j["match_score"]))
                except Exception:
                    pass
    else:
        jobs_array = None

    # Fallback: if the model didn't return structured jobs, generate realistic fictional job items
    if not jobs_array:
        try:
            sample_titles = [
                "Software Engineer",
                "Data Scientist",
                "Backend Developer",
                "Machine Learning Engineer",
                "Product Analyst",
                "DevOps Engineer",
            ]
            fake_jobs: List[Dict] = []
            # try to extract some skills to make fake items feel realistic
            sample_skills = []
            if parsed_obj and isinstance(parsed_obj, dict):
                ks = parsed_obj.get("skills") or parsed_obj.get("skillset") or parsed_obj.get("skills_list")
                if isinstance(ks, list):
                    sample_skills = [str(x) for x in ks[:8]]
                elif isinstance(ks, str) and ks:
                    sample_skills = [s.strip() for s in ks.split(",") if s.strip()][:8]

            for idx in range(6):
                t = sample_titles[idx % len(sample_titles)]
                # make company look realistic
                comp = f"{t.split()[0]}Works" if idx % 2 == 0 else f"{t.split()[0]}Labs"
                key_sk = sample_skills[:4] if sample_skills else ["communication", "teamwork", "problem solving"]
                match_score = max(40, 85 - idx * 5)
                job_id = f"{t.lower().replace(' ','-')}-{1000+idx}"
                comp_id = f"{comp.lower().replace(' ','-')}"
                fake = {
                    "title": t,
                    "seniority": "Mid" if idx % 3 == 1 else ("Senior" if idx % 3 == 2 else "Entry"),
                    "match_score": match_score,
                    "key_skills": key_sk,
                    "short_description": f"Fictional {t} role at {comp} suitable for candidates with experience in {', '.join(key_sk[:3])}.",
                    "recommended_next_steps": ["Tailor your resume to the role", "Highlight relevant projects", "Apply via company portal"],
                    "job_link": f"https://jobs.example.com/{job_id}",
                    "company_link": f"https://company.example.com/{comp_id}",
                    "company": comp,
                }
                fake_jobs.append(fake)

            jobs_array = fake_jobs
            # create a simple human-readable formatted string consistent with front-end display
            textual_lines = []
            for i, fj in enumerate(fake_jobs, start=1):
                textual_lines.append(f"{i}. {fj['title']} ({fj['seniority']}) — {fj['match_score']}%\n{fj['short_description']}\nApply: {fj['job_link']} | Company: {fj['company_link']}")
            resp_textual = "\n\n".join(textual_lines)
        except Exception:
            # final fallback minimal structure
            jobs_array = [{"title": "Software Engineer", "seniority": "Mid", "match_score": 50, "key_skills": ["programming"], "short_description": "Fallback role.", "recommended_next_steps": ["Update resume"], "job_link": "https://jobs.example.com/se-1000", "company_link": "https://company.example.com/seworks"}]
            resp_textual = "1. Software Engineer (Mid) — 50%\nFallback role.\nApply: https://jobs.example.com/se-1000 | Company: https://company.example.com/seworks"

    # Live RapidAPI (best-effort)
    live_jobs = None
    if RAPIDAPI_KEY:
        query_terms = ""
        if parsed_obj and isinstance(parsed_obj, dict):
            skills = parsed_obj.get("skills", [])
            if isinstance(skills, list) and skills:
                query_terms = " ".join(str(s) for s in skills[:6])
            else:
                query_terms = parsed_obj.get("summary", "")[:300]
        if not query_terms and jobs_array:
            titles = [j.get("title", "") for j in jobs_array[:3] if j.get("title")]
            query_terms = " ".join(titles)
        if not query_terms:
            query_terms = "software engineer"
        try:
            live_jobs = await fetch_jobs_rapidapi(query_terms, page=1, num_pages=1)
        except Exception:
            live_jobs = None

    result = {
        "jobs": jobs_array if jobs_array else {"raw": resp_struct},
        "jobs_formatted": resp_textual,
        "live_jobs": live_jobs or {"note": "No RapidAPI key configured or no live results"}
    }
    return JSONResponse(result)

@app.post("/analyze/linkedin_jobs")
async def linkedin_jobs(query: str = Form(...), page: int = Form(1), num_pages: int = Form(1)):
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=400, detail="RapidAPI key not configured. Add X_RAPIDAPI_KEY to .env")
    resp = await fetch_jobs_rapidapi(query=query, page=page, num_pages=num_pages)
    if resp is None:
        return JSONResponse({"error": "Failed to fetch live jobs or no results"}, status_code=500)
    return JSONResponse({"results": resp})

# -------------------------
# Run server (dev)
# -------------------------

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

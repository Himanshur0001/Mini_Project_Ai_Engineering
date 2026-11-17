
"""
AI RESUME SCREENING SYSTEM
-------------------------
Project flow:
1. Amazon ki job description read karna.
2. Groq LLM se job ki structured information extract karwana.
3. "resumes" folder se PDF/DOCX resumes read karna.
4. Har resume se candidate ki information LLM ki help se extract karna.
5. Har resume ko job description ke saath compare karke match score calculate karna.
6. Top 2 aur sabse kam score wale 2 candidates display karna.

Run karne se pehle:
    uv add groq python-dotenv pydantic pypdf python-docx

Expected folder structure:
    day5/
        mini_project.py
        .env
        resumes/
            candidate1.pdf
            candidate2.docx

Tumhari .env file mein ye hona chahiye:
    GROQ_API_KEY=your_actual_api_key
"""

# -------------------- 1. IMPORTS --------------------

import os                       # Environment variables jaise API key read karne ke liye.
import time                     # Do API requests ke beech kuch seconds rukne ke liye.
import json                     # LLM se aaye JSON text ko Python dictionary mein convert karne ke liye.
from pathlib import Path        # Files aur folders ke paths ko easily handle karne ke liye.

from dotenv import load_dotenv  # .env file ke variables ko load karne ke liye.
from groq import Groq            # Groq API ke saath connection banane ke liye.
from pydantic import BaseModel, Field  # Data ka structure define aur validate karne ke liye.

from pypdf import PdfReader      # PDF resume se text extract karne ke liye.
from docx import Document        # Word (.docx) resume se text extract karne ke liye.


# -------------------- 2. LOAD API KEY AND CREATE GROQ CLIENT --------------------

# Current project directory mein available .env file ko load karo.
load_dotenv()

# Environment variable se Groq API key nikalo.
my_api_key = os.getenv("GROQ_API_KEY")

# Agar API key nahi mili toh program ko error ke saath stop kar do.
if not my_api_key:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Groq client create karo. Isi client ki help se hum LLM ko prompts bhejenge.
client = Groq(api_key=my_api_key)

# LLM model select karo jo job description aur resumes ko process karega.
model = "openai/gpt-oss-120b"


# -------------------- 3. JOB DESCRIPTION --------------------

# Ye Amazon ki job description hai.
# Isi ke basis par sabhi candidates ke resumes compare honge.
job_description = """
Description

Do you want to solve real customer problems through innovative technology?
Do you enjoy working on scalable services in a collaborative team environment?
Do you want to see your code directly impact millions of customers worldwide?

At Amazon, we hire the best minds in technology to innovate and build on behalf
of our customers. Customer obsession is part of our company DNA, which has made
us one of the world's most beloved brands.

Our Software Development Engineers (SDEs) use modern technology to solve complex
problems while seeing their work's impact first-hand. The challenges SDEs solve
at Amazon are meaningful and influence millions of customers, sellers, and
products globally. We seek individuals passionate about creating new products,
features, and services while managing ambiguity in an environment where
development cycles are measured in weeks, not years.

At Amazon, we believe in ownership at every level. As an SDE-I, you'll own the
entire lifecycle of your code - from design through deployment and ongoing
operations. This ownership mindset, combined with our commitment to operational
excellence, ensures we deliver the highest quality solutions for our customers.

We're looking for curious minds who think big and want to define tomorrow's
technology. At Amazon, you'll grow into the high-impact engineer you know you
can be, supported by a culture of learning and mentorship. Every day brings
exciting new challenges and opportunities for personal growth.

Key job responsibilities

• Collaborate and communicate effectively with experienced cross-disciplinary
  Amazonians to design, build, and operate innovative products and services
  that delight our customers, while participating in technical discussions
  to drive solutions forward.

• Design and develop scalable solutions using cloud-native architectures and
  microservices in a large distributed computing environment.

• Participate in code reviews and contribute to technical documentation.

• Build and maintain resilient distributed systems that are scalable,
  fault-tolerant, and cost-effective.

• Leverage and contribute to the development of GenAI and AI-powered tools
  to enhance development productivity while staying current with emerging
  technologies.

• Write clean, maintainable code following best practices and design patterns.

• Work in an agile environment practicing CI/CD principles while participating
  in operational responsibilities including on-call duties.

• Demonstrate operational excellence through monitoring, troubleshooting,
  and resolving production issues.

Basic Qualifications

- Experience with at least one general-purpose programming language such as
  Java, Python, C++, C#, Go, Rust, or TypeScript.
- Experience with data structure implementation, basic algorithm development,
  and/or object-oriented design principles.
- Currently has, or is in the process of obtaining a bachelor's degree in
  Computer Science, Computer Engineering, Data Science, Information Systems,
  or related STEM fields.
- Must be 18 years of age or older.

Preferred Qualifications

- Experience from previous technical internship(s) or demonstrated project
  experience.
- Experience with one or more of the following: AI tools for development
  productivity, Cloud platforms (preferably AWS), Database systems (SQL and
  NoSQL), contributing to open-source projects, version control systems,
  debugging and troubleshooting complex systems.
- Demonstrated ability to learn and adapt to new technologies quickly.
- Basic understanding of software development lifecycle (SDLC).
- Strong problem-solving and analytical skills.
- Excellent written and verbal communication skills.
"""


# -------------------- 4. PYDANTIC MODEL FOR JOB DESCRIPTION --------------------

# Pydantic model job description se extract hone wale data ka structure define karta hai.
# LLM jo JSON return karega, usko hum is model ke according validate karenge.
class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]


# Pydantic model se JSON Schema generate karo.
# Is schema ko prompt mein bhejenge taaki LLM ko exact output structure pata ho.
jobd_schema = JobD.model_json_schema()


# -------------------- 5. ASK THE LLM TO ANALYZE THE JOB DESCRIPTION --------------------

# System prompt LLM ko batata hai ki uska role kya hai aur use kya karna hai.
system_prompt = f"""
You are an expert HR assistant.

Your job is to analyze job descriptions and extract structured information
from them.

Return ONLY valid JSON matching this schema:

{jobd_schema}

IMPORTANT:
Do NOT return the schema itself.
Do NOT return fields like "properties", "title" or "type".
Fill the schema with actual information extracted from the job description.

If minimum experience is not mentioned, return null.
If information for a list is missing, return an empty list.
Do not invent information.
"""

# User prompt mein actual job description bheji jaati hai.
user_prompt = f"""
Analyze the following job description:

{job_description}
"""

# System message LLM ko uska role aur instructions deta hai.
message_system = {
    "role": "system",
    "content": system_prompt
}

# User message mein actual job description hoti hai.
message_user = {
    "role": "user",
    "content": user_prompt
}

# LLM ko normal text ke bajaye JSON object return karne ke liye bol rahe hain.
response_format = {
    "type": "json_object"
}

# Dono messages ko ek list mein store kar rahe hain.
messages = [message_system, message_user]

# Groq API ko request bhejo aur job description analyze karwao.
response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

# LLM ke response se actual text nikalo.
# Is stage par humein JSON format mein text mil chuka hai.
answer = response.choices[0].message.content
raw_json = answer


# -------------------- 6. CONVERT JOB JSON INTO A PYDANTIC OBJECT --------------------

# json.loads() JSON string ko normal Python dictionary mein convert karta hai.
job_data = json.loads(raw_json)

# **job_data dictionary ki saari keys aur values ko keyword arguments ki tarah pass karta hai.
# JobD model data ko validate karta hai aur Pydantic object create karta hai.
job = JobD(**job_data)

# Check karne ke liye extracted information display karo.
# Agar minimum experience mention nahi hai toh None print hoga.
print(job.minimum_experience)
print(job.education_requirements)


# -------------------- 7. PYDANTIC MODELS FOR RESUME MATCHING --------------------

# Ye model LLM se milne wale final resume comparison result ka structure define karta hai.
class MatchResult(BaseModel):
    score: float       # Candidate ka overall match score number mein store hoga.
    details: dict      # Matching skills aur missing skills jaisi details store hongi.


# Ye model resume mein available ek work experience ki information represent karta hai.
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


# Ye model define karta hai ki humein candidate ke resume se kaunsi information chahiye.
class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


# Resume model ka JSON Schema generate karo.
# Is schema se LLM ko pata chalega ki resume se kaun-kaun se fields extract karne hain.
resume_schema = Resume.model_json_schema()


# -------------------- 8. FUNCTION: COMPARE JOB DESCRIPTION WITH RESUME --------------------

def final_score(job, resume):
    """
    Ek parsed resume ko job description ke saath compare karta hai.

    Parameters:
        job: JobD Pydantic object jisme job requirements hain.
        resume: Resume Pydantic object jisme candidate ki information hai.

    Returns:
        MatchResult object jisme candidate ka score aur comparison details hongi.
    """

    # Comparison result ke liye JSON Schema create karo.
    match_schema = MatchResult.model_json_schema()

    # Job aur resume dono Pydantic objects ko JSON format mein convert karke prompt mein bhejo.
    prompt = f"""
    You are an HR recruiter.

    Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}

    Return JSON matching this schema:
    {match_schema}

    Give me:
    1. Candidate name
    2. Matching skills
    3. Missing important skills
    4. Whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. A short final verdict

    Keep the response concise and easy to read.
    """

    # Comparison prompt ko user message ke form mein prepare karo.
    message = {
        "role": "user",
        "content": prompt
    }

    # Message ko list mein store karo.
    messages = [message]

    # LLM se JSON object return karne ko bolo.
    response_format = {
        "type": "json_object"
    }

    # Groq API ko request bhejo aur candidate ka score calculate karwao.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format
    )

    # LLM se aaye JSON text ko Python dictionary mein convert karo.
    data = json.loads(response.choices[0].message.content)

    # MatchResult model se data validate karo aur Pydantic object return karo.
    return MatchResult(**data)


# -------------------- 9. FUNCTION: EXTRACT STRUCTURED DATA FROM A RESUME --------------------

def parse_resume(resume_text):
    """
    Resume ka extracted text LLM ko bhejta hai.
    LLM structured JSON return karta hai, jise Resume model se validate karte hain.
    """

    # LLM ko batate hain ki use resume parser ki tarah kaam karna hai.
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:
    {resume_schema}

    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    # Resume se nikala gaya actual text user prompt mein add karo.
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """

    # System message create karo jo LLM ko instructions dega.
    message_system = {
        "role": "system",
        "content": system_prompt
    }

    # User message create karo jisme candidate ka resume text hoga.
    message_user = {
        "role": "user",
        "content": user_prompt
    }

    # Dono messages ko ek list mein store karo.
    messages = [message_system, message_user]

    # LLM se JSON format mein output maango.
    response_format = {
        "type": "json_object"
    }

    # Groq API ko resume parsing ki request bhejo.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format
    )

    # LLM ke response se JSON text nikalo.
    raw_output = response.choices[0].message.content

    # JSON text ko Python dictionary mein convert karo.
    data = json.loads(raw_output)

    # Resume Pydantic model ki help se extracted data validate karo.
    resume = Resume(**data)

    return resume


# -------------------- 10. FUNCTION: READ TEXT FROM PDF --------------------

def read_pdf(file_path):
    """PDF file ke sabhi pages se available text extract karta hai."""

    # PDF file ko open karo.
    reader = PdfReader(file_path)

    text = ""

    # PDF ke har page par ek-ek karke loop chalao.
    for page in reader.pages:

        # Current page se text extract karo.
        page_text = page.extract_text()

        # Kuch PDF pages mein extract karne layak text nahi hota.
        if page_text:
            text += page_text + "\n"

    return text


# -------------------- 11. FUNCTION: READ TEXT FROM DOCX --------------------

def read_docx(file_path):
    """Word document ke paragraphs aur tables dono se text extract karta hai."""

    # Word document ko open karo.
    document = Document(file_path)

    text = ""

    # Document ke normal paragraphs se text extract karo.
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    # Tables ke andar stored text ko bhi extract karo.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text += cell.text + "\n"

    return text


# -------------------- 12. FUNCTION: CHOOSE HOW TO READ A RESUME --------------------

def read_resume(file_path):
    """
    File ka extension check karta hai aur uske according correct reading function call karta hai.
    Is project mein sirf PDF aur DOCX files supported hain.
    """

    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)

    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)

    else:
        # Agar file ka format supported nahi hai toh None return karo.
        return None


# -------------------- 13. MAIN PROCESS: PROCESS ALL RESUMES --------------------

# Program expect karta hai ki day5 folder ke andar "resumes" naam ka folder available ho.
resume_folder = Path("resumes")

# Is list mein har candidate ka naam, score aur comparison details store hongi.
all_results = []

# Resume files read karne se pehle check karo ki resumes folder exist karta hai ya nahi.
if not resume_folder.exists():
    raise FileNotFoundError(
        "The 'resumes' folder was not found. Create it inside the day5 folder."
    )

# Resumes folder ke andar available har file par loop chalao.
for file_path in resume_folder.iterdir():

    # Jo files PDF ya DOCX nahi hain, unhe skip kar do.
    if file_path.suffix.lower() not in [".pdf", ".docx"]:
        continue

    print("\nProcessing:", file_path.name)

    # Step 1: Current resume file se plain text extract karo.
    resume_text = read_resume(file_path)

    # Agar resume se text extract nahi hua toh us file ko skip kar do.
    if not resume_text or not resume_text.strip():
        print("No readable text found. Skipping this file.")
        continue

    # Step 2: Resume text ko structured data mein convert karne ke liye pehli LLM call karo.
    parsed_resume = parse_resume(resume_text)

    # Next API request bhejne se pehle 5 seconds wait karo.
    # Isse requests ke beech gap milta hai, lekin rate limit follow hone ki guarantee nahi hai.
    time.sleep(5)

    # Step 3: Parsed resume ko job description ke saath compare karne ke liye doosri LLM call karo.
    result = final_score(job, parsed_resume)

    # Agla resume process karne se pehle wait karo.
    time.sleep(5)

    # Current candidate ka match score display karo.
    print("Score:", result.score)

    # Candidate ki information aur score ko all_results list mein save karo.
    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })


# -------------------- 14. SORT CANDIDATES BY MATCH SCORE --------------------

# Candidates ko score ke basis par descending order mein sort karo.
# reverse=True ka matlab hai highest score sabse pehle aayega.
all_results.sort(
    key=lambda candidate: candidate["score"],
    reverse=True
)

# Starting ke 2 candidates select karo, jinka score sabse zyada hai.
top_2 = all_results[:2]

# List ke last ke 2 candidates select karo, jinka score sabse kam hai.
worst_2 = all_results[-2:]


# -------------------- 15. DISPLAY TOP 2 CANDIDATES --------------------

print("\nTOP 2 CANDIDATES")

# Top 2 candidates par loop chalao.
for candidate in top_2:
    print(
        candidate["name"],
        "-",
        candidate["score"],
        "%"
    )

    # Candidate ke score ki explanation bhi print karo.
    print(candidate["details"])


# -------------------- 16. DISPLAY LOWEST 2 CANDIDATES --------------------

print("\nLOWEST 2 CANDIDATES")

# Lowest 2 candidates par loop chalao.
for candidate in worst_2:
    print(
        candidate["name"],
        "-",
        candidate["score"],
        "%"
    )

    # Candidate ko kam score milne ka reason print karo.
    print(candidate["details"])

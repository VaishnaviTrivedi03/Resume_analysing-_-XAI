import os
import warnings
import numpy as np
import pandas as pd
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge

# ---------------------------
# Google Generative AI import
# ---------------------------
import google.generativeai as genai

# =============================================================================
#   1. Suppress Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
#   2. Configuration and Setup
# =============================================================================
# Securely load the Gemini API key or use an environment variable (recommended).
GENIUS_API_KEY = "AIzaSyBWf9DqAOLq26z1IxV0gb4u1ZTkcwpnigM"  # Replace with your key or use os.getenv('GENIUS_API_KEY')
if not GENIUS_API_KEY:
    raise ValueError("Please set the 'GENIUS_API_KEY' environment variable or your API key.")

# Configure the Generative AI with your API key
genai.configure(api_key=GENIUS_API_KEY)

# Define the Gemini model — update to the desired or latest model version
model_name = 'gemini-1.5-flash'

# =============================================================================
#   3. File Reading Functions
# =============================================================================
try:
    import PyPDF2
except ImportError:
    raise ImportError("Please install PyPDF2: pip install PyPDF2")

try:
    import docx
except ImportError:
    raise ImportError("Please install python-docx: pip install python-docx")

def read_resume_file(filepath: str) -> str:
    """
    Reads a resume file (PDF, DOCX, or TXT) and returns its textual content.
    """
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        elif ext == ".docx":
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Unsupported file extension {ext} for file {filepath}. Skipping...")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return text.strip()

# =============================================================================
#   4. Load Resumes from a Local Folder
# =============================================================================
resume_dir = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\resumes"  # <-- Update this path to your resumes folder
resumes = {}

for file in os.listdir(resume_dir):
    filepath = os.path.join(resume_dir, file)
    if os.path.isfile(filepath) and file.lower().endswith((".pdf", ".docx", ".txt")):
        content = read_resume_file(filepath)
        if content:
            resumes[file] = content

if not resumes:
    raise ValueError("No resumes were loaded. Please check the folder path and file formats.")

# =============================================================================
#   5. Job Description Text
# =============================================================================
jd_text = """
We are looking for a Software Engineer with experience in Python, machine learning, and data analysis.
The candidate should be proficient in developing scalable applications, have a strong understanding of algorithms,
and possess excellent problem-solving skills. Familiarity with cloud platforms such as AWS or Azure is a plus.
"""

# =============================================================================
#   6. Text Vectorization using TF-IDF
# =============================================================================
documents = [jd_text] + list(resumes.values())

vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
tfidf_matrix = vectorizer.fit_transform(documents)

jd_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]

# =============================================================================
#   7. Similarity Computation (Cosine Similarity)
# =============================================================================
similarities = cosine_similarity(resume_vectors, jd_vector).flatten()

ranking_df = pd.DataFrame({
    'Resume': list(resumes.keys()),
    'Similarity_Score': similarities
})

# Sort descending by similarity
ranking_df = ranking_df.sort_values(by='Similarity_Score', ascending=False).reset_index(drop=True)
ranking_df['Rank'] = ranking_df.index + 1

print("### Ranked Resumes:")
print(ranking_df[['Rank', 'Resume', 'Similarity_Score']].to_markdown(index=False))

# =============================================================================
#   8. SHAP Explanation
# =============================================================================
def predict_similarity(x):
    """
    Predicts cosine similarity to the Job Description vector for each input TF-IDF vector.
    """
    return cosine_similarity(x, jd_vector).flatten()

# Use Ridge to avoid LassoLarsIC issues
custom_regressor = Ridge(alpha=1.0)

# Create a KernelExplainer
explainer = shap.KernelExplainer(
    predict_similarity,
    tfidf_matrix[0:1],  # background data
    model_regressor=custom_regressor
)

shap_values = explainer.shap_values(resume_vectors, nsamples=100)
feature_names = vectorizer.get_feature_names_out()

# Print SHAP values for each resume
print("\n=== SHAP Scores for Each Resume ===")
for idx, row in ranking_df.iterrows():
    resume_name = row['Resume']
    rank = row['Rank']
    similarity_score = row['Similarity_Score']
    shap_val = shap_values[idx]

    shap_contrib = dict(zip(feature_names, shap_val))
    sorted_shap_contrib = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n=== SHAP Scores for {resume_name} (Rank {rank}, Score {similarity_score:.4f}) ===")
    print("Feature (Keyword) and SHAP Value:")
    for feature, score in sorted_shap_contrib[:10]:
        print(f"  {feature}: {score:.4f}")

# =============================================================================
#   9. AI Explanation Generation Functions
# =============================================================================
def generate_ai_explanation(resume_name: str, rank: int, score: float, shap_contributions: dict, top_n: int = 10) -> str:
    """
    Generates an HR-focused explanation using Gemini based on SHAP contributions.
    """
    sorted_feats = sorted(shap_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top_feats = sorted_feats[:top_n]
    pos_feats = {k: v for k, v in top_feats if v > 0}
    neg_feats = {k: v for k, v in top_feats if v < 0}

    prompt = f"""
You are an experienced HR analyst specializing in technical recruitment.
Your task is to evaluate the alignment of a candidate's resume with a specific Job Description.

**Resume Name**: {resume_name}
**Job Description Similarity Score**: {score:.4f}
**Rank**: #{rank}

**Top Positive Contributions (indicating strong alignment)**:
- {'\n'.join([f"{k} ({v:.4f})" for k, v in pos_feats.items()])}

**Top Negative Contributions (indicating gaps or weaker alignment)**:
- {'\n'.join([f"{k} ({v:.4f})" for k, v in neg_feats.items()])}

---
**Instructions**:
Provide a comprehensive and professional explanation for the HR department detailing why this resume achieved its current ranking.
Focus on the candidate’s strengths and any potential gaps. Use the features above to illustrate how the candidate’s experience
matches or falls short of the job requirements (e.g., Python, machine learning, data analysis, and cloud platforms like AWS/Azure).
Your explanation should offer clear, data-driven insights into how each feature influences alignment with the role requirements.
Use structured paragraphs and maintain a formal tone.
"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=400,
                temperature=0.3
            )
        )
        explanation = response.text.strip()
    except Exception as e:
        explanation = f"Error generating explanation: {e}"

    return explanation

def generate_improvement_suggestions(resume_name: str, shap_contributions: dict, top_n: int = 10) -> str:
    """
    Generates constructive feedback for the applicant on improving the resume's alignment.
    """
    sorted_feats = sorted(shap_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top_feats = sorted_feats[:top_n]
    pos_feats = {k: v for k, v in top_feats if v > 0}
    neg_feats = {k: v for k, v in top_feats if v < 0}

    prompt = f"""
You are a seasoned career consultant with expertise in resume optimization for technical roles.
Provide personalized guidance to a job seeker based on their resume analysis.

**Resume Name**: {resume_name}

**Top Positive Features (strengths to emphasize)**:
1. {'\n'.join([f"{k} ({v:.4f})" for k, v in pos_feats.items()])}

**Top Negative Features (areas needing improvement)**:
1. {'\n'.join([f"{k} ({v:.4f})" for k, v in neg_feats.items()])}

---
**Instructions**:
Based on the above analysis, provide a structured list of clear, actionable, and detailed suggestions
to help the candidate enhance their resume. Focus on:
- **Showcasing Relevant Skills**: How to better highlight skills such as Python, machine learning, data analysis.
- **Addressing Gaps**: Strategies to incorporate mentions of AWS/Azure, developing scalable applications.
- **Professional Language and Formatting**: Recommendations for improving clarity and layout.

Present the suggestions as a numbered list for clarity.
"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=400,
                temperature=0.3
            )
        )
        improvement_text = response.text.strip()
    except Exception as e:
        improvement_text = f"Error generating improvement suggestions: {e}"

    return improvement_text

# =============================================================================
#   10. Generate HR Table & Applicant Table
# =============================================================================
hr_table_data = []
applicant_table_data = []

for idx, row in ranking_df.iterrows():
    resume_name = row['Resume']
    rank = row['Rank']
    score = row['Similarity_Score']
    shap_val = shap_values[idx]

    shap_contrib = dict(zip(feature_names, shap_val))

    # HR Explanation
    hr_explanation = generate_ai_explanation(
        resume_name=resume_name,
        rank=rank,
        score=score,
        shap_contributions=shap_contrib,
        top_n=10
    )

    # Improvement Suggestions
    improvement_suggestions = generate_improvement_suggestions(
        resume_name=resume_name,
        shap_contributions=shap_contrib,
        top_n=10
    )

    # Collect data for the HR Table
    hr_table_data.append({
        "Resume": resume_name,
        "Rank": rank,
        "Similarity Score": f"{score:.4f}",
        "HR Remarks": hr_explanation
    })

    # Collect data for the Applicant Table
    applicant_table_data.append({
        "Resume": resume_name,
        "Improvement Suggestions": improvement_suggestions
    })

# Create DataFrames for HR and Applicant tables
hr_table = pd.DataFrame(hr_table_data)
applicant_table = pd.DataFrame(applicant_table_data)

# =============================================================================
#   11. Display the Tables (Markdown) & Save Them
# =============================================================================
def display_markdown_table(df: pd.DataFrame, title: str):
    """
    Print a pandas DataFrame as a Markdown table with a given title.
    """
    print(f"\n### {title}")
    print(df.to_markdown(index=False))

# Display HR Table
display_markdown_table(hr_table, "HR Table: Professional Remarks for Each Resume")

# Display Applicant Table
display_markdown_table(applicant_table, "Applicant Table: Detailed Improvement Suggestions")

# Save the tables to CSV files
hr_table.to_csv("hrtry_table.csv", index=False, encoding="utf-8-sig")
applicant_table.to_csv("applicanttry_table.csv", index=False, encoding="utf-8-sig")

# Optionally, you can also save them as Excel:
# hr_table.to_excel("hr_table.xlsx", index=False)
# applicant_table.to_excel("applicant_table.xlsx", index=False)

# If you are using Jupyter Notebook, you can style the tables as follows:
try:
    from IPython.display import display, Markdown

    def display_properly(df: pd.DataFrame, title: str):
        display(Markdown(f"### {title}"))
        display(df.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap'
        }))

    display_properly(hr_table, "HR Table: Professional Remarks for Each Resume")
    display_properly(applicant_table, "Applicant Table: Detailed Improvement Suggestions")
except ImportError:
    pass

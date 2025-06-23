import os
import numpy as np
import pandas as pd
import shap
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================
#   Configuration and Setup
# ============================

# Configure the Gemini API
genai.configure(api_key="AIzaSyBWf9DqAOLq26z1IxV0gb4u1ZTkcwpnigM")

model_name = 'gemini-1.5-flash'

# ============================
#   Input Job Description
# ============================

jd_text = """
We're hiring a machine learning engineer with a focus on deploying ML models into production environments. 
Candidates should have expertise in Python, TensorFlow or PyTorch, and be familiar with MLOps tools like Docker, Kubernetes, and cloud platforms such as AWS or GCP. 
A good understanding of CI/CD practices, RESTful APIs, and collaboration in Agile teams is essential.
"""

# ============================
#   Sample Resumes
# ============================

resumes = {
    "Resume_1": """
Experienced Software Developer with expertise in Java and cloud computing. Skilled in AWS and DevOps practices.
Proven ability to develop scalable applications and manage databases.
""",
    "Resume_2": """
Data Scientist proficient in Python, machine learning, and data visualization.
Experience with TensorFlow, pandas, and SQL. Strong background in statistical analysis and algorithm development.
""",
    "Resume_3": """
Frontend Developer specializing in React and JavaScript. Adept at creating responsive and user-friendly interfaces.
Knowledge of HTML, CSS, and UX/UI design principles.
""",
    "Resume_4": """
Machine Learning Engineer with experience in Python, deep learning, and natural language processing.
Worked on projects involving image recognition and predictive modeling.
""",
    "Resume_5": """
Backend Developer skilled in Python, Django, and RESTful APIs. Familiar with Docker and Kubernetes for containerization.
Experience in database management and optimization.
""",
}

# ============================
#   Text Vectorization
# ============================

documents = [jd_text] + list(resumes.values())
doc_names = ['Job_Description'] + list(resumes.keys())

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# ============================
#   Similarity Computation
# ============================

jd_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]
similarities = cosine_similarity(resume_vectors, jd_vector).flatten()

df = pd.DataFrame({
    'Resume': list(resumes.keys()),
    'Similarity_Score': similarities
})
df = df.sort_values(by='Similarity_Score', ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

print("Ranked Resumes:")
print(df[['Rank', 'Resume', 'Similarity_Score']])

# ============================
#   Explainable AI with SHAP
# ============================

def predict_similarity(x):
    return cosine_similarity(x, jd_vector).flatten()

explainer = shap.KernelExplainer(predict_similarity, tfidf_matrix[0:1])
shap_values = explainer.shap_values(resume_vectors, nsamples=100)

# ============================
#   AI Explanation Generation
# ============================

def generate_ai_explanation(resume_name, rank, score, shap_contributions, feature_names, top_n=10):
    sorted_feats = sorted(shap_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top_feats = sorted_feats[:top_n]

    pos_feats = {k: v for k, v in top_feats if v > 0}
    neg_feats = {k: v for k, v in top_feats if v < 0}

    prompt = f"""
You are an AI HR expert. For the following resume match result, provide a professional, human-readable evaluation that includes:

1. Positive Feedback (Why it ranks high)
2. Negative Feedback (Why it is not selected or not ranked higher)
3. Suggestions to improve alignment with the JD

Resume Name: {resume_name}
Rank: {rank}
Similarity Score: {score:.4f}

Positive SHAP contributions (word: score):
{', '.join([f"{k} ({v:.4f})" for k, v in pos_feats.items()])}

Negative SHAP contributions (word: score):
{', '.join([f"{k} ({v:.4f})" for k, v in neg_feats.items()])}

The explanation should use structured headings and be written in clear, human-friendly, feedback-oriented style.
"""

    try:
        response = genai.GenerativeModel(model_name).generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=500,
                temperature=0.7
            )
        )
        explanation = response.text.strip()
    except Exception as e:
        explanation = f"Error generating explanation: {e}"
    return explanation

# ============================
#   Generate and Print Explanations
# ============================

print("\nDetailed Explanations:\n")

feature_names = vectorizer.get_feature_names_out()

for idx, row in df.iterrows():
    resume_name = row['Resume']
    rank = row['Rank']
    score = row['Similarity_Score']
    shap_val = shap_values[idx]

    shap_contrib = dict(zip(feature_names, shap_val))

    explanation = generate_ai_explanation(
        resume_name=resume_name,
        rank=rank,
        score=score,
        shap_contributions=shap_contrib,
        feature_names=feature_names,
        top_n=10
    )

    print(f"=== Explanation for {resume_name} (Rank #{rank}) ===")
    print(explanation)
    print("\n" + "-"*100 + "\n")


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
hr_table.to_csv("hr_table.csv", index=False, encoding="utf-8-sig")
applicant_table.to_csv("job_table.csv", index=False, encoding="utf-8-sig")

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

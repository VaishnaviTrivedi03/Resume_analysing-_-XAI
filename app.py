from flask import Flask, request, render_template_string, send_file
import os
import subprocess
import csv

app = Flask(__name__)

# Set the upload folder to your desired directory
UPLOAD_FOLDER = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to index.html in the same directory as app.py
INDEX_HTML_PATH = os.path.join(os.getcwd(), 'index.html')

@app.route('/')
def index():
    # Load and render index.html (with your original beautiful styling)
    with open(INDEX_HTML_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if not file.filename.lower().endswith('.pdf'):
        return "Only PDF files are allowed!", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return f"File '{file.filename}' uploaded successfully to {UPLOAD_FOLDER}!", 200

@app.route('/run', methods=['POST'])
def run_script():
    script_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\try.py"
    venv_python = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\myenv\Scripts\python.exe"

    try:
        result = subprocess.run(
            [venv_python, script_path],
            capture_output=True,
            text=True
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\nError: {result.stderr}"
    except Exception as e:
        output = f"Exception occurred while running script: {str(e)}"

    # (Your HTML remains unchanged)
    ...

    # A new, professional design for the script output page
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Script Output</title>
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
        body {{
          margin: 0;
          padding: 40px;
          font-family: 'Roboto', sans-serif;
          background: linear-gradient(135deg, #ff758c, #ff7eb3);
          color: #333;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
        }}
        .container {{
          background: #fff;
          border-radius: 15px;
          padding: 40px;
          max-width: 800px;
          width: 90%;
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
          animation: fadeIn 0.8s ease-in-out;
        }}
        h2 {{
          color: #ff758c;
          margin-bottom: 20px;
          text-align: center;
        }}
        pre {{
          background: #f9f9f9;
          padding: 20px;
          border-radius: 8px;
          max-height: 400px;
          overflow-y: auto;
          font-size: 0.95rem;
          border: 1px solid #ececec;
        }}
        a.button {{
          display: inline-block;
          margin-top: 30px;
          text-decoration: none;
          text-align: center;
          background: #ff7eb3;
          color: #fff;
          padding: 12px 30px;
          border-radius: 30px;
          transition: background 0.3s ease;
        }}
        a.button:hover {{
          background: #ff6a9e;
        }}
        @keyframes fadeIn {{
          from {{ opacity: 0; transform: translateY(10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h2>Script Output</h2>
        <pre>{output}</pre>
        <a class="button" href="/">Back to Home</a>
      </div>
    </body>
    </html>
    """
    return html_output

# Endpoints to serve CSV files directly or as downloadable attachments
@app.route('/hr_table')
def hr_table():
    hr_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\hr_table.csv"
    try:
        return send_file(hr_csv_path, mimetype='text/csv', as_attachment=False)
    except Exception as e:
        return f"Error serving HR table: {e}", 500

@app.route('/job_table')
def job_table():
    job_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\job_table.csv"
    try:
        return send_file(job_csv_path, mimetype='text/csv', as_attachment=False)
    except Exception as e:
        return f"Error serving Job Seeker table: {e}", 500

# Endpoints to download the CSV files
@app.route('/download_hr')
def download_hr():
    hr_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\hr_table.csv"
    try:
        return send_file(hr_csv_path, mimetype='text/csv', as_attachment=True)
    except Exception as e:
        return f"Error downloading HR table: {e}", 500

@app.route('/download_job')
def download_job():
    job_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\job_table.csv"
    try:
        return send_file(job_csv_path, mimetype='text/csv', as_attachment=True)
    except Exception as e:
        return f"Error downloading Job Seeker table: {e}", 500

# Dedicated endpoint for displaying HR table in a colorful, professional page
@app.route('/view_hr_table')
def view_hr_table():
    hr_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\hr_table.csv"
    html_table = render_csv_table(hr_csv_path, "HR Table", download_url="/download_hr")
    return html_table

# Dedicated endpoint for displaying Job Seeker table in a colorful, professional page
@app.route('/view_job_table')
def view_job_table():
    job_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\job_table.csv"
    html_table = render_csv_table(job_csv_path, "Job Seeker Table", download_url="/download_job")
    return html_table

# Endpoints for HR and Job buttons that redirect to the view table pages
@app.route('/hr')
def hr():
    hr_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\hr_table.csv"
    html_table = render_csv_table(hr_csv_path, "HR Table", download_url="/download_hr")
    return html_table

@app.route('/job')
def job():
    job_csv_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\job_table.csv"
    html_table = render_csv_table(job_csv_path, "Job Seeker Table", download_url="/download_job")
    return html_table

def render_csv_table(csv_path, title, download_url):
    """
    Reads a CSV file and returns an HTML page that displays
    the CSV content in a styled table along with navigation
    and download buttons.
    """
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
    except Exception as e:
        return f"<h2>Error loading CSV file: {e}</h2>"

    # Build HTML table rows
    table_rows = ""
    if rows:
        # Create table header
        header = rows[0]
        header_html = "".join([f"<th>{cell.strip()}</th>" for cell in header])
        table_rows += f"<thead><tr>{header_html}</tr></thead>"
        # Create table body
        body_html = ""
        for row in rows[1:]:
            row_html = "".join([f"<td>{cell.strip()}</td>" for cell in row])
            body_html += f"<tr>{row_html}</tr>"
        table_rows += f"<tbody>{body_html}</tbody>"
    else:
        table_rows = "<tr><td>No data available</td></tr>"

    # A professional HTML page design with side-by-side buttons for navigation
    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title}</title>
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
      <style>
        body {{
          margin: 0;
          font-family: 'Roboto', sans-serif;
          background: linear-gradient(135deg, #f093fb, #f5576c);
          color: #333;
          padding: 20px;
        }}
        .container {{
          max-width: 1200px;
          margin: 0 auto;
          background: #fff;
          border-radius: 15px;
          padding: 30px;
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
          animation: fadeIn 1s ease-in-out;
        }}
        h1 {{
          text-align: center;
          color: #f5576c;
          margin-bottom: 30px;
        }}
        .table-container {{
          overflow-x: auto;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 1rem;
          background-color: #f8f8f8;
        }}
        th, td {{
          padding: 15px;
          text-align: left;
          border: 1px solid #ddd;
        }}
        th {{
          background-color: #f5576c;
          color: #fff;
        }}
        tr:nth-child(even) {{
          background-color: #f2f2f2;
        }}
        tr:hover {{
          background-color: #e0e0e0;
        }}
        .btn-container {{
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
          margin-top: 30px;
          flex-wrap: wrap;
        }}
        .btn {{
          display: inline-block;
          text-decoration: none;
          background: #f093fb;
          color: #fff;
          padding: 12px 25px;
          border-radius: 30px;
          transition: background 0.3s ease;
        }}
        .btn:hover {{
          background: #e783f4;
        }}
        @keyframes fadeIn {{
          from {{ opacity: 0; transform: translateY(20px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>{title}</h1>
        <div class="table-container">
          <table>
            {table_rows}
          </table>
        </div>
        <div class="btn-container">
          <a href="/" class="btn">Back to Home</a>
          <a href="/display_tables" class="btn">Display Tables</a>
          <a href="{download_url}" class="btn">Download CSV</a>
        </div>
      </div>
    </body>
    </html>
    """
    return html_page

# Updated endpoint to display a page with buttons to view both tables side by side.
@app.route('/display_tables')
def display_tables():
    html_tables = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>CSV Tables</title>
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
      <style>
        body {
          margin: 0;
          font-family: 'Roboto', sans-serif;
          background: linear-gradient(135deg, #43cea2, #185a9d);
          color: #333;
          padding: 20px;
        }
        .page-container {
          max-width: 1200px;
          margin: 0 auto;
          background: #fff;
          border-radius: 15px;
          padding: 30px;
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
          animation: fadeIn 1s ease-in-out;
        }
        h1 {
          text-align: center;
          color: #185a9d;
          margin-bottom: 30px;
        }
        .btn-container {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
          margin-top: 20px;
          flex-wrap: wrap;
        }
        .btn {
          display: inline-block;
          text-decoration: none;
          background: #43cea2;
          color: #fff;
          padding: 12px 25px;
          border-radius: 30px;
          transition: background 0.3s;
        }
        .btn:hover {
          background: #3ab38b;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      </style>
    </head>
    <body>
      <div class="page-container">
        <h1>Accessible CSV Tables</h1>
        <div class="btn-container">
          <a href="/hr" class="btn">View HR Table</a>
          <a href="/job" class="btn">View Job Seeker Table</a>
        </div>
        <div class="btn-container">
          <a href="/" class="btn">Back to Home</a>
        </div>
      </div>
    </body>
    </html>
    """
    return html_tables
if __name__ == '__main__':
    app.run(debug=True)

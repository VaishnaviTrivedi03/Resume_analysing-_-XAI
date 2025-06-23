from flask import Flask, request, render_template_string
import os
import subprocess

app = Flask(__name__)

# Set the upload folder to your desired directory
UPLOAD_FOLDER = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to index.html in the same directory as app.py
INDEX_HTML_PATH = os.path.join(os.getcwd(), 'index.html')

@app.route('/')
def index():
    # Load and render index.html
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

    # Validate that the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return "Only PDF files are allowed!", 400

    if file:
        # Save the file in the specified upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return f"File '{file.filename}' uploaded successfully to {UPLOAD_FOLDER}!", 200

@app.route('/run', methods=['POST'])
def run_script():
    # Path to the Python script
    script_path = r"C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\try.py"

    # Path to the virtual environment activation script
    activate_script = r".\myenv\Scripts\Activate"

    try:
        # Construct the command to activate the virtual environment and run the Python script
        command = f'"{activate_script}" && python "{script_path}"'
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,  # Use shell to execute the command as a single string
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout  # Capture the script's output
    except subprocess.CalledProcessError as e:
        output = f"Error while running the script:\n{e.stderr}"

    # Render the output on a colorful, professional HTML page
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Script Output</title>
      <style>
        body {{
          margin: 0;
          padding: 20px;
          font-family: Arial, sans-serif;
          background: linear-gradient(to right, #ff7e5f, #feb47b);
          color: #333;
          text-align: center;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .container {{
          background: #fff;
          padding: 30px 40px;
          border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          max-width: 800px;
          width: 90%;
        }}
        h2 {{
          color: #6a11cb;
          margin-bottom: 20px;
        }}
        pre {{
          text-align: left;
          background: #f4f4f4;
          padding: 20px;
          border-radius: 5px;
          overflow-x: auto;
          font-size: 0.95rem;
        }}
        a.button {{
          display: inline-block;
          margin-top: 20px;
          text-decoration: none;
          color: #fff;
          background: #2575fc;
          padding: 10px 20px;
          border-radius: 5px;
          transition: background 0.3s ease;
        }}
        a.button:hover {{
          background: #1a5bb8;
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

@app.route('/view_table')
def view_table():
    # Sample data for the table
    data = [
        {'Name': 'Alice', 'Role': 'Developer', 'Location': 'New York'},
        {'Name': 'Bob', 'Role': 'Designer', 'Location': 'San Francisco'},
        {'Name': 'Charlie', 'Role': 'Manager', 'Location': 'London'},
        {'Name': 'Diana', 'Role': 'Analyst', 'Location': 'Berlin'},
    ]
    # Build table rows dynamically
    rows = []
    for row in data:
        rows.append(f"<tr><td>{row['Name']}</td><td>{row['Role']}</td><td>{row['Location']}</td></tr>")
    
    table_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Data Table</title>
      <style>
        body {{
           font-family: Arial, sans-serif;
           background: linear-gradient(to right, #ff7e5f, #feb47b);
           padding: 20px;
           color: #333;
           margin: 0;
        }}
        .container {{
           max-width: 800px;
           margin: 40px auto;
           background: #fff;
           border-radius: 10px;
           padding: 20px;
           box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        h2 {{
           color: #6a11cb;
           margin-bottom: 20px;
        }}
        table {{
           width: 100%;
           border-collapse: collapse;
        }}
        th, td {{
           padding: 12px 15px;
           border: 1px solid #ddd;
           text-align: left;
        }}
        th {{
           background-color: #6a11cb;
           color: #fff;
        }}
        tr:nth-child(even) {{
           background-color: #f3f3f3;
        }}
        tr:hover {{
           background-color: #e9e9e9;
        }}
        a.button {{
          display: inline-block;
          margin-top: 20px;
          text-decoration: none;
          color: #fff;
          background: #2575fc;
          padding: 10px 20px;
          border-radius: 5px;
          transition: background 0.3s ease;
        }}
        a.button:hover {{
          background: #1a5bb8;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h2>Employee Data</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Role</th>
              <th>Location</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        <a class="button" href="/">Back to Home</a>
      </div>
    </body>
    </html>
    """
    return table_html

if __name__ == '__main__':
    app.run(debug=True)

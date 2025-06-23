from flask import Flask, render_template_string, send_file
import pandas as pd

app = Flask(__name__)

# File paths for the tables
APPLICANT_TABLE_PATH = "C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\applicanttry_table.csv"
HR_TABLE_PATH = "C:\Users\Vaishnavi K Trivedi\OneDrive\Desktop\Research AI\hrtry_table.csv"

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload and Run</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background: linear-gradient(to right, #ff7e5f, #feb47b);
                color: #fff;
                padding: 20px;
            }
            .container {
                margin: auto;
                max-width: 800px;
                padding: 20px;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            a {
                display: inline-block;
                margin: 10px;
                padding: 10px 20px;
                color: #fff;
                text-decoration: none;
                border-radius: 5px;
                background: linear-gradient(to right, #6a11cb, #2575fc);
            }
            a:hover {
                background: #1a5bb8;
            }
        </style>
    </head>
    <body>
        <h1>Upload and Run Application</h1>
        <div class="container">
            <a href="/view_table/applicant">View Applicant Table</a>
            <a href="/view_table/hr">View HR Table</a>
        </div>
    </body>
    </html>
    """

@app.route('/view_table/<table_type>')
def view_table(table_type):
    if table_type == "applicant":
        table_path = APPLICANT_TABLE_PATH
        title = "Applicant Table"
    elif table_type == "hr":
        table_path = HR_TABLE_PATH
        title = "HR Table"
    else:
        return "Invalid table type.", 400

    try:
        df = pd.read_csv(table_path)
        table_html = df.to_html(classes='styled-table', index=False, justify='center')
    except Exception as e:
        return f"Error loading table: {e}", 500

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                background: linear-gradient(to right, #ff7e5f, #feb47b);
                color: #fff;
                padding: 20px;
            }}
            .container {{
                margin: auto;
                max-width: 1000px;
                padding: 20px;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 1rem;
                text-align: left;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #6a11cb;
                color: #fff;
            }}
            tr:nth-child(even) {{
                background-color: #f3f3f3;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            a {{
                margin-top: 20px;
                display: inline-block;
                padding: 10px 20px;
                color: #fff;
                text-decoration: none;
                background: linear-gradient(to right, #6a11cb, #2575fc);
                border-radius: 5px;
            }}
            a:hover {{
                background: #1a5bb8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            {table_html}
            <a href="/">Back to Home</a>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)

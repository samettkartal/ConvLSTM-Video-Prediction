import markdown
from xhtml2pdf import pisa
import os
import re

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Preprocess Images: 
    # Markdown has "file:///c:/...". xhtml2pdf prefers local paths "c:/..." or relative.
    # Let's strip "file:///" to make them absolute OS paths.
    text = text.replace('file:///', '')
    # Also handle forward slashes if needed, but python usually handles them fine.

    # 3. Convert to HTML
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # 4. Add CSS for styling
    css = """
    <style>
        @page {
            size: A4 landscape;
            margin: 2cm;
            @frame footer_frame {
                -pdf-frame-content: footer_content;
                bottom: 1cm;
                margin-left: 2cm;
                margin-right: 2cm;
                height: 1cm;
            }
        }
        body {
            font-family: Helvetica, sans-serif;
            font-size: 14pt;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 32pt;
            margin-bottom: 50px;
        }
        h2 {
            color: #2980b9;
            border-bottom: 2px solid #2980b9;
            padding-bottom: 10px;
            page-break-before: always;
        }
        p {
            line-height: 1.5;
        }
        img {
            max-width: 90%;
            height: auto;
            display: block;
            margin: 20px auto;
        }
        blockquote {
            background-color: #f9f9f9;
            border-left: 10px solid #ccc;
            margin: 1.5em 10px;
            padding: 0.5em 10px;
            font-style: italic;
            font-size: 12pt;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #2980b9;
            color: white;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            font-family: monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            white-space: pre-wrap;
        }
    </style>
    """

    full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"

    # 5. Write PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,
            dest=result_file
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created {output_pdf}")

if __name__ == "__main__":
    # Source is the artifact path
    source = r"C:\Users\samet\.gemini\antigravity\brain\e29527e0-a308-4e04-b2d6-6ff541b7ddc1\presentation_draft.md"
    project_root = r"c:\Users\samet\Desktop\YAZILIM\Python\zamanserileriyeni"
    output = os.path.join(project_root, "presentation.pdf")
    
    convert_md_to_pdf(source, output)

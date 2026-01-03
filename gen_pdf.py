from convert_to_pdf import convert_md_to_pdf
import os

source_md = r"c:\Users\samet\Desktop\YAZILIM\Python\zamanserileriyeni\report_temp.md"
output_pdf = r"c:\Users\samet\Desktop\YAZILIM\Python\zamanserileriyeni\Proje_Raporu.pdf"

print(f"Converting {source_md} to {output_pdf}...")
try:
    convert_md_to_pdf(source_md, output_pdf)
    print("Done.")
except Exception as e:
    print(f"Error: {e}")

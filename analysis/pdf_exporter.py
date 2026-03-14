"""
pdf_exporter.py
===============
Utility to generate a formatted PDF report summarizing the RAG Data Preprocessing 
Pipeline results and dataset analysis charts.
"""

from fpdf import FPDF
from pathlib import Path

class PDFReportGenerator(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        # Title
        self.cell(0, 10, 'Stratify Preprocessing Pipeline Report', border=False, align='C')
        self.ln(20)

    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, num, title):
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, f'Section {num} : {title}', 0, 1, 'L', fill=False)
        self.ln(4)

    def chapter_body(self, stat_dict):
        self.set_font('helvetica', '', 12)
        
        for key, value in stat_dict.items():
            formatted_key = key.replace('_', ' ').title()
            # If value is numeric and large, format it with commas
            if isinstance(value, (int, float)) and value > 999:
                 formatted_value = f"{value:,}"
            else:
                 formatted_value = str(value)
            
            self.cell(80, 10, formatted_key, border=0)
            self.cell(80, 10, formatted_value, border=0, ln=1)
        
        self.ln(10)

    def chapter_paragraph(self, text):
        """Draws a wrapped text block for descriptions/summaries."""
        self.set_font('helvetica', '', 11)
        # multi_cell automatically handles line breaks and page breaks.
        self.multi_cell(0, 7, text)
        self.ln(10)

def generate_pdf_report(pipeline_stats: dict, analysis_report: dict, rag_summary_text: str = None, ai_stat_text: str = None) -> bytes:
    """
    Generates a PDF from the provided JSON dictionaries and image paths.
    Returns the raw byte string to be offered as a download.
    """
    pdf = PDFReportGenerator()
    pdf.add_page()
    
    # 1. Pipeline Summary Section
    pdf.chapter_title(1, 'Pipeline Core Statistics')
    
    core_stats = {
        "Total Files Processed": pipeline_stats.get("total_files", 0),
        "Total Raw Characters": pipeline_stats.get("total_raw_chars", 0),
        "Chunks Before Filter": pipeline_stats.get("chunks_before_filter", 0),
        "Chunks After QA Filter": pipeline_stats.get("chunks_after_filter", 0),
        "Filter Drop Rate (%)": pipeline_stats.get("filter_rate_pct", 0),
        "Dropped (Too Short)": pipeline_stats.get("dropped_too_short", 0),
        "Dropped (Low Quality)": pipeline_stats.get("dropped_low_quality", 0),
        "Dropped (Duplicates)": pipeline_stats.get("dropped_duplicates", 0),
    }
    pdf.chapter_body(core_stats)
    
    # Optional 1.5: Executive Document Summary (AI Generated)
    next_chapter = 2
    if rag_summary_text or ai_stat_text:
        pdf.add_page()
        
        if rag_summary_text:
             pdf.chapter_title(next_chapter, 'Executive Document Summary (AI Generated)')
             pdf.chapter_paragraph(rag_summary_text)
             next_chapter += 1
             
        if ai_stat_text:
             pdf.chapter_title(next_chapter, 'Statistical Analysis Summary (AI Generated)')
             pdf.chapter_paragraph(ai_stat_text)
             next_chapter += 1
    
    # 3. Add Charts Section
    chart_paths = analysis_report.get("chart_paths", {})
    if chart_paths:
        if not (rag_summary_text or ai_stat_text): # we already added a page if summary existed
             pdf.add_page()
        else:
             # Ensure there is enough room for charts, otherwise break
             if pdf.get_y() > 150:
                  pdf.add_page()
             
        pdf.chapter_title(next_chapter, 'Dataset Distributions (Charts)')
        
        y_cursor = pdf.get_y()
        chart_count = 0
        for fig_id, path_str in chart_paths.items():
            img_path = Path(path_str)
            if img_path.exists():
                # Arrange 2 images per row
                x = 10 if chart_count % 2 == 0 else 110
                y = pdf.get_y()
                
                # If moving to the second on a line, dont advance Y.
                # If moving to the next line, bump Y down
                if chart_count > 0 and chart_count % 2 == 0:
                     pdf.set_y(y + 90)  # Move down a chunk
                     y = pdf.get_y()
                     
                     # Page break check
                     if y > 200:
                          pdf.add_page()
                          y = pdf.get_y()
                     
                
                # Image dimensions approx 90x70 mm
                pdf.image(str(img_path), x=x, y=y, w=90)
                
                # Captions
                pdf.set_xy(x, y + 75)
                pdf.set_font('helvetica', 'I', 10)
                pdf.cell(90, 10, fig_id, align='C')
                
                # Reset Y to baseline block position
                pdf.set_y(y) 
                
                chart_count += 1
                
    
    # Return bytes directly
    return bytes(pdf.output())


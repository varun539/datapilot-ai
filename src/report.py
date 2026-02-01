# src/report.py

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import datetime


def generate_pdf_report(model_card, business_insights, output_path="DataPilot_Report.pdf"):
    """
    Generates a professional ML PDF report
    """

    styles = getSampleStyleSheet()
    story = []

    # ==========================
    # TITLE
    # ==========================
    story.append(Paragraph(
        "<b>DataPilot AI — Model Report</b>",
        styles["Title"]
    ))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.3 * inch))

    # ==========================
    # MODEL OVERVIEW
    # ==========================
    story.append(Paragraph("<b>Model Overview</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        f"""
        <b>Model:</b> {model_card.get('model')}<br/>
        <b>Problem Type:</b> {model_card.get('problem')}<br/>
        <b>Training Mode:</b> {model_card.get('mode')}<br/>
        <b>Target Variable:</b> {model_card.get('target')}
        """,
        styles["Normal"]
    ))

    story.append(Spacer(1, 0.3 * inch))

    # ==========================
    # DATASET SUMMARY
    # ==========================
    story.append(Paragraph("<b>Dataset Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        f"""
        <b>Total Rows:</b> {model_card.get('rows')}<br/>
        <b>Features Used:</b> {model_card.get('features')}
        """,
        styles["Normal"]
    ))

    story.append(Spacer(1, 0.3 * inch))

    # ==========================
    # PERFORMANCE
    # ==========================
    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    perf = model_card.get("performance", {})
    for k, v in perf.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

    story.append(Spacer(1, 0.3 * inch))

    # ==========================
    # BUSINESS IMPACT
    # ==========================
    story.append(Paragraph("<b>Business Impact Insights</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    if business_insights:
        for insight in business_insights:
            story.append(Paragraph(f"- {insight}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("No significant business insights detected.", styles["Normal"]))

    story.append(Spacer(1, 0.3 * inch))

    # ==========================
    # LIMITATIONS
    # ==========================
    story.append(Paragraph("<b>Limitations & Risk Notice</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        """
        • Model performance depends on historical data quality.<br/>
        • Predictions may be unreliable for unseen or extreme values.<br/>
        • This model should support decisions, not replace human judgment.
        """,
        styles["Normal"]
    ))

    # ==========================
    # BUILD PDF
    # ==========================
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    doc.build(story)

    return output_path

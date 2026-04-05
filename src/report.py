# # src/report.py

# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     ListFlowable,
#     ListItem
# )
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.units import inch
# import datetime
# import re


# def _clean_text(text: str) -> str:
#     """
#     Remove emojis and markdown for PDF safety
#     """
#     text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove **
#     text = re.sub(r"[^\x00-\x7F]+", "", text)     # remove emojis
#     return text


# def generate_pdf_report(
#     model_card,
#     business_insights,
#     output_path="DataPilot_Report.pdf"
# ):
#     """
#     Generates a professional, business-safe ML PDF report
#     """

#     styles = getSampleStyleSheet()
#     story = []

#     # ==========================
#     # TITLE
#     # ==========================
#     story.append(Paragraph(
#         "DataPilot AI — Model Report",
#         styles["Title"]
#     ))
#     story.append(Spacer(1, 0.25 * inch))

#     story.append(Paragraph(
#         f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
#         styles["Normal"]
#     ))
#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # MODEL OVERVIEW
#     # ==========================
#     story.append(Paragraph("Model Overview", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     overview_text = f"""
#     <b>Model:</b> {model_card.get('model')}<br/>
#     <b>Problem Type:</b> {model_card.get('problem')}<br/>
#     <b>Training Mode:</b> {model_card.get('mode')}<br/>
#     <b>Target Variable:</b> {model_card.get('target')}
#     """

#     story.append(Paragraph(overview_text, styles["Normal"]))
#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # DATASET SUMMARY
#     # ==========================
#     story.append(Paragraph("Dataset Summary", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     dataset_text = f"""
#     <b>Total Rows:</b> {model_card.get('rows')}<br/>
#     <b>Features Used:</b> {model_card.get('features')}
#     """

#     story.append(Paragraph(dataset_text, styles["Normal"]))
#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # PERFORMANCE
#     # ==========================
#     story.append(Paragraph("Model Performance", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     perf = model_card.get("performance", {})
#     for k, v in perf.items():
#         story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # BUSINESS IMPACT
#     # ==========================
#     story.append(Paragraph("Business Impact Insights", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     if business_insights:
#         clean_items = [
#             ListItem(
#                 Paragraph(_clean_text(insight), styles["Normal"])
#             )
#             for insight in business_insights
#         ]

#         story.append(ListFlowable(
#             clean_items,
#             bulletType="bullet",
#             start="circle"
#         ))
#     else:
#         story.append(Paragraph(
#             "No significant business insights detected.",
#             styles["Normal"]
#         ))

#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # INTERPRETATION NOTICE
#     # ==========================
#     story.append(Paragraph("Interpretation Notice", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     interpretation_points = [
#         "Feature importance reflects statistical association, not causation.",
#         "Observed relationships may be influenced by external economic or behavioral factors.",
#         "Model insights should be combined with domain expertise."
#     ]

#     story.append(ListFlowable(
#         [ListItem(Paragraph(p, styles["Normal"])) for p in interpretation_points],
#         bulletType="bullet"
#     ))

#     story.append(Spacer(1, 0.25 * inch))

#     # ==========================
#     # LIMITATIONS
#     # ==========================
#     story.append(Paragraph("Limitations & Risk Notice", styles["Heading2"]))
#     story.append(Spacer(1, 0.1 * inch))

#     limitation_points = [
#         "Model performance depends on historical data quality.",
#         "Predictions may be unreliable for unseen or extreme values.",
#         "This model is intended for decision support, not autonomous decisions."
#     ]

#     story.append(ListFlowable(
#         [ListItem(Paragraph(p, styles["Normal"])) for p in limitation_points],
#         bulletType="bullet"
#     ))

#     story.append(Spacer(1, 0.35 * inch))

#     # ==========================
#     # FOOTER
#     # ==========================
#     story.append(Paragraph(
#         "Generated by DataPilot AI<br/>Built by Varun B",
#         styles["Normal"]
#     ))

#     # ==========================
#     # BUILD PDF
#     # ==========================
#     doc = SimpleDocTemplate(
#         output_path,
#         pagesize=A4,
#         rightMargin=36,
#         leftMargin=36,
#         topMargin=36,
#         bottomMargin=36
#     )


#     doc.build(story)

#     return output_path
















from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import datetime


def generate_pdf_report(model_card, insights, output_path="report.pdf"):

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("DataPilot AI Report", styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Generated: {datetime.datetime.now()}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Model Info
    story.append(Paragraph("Model Overview", styles["Heading2"]))
    story.append(Paragraph(f"Model: {model_card.get('model')}", styles["Normal"]))
    story.append(Paragraph(f"Target: {model_card.get('target')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Performance
    story.append(Paragraph("Performance", styles["Heading2"]))
    for k, v in model_card.get("performance", {}).items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Features
    story.append(Paragraph("Top Features", styles["Heading2"]))
    for f, v in model_card.get("top_features", []):
        story.append(Paragraph(f"{f} ({round(v,4)})", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Insights
    story.append(Paragraph("Business Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(i, styles["Normal"]))

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(story)

    return output_path

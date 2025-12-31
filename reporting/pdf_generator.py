from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os


def format_timestamps(ts_list, max_lines=12):
    """
    Shows first N start times only.
    Prevents ReportLab table overflow.
    """
    if not ts_list:
        return "-"

    lines = []
    for t in ts_list[:max_lines]:
        if isinstance(t, dict) and "start" in t:
            lines.append(t["start"])

    remaining = len(ts_list) - max_lines
    if remaining > 0:
        lines.append(f"+ {remaining} more movements")

    return "<br/>".join(lines)


def generate_participant_pdf(
    output_dir,
    participant_id,
    participant_report,
    role,
    index
):
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(
        output_dir, f"{participant_id}_{role}_report.pdf"
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    wrap_style = styles["Normal"]
    story = []

    # -------------------------
    # HEADER
    # -------------------------
    story.append(Paragraph(
        "<b>Meditation Proctoring Report</b>", styles["Title"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Participant ID:</b> {participant_id}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Position:</b> {role}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Index:</b> {index}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Overall Status:</b> {participant_report['overall_status']}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 16))

    # -------------------------
    # MOVEMENT TABLE
    # -------------------------
    table_data = [
        ["Movement", "Count", "Allowed", "Status", "Timestamps (Start Time)"]
    ]

    for part in ["neck", "arm", "leg"]:
        p = participant_report[part]

        ts_text = format_timestamps(p.get("timestamps", []))
        ts_para = Paragraph(ts_text, wrap_style)

        table_data.append([
            part.upper(),
            str(p["count"]),
            str(p["allowed"]),
            p["status"],
            ts_para
        ])

    table = Table(
        table_data,
        colWidths=[70, 50, 60, 60, 260],  # slightly wider timestamp column
        repeatRows=1                      # header repeats on page break
    )

    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    story.append(Spacer(1, 14))

    # -------------------------
    # REMARKS
    # -------------------------
    if participant_report["remarks"]:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Remarks:</b>", styles["Heading3"]))
        story.append(Spacer(1, 6))
        for r in participant_report["remarks"]:
            story.append(Paragraph(f"- {r}", styles["Normal"]))


    doc.build(story)

    return file_path

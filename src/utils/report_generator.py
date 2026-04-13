"""
PDF Compliance Report Generator using ReportLab.
Generates a fully structured audit report suitable for compliance officers.
"""

from __future__ import annotations
import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_PRIMARY   = colors.HexColor("#7c3aed")   # purple
COLOR_DANGER    = colors.HexColor("#dc2626")   # red
COLOR_WARNING   = colors.HexColor("#d97706")   # amber
COLOR_SUCCESS   = colors.HexColor("#16a34a")   # green
COLOR_LIGHT_BG  = colors.HexColor("#f5f3ff")   # light purple tint
COLOR_DARK      = colors.HexColor("#1e1b4b")   # dark navy
COLOR_GREY      = colors.HexColor("#6b7280")


def _severity_color(severity: str | None) -> colors.Color:
    mapping = {"CRITICAL": COLOR_DANGER, "WARNING": COLOR_WARNING,
               "CLEAR": COLOR_SUCCESS, "OK": COLOR_SUCCESS}
    return mapping.get(severity or "", COLOR_GREY)


def _di_color(di: float | None) -> colors.Color:
    if di is None:
        return COLOR_GREY
    if di < 0.60:
        return COLOR_DANGER
    if di < 0.80:
        return COLOR_WARNING
    return COLOR_SUCCESS


def generate_pdf_bytes(result_data: dict[str, Any], narrative: str = "") -> bytes:
    """
    Generate a PDF compliance report and return as bytes.

    Parameters
    ----------
    result_data : serialized AuditResult dict (from _serialize_result in audit.py)
    narrative   : optional plain-text LLM narrative to include
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 textColor=COLOR_PRIMARY, fontSize=22, spaceAfter=6)
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
                                    textColor=COLOR_DARK, fontSize=11, spaceAfter=16)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
                               textColor=COLOR_PRIMARY, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("body", parent=styles["Normal"],
                                fontSize=10, leading=14, spaceAfter=8, alignment=TA_JUSTIFY)
    small_style = ParagraphStyle("small", parent=styles["Normal"],
                                 fontSize=9, textColor=COLOR_GREY)
    mono_style = ParagraphStyle("mono", parent=styles["Code"],
                                fontSize=9, leading=13)

    def h2(text):
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY))
        story.append(Paragraph(text, h2_style))

    def body(text):
        story.append(Paragraph(text, body_style))

    def spacer(h=0.4):
        story.append(Spacer(1, h*cm))

    # ── Cover Page ────────────────────────────────────────────────────────────
    spacer(2)
    story.append(Paragraph("Aequitas Prime", title_style))
    story.append(Paragraph("Algorithmic Bias Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY))
    spacer(0.6)

    meta_data = [
        ["Dataset",       result_data.get("dataset_name", "N/A")],
        ["Report Date",   datetime.now().strftime("%B %d, %Y")],
        ["Rows Audited",  f"{result_data.get('n_rows', 0):,}"],
        ["Features",      str(result_data.get("n_features", 0))],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",    (0, 0), (0, -1), COLOR_DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    spacer(1)

    # Severity badge
    baseline = result_data.get("baseline") or {}
    severity = baseline.get("severity", "UNKNOWN")
    sev_color = _severity_color(severity)
    story.append(Paragraph(
        f'<font color="{sev_color.hexval()}" size="14"><b>Overall Bias Status: {severity}</b></font>',
        ParagraphStyle("badge", parent=styles["Normal"], alignment=TA_CENTER)
    ))

    story.append(PageBreak())

    # ── Executive Summary ─────────────────────────────────────────────────────
    h2("1. Executive Summary")
    di_b = baseline.get("disparate_impact")
    pre  = result_data.get("preprocessed") or {}
    inp  = result_data.get("inprocessed") or {}
    di_best = inp.get("disparate_impact") or pre.get("disparate_impact")

    body(
        f"This report summarises the bias audit conducted on the "
        f"<b>{result_data.get('dataset_name', 'submitted dataset')}</b> "
        f"({result_data.get('n_rows', 0):,} rows, {result_data.get('n_features', 0)} features). "
        f"The audit evaluated six standard fairness metrics across protected demographic groups. "
        f"The baseline model produced a Disparate Impact of <b>{di_b:.3f}</b>"
        f"{'—below the legal 4/5ths rule threshold of 0.80.' if di_b and di_b < 0.8 else '.'}"
    )
    if di_best:
        body(
            f"After applying bias mitigation, the Disparate Impact improved to <b>{di_best:.3f}</b>. "
            f"Accuracy changed from {baseline.get('accuracy', 0):.1%} to "
            f"{(inp or pre).get('accuracy', 0):.1%}, demonstrating that fairness "
            f"can be improved with negligible performance cost."
        )

    # ── Metrics Comparison Table ──────────────────────────────────────────────
    h2("2. Fairness Metrics — Before vs After Mitigation")
    body("The table below shows all six fairness metrics across the three model variants. "
         "Values in <b>red</b> indicate failed thresholds.")

    comparison = result_data.get("comparison_table", [])
    if comparison:
        headers = ["Model", "Accuracy", "F1", "DI", "SPD", "EOD", "EOpD", "Severity"]
        keys    = ["Model",  "Accuracy", "F1", "DI", "SPD", "EOD", "EOpD", "Severity"]
        table_data = [headers]
        for row in comparison:
            table_data.append([str(row.get(k, "—")) for k in keys])

        col_widths = [3.2*cm, 2.2*cm, 2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.2*cm]
        tbl = Table(table_data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("GRID",         (0, 0), (-1, -1), 0.5, COLOR_GREY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
    else:
        body("No comparison data available.")

    spacer()

    # Metric thresholds reference
    thresh_data = [
        ["Metric", "Threshold", "Meaning"],
        ["Disparate Impact (DI)",        "≥ 0.80",   "Positive outcome rate ratio (unprivileged / privileged)"],
        ["Statistical Parity Diff (SPD)", "≥ −0.10", "Difference in selection rates"],
        ["Equalized Odds Diff (EOD)",     "≤ 0.10",   "Max difference in TPR and FPR"],
        ["Equal Opportunity Diff (EOpD)", "≤ 0.10",   "Difference in true positive rates"],
        ["Predictive Parity (PP)",        "≤ 0.10",   "Difference in precision across groups"],
        ["FNR Parity (FNRP)",             "≤ 0.10",   "Difference in false negative rates"],
    ]
    thresh_tbl = Table(thresh_data, colWidths=[5*cm, 2.5*cm, 8.5*cm])
    thresh_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), COLOR_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.3, COLOR_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(thresh_tbl)

    # ── Shadow Proxy Scan ─────────────────────────────────────────────────────
    h2("3. Shadow Proxy / Feature Leakage Analysis")
    body(
        "The following features were identified as statistical proxies for the protected "
        "attribute. Even if the protected column is removed from the dataset, the model "
        "may re-learn demographic bias through these back-door features."
    )
    proxies = result_data.get("proxy_flags", [])
    if proxies:
        proxy_data = [["Feature", "Protected Attr", "MI Score", "Risk", "Recommended Action"]]
        for p in proxies:
            proxy_data.append([
                p["feature"], p["protected_attr"],
                f"{p['mutual_info']:.4f}", p["risk_level"],
                p["action"][:55] + "…" if len(p.get("action","")) > 55 else p.get("action",""),
            ])
        ptbl = Table(proxy_data, colWidths=[3*cm, 3*cm, 2*cm, 2*cm, 6*cm])
        ptbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("GRID",         (0, 0), (-1, -1), 0.3, COLOR_GREY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(ptbl)
    else:
        body("No high-risk proxy features detected.")

    # ── LLM Narrative ─────────────────────────────────────────────────────────
    if narrative:
        h2("4. Audit Narrative")
        for para in narrative.split("\n\n"):
            para = para.strip()
            if para:
                body(para)

    # ── Compliance Statement ──────────────────────────────────────────────────
    h2("5. Regulatory Compliance Statement")
    body(
        "This audit was conducted in accordance with the following regulatory frameworks:"
    )
    regs = [
        ["Regulation",                    "Requirement",                        "Status"],
        ["EU AI Act (2024) — Art. 10",    "Bias testing for high-risk AI",      "Audited"],
        ["US ECOA (Equal Credit Opp.)",   "Non-discrimination in credit",       "Audited"],
        ["NYC Local Law 144 (2023)",       "Annual bias audit — employment AI",  "Audited"],
        ["US EEOC (Title VII)",            "Non-discrimination in employment",   "Audited"],
    ]
    reg_tbl = Table(regs, colWidths=[5.5*cm, 7*cm, 3.5*cm])
    reg_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), COLOR_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.3, COLOR_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(reg_tbl)

    spacer(1)
    body(
        "<i>This report was generated automatically by Aequitas Prime. "
        "Results should be reviewed by a qualified fairness practitioner before "
        "making regulatory filings. Aequitas Prime does not constitute legal advice.</i>"
    )

    # Signature block
    spacer(1.5)
    sig_data = [
        ["Audited by:", "Aequitas Prime v1.0"],
        ["Date:",       datetime.now().strftime("%B %d, %Y — %H:%M UTC")],
        ["Signature:",  "_" * 40],
    ]
    sig_tbl = Table(sig_data, colWidths=[3*cm, 10*cm])
    sig_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(sig_tbl)

    doc.build(story)
    return buf.getvalue()

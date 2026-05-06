import io
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from models import state

router = APIRouter()

_METHOD_LABELS = {
    "smote":    "SMOTE (Over-sampling)",
    "nearmiss": "NearMiss (Under-sampling)",
    "combined": "Combined (SMOTE + NearMiss)",
}


def _get_completed_run(run_id: str):
    run = state.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail="Run is not completed yet.")
    return run


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _build_summary_pdf(run_id: str, run: dict) -> bytes:
    result = run["result"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    dataset_name = "Unknown"
    dataset_id = run.get("dataset_id", "")
    if dataset_id:
        ds = state.get_dataset(dataset_id)
        if ds:
            dataset_name = ds.get("filename", "Unknown")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    base = getSampleStyleSheet()

    def _s(base_name, **kw):
        s = ParagraphStyle(base_name + "_ik", parent=base[base_name])
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    title_s    = _s("Title",    fontSize=20, spaceAfter=4)
    subtitle_s = _s("Normal",   fontSize=10, spaceAfter=2,
                    textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER)
    section_s  = _s("Heading2", fontSize=13, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1d4ed8"))
    footer_s   = _s("Normal",   fontSize=8, alignment=TA_CENTER,
                    textColor=colors.HexColor("#9ca3af"))

    BLUE   = colors.HexColor("#1d4ed8")
    GREEN  = colors.HexColor("#15803d")
    RED    = colors.HexColor("#dc2626")
    GRAY_L = colors.HexColor("#f3f4f6")
    GRAY_B = colors.HexColor("#d1d5db")

    def _tbl_style(extra=None):
        cmds = [
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_L]),
            ("GRID",           (0, 0), (-1, -1), 0.4, GRAY_B),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
            ("BACKGROUND",     (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        if extra:
            cmds.extend(extra)
        return TableStyle(cmds)

    story = []

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("ImbalanceKit — Run Summary", title_s))
    story.append(Paragraph(
        "Capstone Project 26-1-D-3 · Braude College of Engineering", subtitle_s
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE))
    story.append(Spacer(1, 0.4 * cm))

    # ── Run configuration ─────────────────────────────────────────────────────
    story.append(Paragraph("Run Configuration", section_s))

    method_label = _METHOD_LABELS.get(result["method"], result["method"].upper())
    params       = result.get("params", {})
    params_str   = ", ".join(f"{k}: {v}" for k, v in params.items()) or "—"

    config_rows = [
        ["Field",              "Value"],
        ["Project",            "ImbalanceKit"],
        ["Dataset",            dataset_name],
        ["Label Column",       result.get("label_col", "—")],
        ["Balancing Method",   method_label],
        ["Parameters",         params_str],
        ["Run ID",             run_id],
        ["Elapsed Time",       f"{result.get('elapsed_seconds', '—')} s"],
        ["Generated At",       now],
    ]
    t = Table(config_rows, colWidths=[5 * cm, 12 * cm])
    t.setStyle(_tbl_style())
    story.append(t)

    # ── Imbalance Ratio ───────────────────────────────────────────────────────
    story.append(Paragraph("Imbalance Ratio (IR)", section_s))

    ir_b     = result["ir_before"]
    ir_a     = result["ir_after"]
    ir_delta = round(ir_b - ir_a, 4)
    ir_pct   = round((ir_delta / ir_b) * 100, 1) if ir_b else 0

    ir_rows = [
        ["Metric",           "Before",      "After",       "Improvement"],
        ["Imbalance Ratio",  f"{ir_b:.4f}", f"{ir_a:.4f}", f"↓ {ir_delta:.4f}  ({ir_pct}%)"],
    ]
    t = Table(ir_rows, colWidths=[5.5 * cm, 3.5 * cm, 3.5 * cm, 4.5 * cm])
    t.setStyle(_tbl_style([
        ("TEXTCOLOR", (3, 1), (3, 1), GREEN),
        ("FONTNAME",  (3, 1), (3, 1), "Helvetica-Bold"),
    ]))
    story.append(t)

    # ── Class Distribution ────────────────────────────────────────────────────
    story.append(Paragraph("Class Distribution", section_s))

    dist_b      = result["class_distribution_before"]
    dist_a      = result["class_distribution_after"]
    all_classes = sorted(set(list(dist_b.keys()) + list(dist_a.keys())))
    total_b     = sum(dist_b.values()) or 1
    total_a     = sum(dist_a.values()) or 1

    dist_rows = [["Class", "Before (n)", "Before (%)", "After (n)", "After (%)", "Change"]]
    delta_cmds = []
    for row_i, cls in enumerate(all_classes, start=1):
        b_n = dist_b.get(cls, 0)
        a_n = dist_a.get(cls, 0)
        d   = a_n - b_n
        d_str = f"+{d}" if d > 0 else str(d)
        dist_rows.append([
            str(cls),
            str(b_n), f"{b_n / total_b * 100:.1f}%",
            str(a_n), f"{a_n / total_a * 100:.1f}%",
            d_str,
        ])
        delta_cmds.append(("TEXTCOLOR", (5, row_i), (5, row_i), GREEN if d > 0 else RED if d < 0 else colors.black))

    # Totals row
    total_d = total_a - total_b
    total_d_str = f"+{total_d}" if total_d > 0 else str(total_d)
    dist_rows.append(["Total", str(total_b), "100%", str(total_a), "100%", total_d_str])
    last = len(dist_rows) - 1
    delta_cmds += [
        ("FONTNAME",   (0, last), (-1, last), "Helvetica-Bold"),
        ("BACKGROUND", (0, last), (-1, last), GRAY_L),
        ("TEXTCOLOR",  (5, last), (5, last), GREEN if total_d > 0 else RED if total_d < 0 else colors.black),
    ]

    t = Table(dist_rows, colWidths=[4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm])
    t.setStyle(_tbl_style(delta_cmds))
    story.append(t)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=GRAY_B))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Generated by ImbalanceKit · {now} · "
        "Authors: Niv Oren &amp; Daniel Levovsky · "
        "Advisor: Dr. Avital Shulner Tal",
        footer_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/export/dataset/{run_id}")
def export_dataset(run_id: str):
    run    = _get_completed_run(run_id)
    result = run["result"]
    df     = result["balanced_df"]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"balanced_dataset_{run_id[:8]}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/summary/{run_id}")
def export_summary(run_id: str):
    run = _get_completed_run(run_id)

    pdf_bytes = _build_summary_pdf(run_id, run)
    filename  = f"run_summary_{run_id[:8]}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/log/{run_id}")
def export_log(run_id: str):
    run    = _get_completed_run(run_id)
    log_df = run["result"].get("log_df")
    if log_df is None:
        raise HTTPException(status_code=404, detail="No change log available for this run.")

    buf = io.StringIO()
    log_df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"changes_log_{run_id[:8]}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/all/{run_id}")
def export_all(run_id: str):
    run    = _get_completed_run(run_id)
    result = run["result"]

    csv_buf = io.StringIO()
    result["balanced_df"].to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode()

    pdf_bytes = _build_summary_pdf(run_id, run)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"balanced_dataset_{run_id[:8]}.csv", csv_bytes)
        zf.writestr(f"run_summary_{run_id[:8]}.pdf", pdf_bytes)
        log_df = result.get("log_df")
        if log_df is not None:
            log_buf = io.StringIO()
            log_df.to_csv(log_buf, index=False)
            zf.writestr(f"changes_log_{run_id[:8]}.csv", log_buf.getvalue().encode())
    zip_buf.seek(0)

    filename = f"imbalancekit_export_{run_id[:8]}.zip"
    return StreamingResponse(
        iter([zip_buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

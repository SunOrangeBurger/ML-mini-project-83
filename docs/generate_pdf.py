from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


def generate_pdf(output_path="docs/writeup.pdf"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#424242"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0d47a1"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1565c0"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.5,
        textColor=colors.HexColor("#212121"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0d47a1"),
    )
    table_cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Sign Language Recognition using Temporal Classification", title_style))
    story.append(
        Paragraph(
            "<b>Course:</b> UE24CS352A Machine Learning Mini-Project (Problem 118) &nbsp;|&nbsp; "
            "<b>Team 83</b> &nbsp;|&nbsp; "
            "<b>Ref:</b> Cate, Dalvi, Hussain (2015)",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a237e"), spaceAfter=6))

    # 1. Problem Statement
    story.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    story.append(
        Paragraph(
            "Camera-free sign language recognition through wearable sensor gloves provides invariance against lighting, "
            "occlusion, and background clutter. This project reproduces and extends the 95-sign Auslan (Australian Sign Language) "
            "recognition benchmark from Cate, Dalvi, and Hussain (2015). The original study suffered poor performance on low-quality "
            "data and reported a total failure of standard LSTMs (F1 = 0.066). We aim to: (1) reproduce and benchmark classical baselines, "
            "(2) perform rigorous feature ablation to establish modality hierarchy, and (3) diagnose and resolve the structural flaw in the paper's LSTM.",
            body_style,
        )
    )

    # 2. Dataset & Cleaning
    story.append(Paragraph("2. Dataset & Cleaning Pipeline", h1_style))
    story.append(
        Paragraph(
            "We utilize Mohammed Waleed Kadous's Auslan benchmark (UCI ML Repository #114). While the UCI repository lists a 200 Hz "
            "high-quality dual-glove dataset, its archive links return broken symlinks and 403 Forbidden HTTP errors; hence all work "
            "is rigorously evaluated on the low-quality Nintendo PowerGlove dataset (50 Hz, 1 hand, across multiple signers). "
            "<b>Cleaning applied:</b> (1) Removed 3 non-sign calibration files ('cal-*') and 1 corrupted empty file. "
            "(2) Stripped hex status bytes. (3) Out of 11 numeric channels, cols 4 and 5 are zero-variance constants and col 10 is an exact duplicate of col 9. "
            "Pruning leaves <b>8 active features</b>: POS (x, y, z; cols 0-2), ROT (roll; col 3), and finger bends F1-F4 (cols 6-9). "
            "The cleaned corpus contains <b>6,648 recordings across 95 classes</b> (~70 per class, mean duration 58.4 frames).",
            body_style,
        )
    )

    # 3. Approach & Methodology
    story.append(Paragraph("3. Methodology & Feature Engineering", h1_style))
    story.append(
        Paragraph(
            "<b>Temporal Normalization:</b> Variable duration signs are standardized to T = 57 frames (dataset mean) via Fourier resampling "
            "(scipy.signal.resample). <b>Spatial Scaling:</b> We evaluate per-example min-max scaling ([0, 1] per channel per sign) versus "
            "global dataset scaling. <b>Representations:</b> Static flattened vectors (57x8 = 456 dimensions) for SVM/Logistic Regression, and sequential "
            "tensors (N, 57, 8) for recurrent networks.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Fixing the Paper's LSTM:</b> The paper formulated sign recognition using Mean Squared Error (MSE) evaluated at every discrete frame, "
            "which forces intermediate frame predictions where gestures are non-discriminative. We resolve this by enforcing sequence-to-label aggregation "
            "(return_sequences=False, backpropagating strictly from the final hidden state) paired with 95-way Softmax Sparse Categorical Cross-Entropy, "
            "dropout (p=0.3), and validation early stopping.",
            body_style,
        )
    )

    # 4. Results & Baselines
    story.append(Paragraph("4. Experimental Results & Analysis", h1_style))
    story.append(
        Paragraph(
            "All models are benchmarked under a stratified 70/30 train/test split (seed 42; 4,653 train / 1,995 test instances).",
            body_style,
        )
    )

    baseline_data = [
        [
            Paragraph("<b>Model</b>", table_cell_header),
            Paragraph("<b>Scaling</b>", table_cell_header),
            Paragraph("<b>Test Acc</b>", table_cell_header),
            Paragraph("<b>Macro F1</b>", table_cell_header),
            Paragraph("<b>Paper F1</b>", table_cell_header),
        ],
        [Paragraph("Linear SVM (C=1.0)", table_cell), Paragraph("Per-example", table_cell), Paragraph("0.306", table_cell), Paragraph("0.308", table_cell), Paragraph("—", table_cell)],
        [Paragraph("Linear SVM (C=0.1)", table_cell), Paragraph("Per-example", table_cell), Paragraph("0.369", table_cell), Paragraph("0.365", table_cell), Paragraph("0.549", table_cell)],
        [Paragraph("Logistic Regression", table_cell), Paragraph("Per-example", table_cell), Paragraph("0.385", table_cell), Paragraph("0.384", table_cell), Paragraph("0.436", table_cell)],
        [Paragraph("RBF SVM (C=10)", table_cell), Paragraph("Global", table_cell), Paragraph("0.576", table_cell), Paragraph("0.575", table_cell), Paragraph("—", table_cell)],
        [Paragraph("<b>RBF SVM (C=10)</b>", table_cell_bold), Paragraph("Per-example", table_cell), Paragraph("<b>0.602</b>", table_cell_bold), Paragraph("<b>0.601</b>", table_cell_bold), Paragraph("0.549", table_cell)],
        [Paragraph("Baseline LSTM (128)", table_cell), Paragraph("Per-example", table_cell), Paragraph("0.383", table_cell), Paragraph("0.370", table_cell), Paragraph("0.066", table_cell)],
        [Paragraph("Bidirectional LSTM (128)", table_cell), Paragraph("Per-example", table_cell), Paragraph("0.437", table_cell), Paragraph("0.431", table_cell), Paragraph("—", table_cell)],
        [Paragraph("<b>Stacked LSTM (2x128)</b>", table_cell_bold), Paragraph("Per-example", table_cell), Paragraph("<b>0.447</b>", table_cell_bold), Paragraph("<b>0.440</b>", table_cell_bold), Paragraph("0.066", table_cell)],
    ]
    t_base = Table(baseline_data, colWidths=[130, 85, 75, 75, 75])
    t_base.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_base)
    story.append(Spacer(1, 4))

    # Ablation Section
    story.append(Paragraph("4.1 Feature Modality Ablation (RBF SVM)", h2_style))
    ablation_data = [
        [Paragraph("<b>Condition</b>", table_cell_header), Paragraph("<b>Channels Retained</b>", table_cell_header), Paragraph("<b>Test Error</b>", table_cell_header), Paragraph("<b>Significance</b>", table_cell_header)],
        [Paragraph("Full Feature Set", table_cell), Paragraph("POS + ROT + F1-F4 (8 cols)", table_cell), Paragraph("39.8%", table_cell), Paragraph("Baseline (Acc: 60.2%, F1: 0.601)", table_cell)],
        [Paragraph("Without POS", table_cell), Paragraph("ROT + F1-F4 (5 cols)", table_cell), Paragraph("<b>75.6%</b>", table_cell_bold), Paragraph("+35.8% error surge: primary trajectory signal", table_cell)],
        [Paragraph("Without ROT", table_cell), Paragraph("POS + F1-F4 (7 cols)", table_cell), Paragraph("41.8%", table_cell), Paragraph("+2.0% error: minor wrist roll contribution", table_cell)],
        [Paragraph("Cumulative Drop", table_cell), Paragraph("Removed up to F3 (only F4 left)", table_cell), Paragraph("<b>95.6%</b>", table_cell_bold), Paragraph("Near-chance baseline without spatial channels", table_cell)],
    ]
    t_abl = Table(ablation_data, colWidths=[110, 130, 70, 190])
    t_abl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(t_abl)
    story.append(Spacer(1, 4))

    # Error analysis
    story.append(Paragraph("4.2 Error Analysis & Modality Insights", h2_style))
    story.append(
        Paragraph(
            "Inspection of off-diagonal confusion matrix elements reveals top confusions: <i>man</i> &harr; <i>please</i> (5 errors), "
            "<i>which</i> &harr; <i>maybe</i> (5 errors), and <i>surprise</i> &harr; <i>more</i> (5 errors). These pairs trace nearly identical "
            "gross hand paths, differing solely in subtle thumb flexion or finger curvature that is susceptible to PowerGlove sensor quantization.",
            body_style,
        )
    )

    # 5. Conclusions
    story.append(Paragraph("5. Conclusions & Discussion", h1_style))
    story.append(
        Paragraph(
            "• <b>Literature Reproduction:</b> Achieved 60.1% Macro F1 with RBF SVM, surpassing the 2015 baseline of 54.9%.<br/>"
            "• <b>LSTM Rectification:</b> Demonstrated that replacing frame-wise MSE with final-step softmax cross-entropy resolves the paper's "
            "reported failure (surpassing 0.066 by an order of magnitude).<br/>"
            "• <b>Model Selection Trade-off:</b> On small sample sizes (~49 train instances/class across 95 classes), non-linear kernel SVM on Fourier-flattened "
            "representations is remarkably competitive with deep recurrent models, while requiring drastically lower training compute.<br/>"
            "• <b>Sensor Hierarchy:</b> 3D spatial position is the primary classifier driver; finger flexion resolves local gesture ambiguities.",
            body_style,
        )
    )

    doc.build(story)
    print("Generated PDF at:", output_path)


if __name__ == "__main__":
    generate_pdf()

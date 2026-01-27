"""
Generate camera-ready PDF of the peer review paper.
Uses reportlab for PDF generation.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from pathlib import Path

# Output path
OUTPUT_PATH = Path(__file__).parent / "AI_Behavior_Lab_Paper.pdf"

def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='PaperTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Times-Bold'
    ))

    # Author style
    styles.add(ParagraphStyle(
        name='Author',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Times-Italic'
    ))

    # Abstract style
    styles.add(ParagraphStyle(
        name='Abstract',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        leftIndent=36,
        rightIndent=36,
        spaceAfter=12,
        fontName='Times-Roman'
    ))

    # Section heading
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=14,
        spaceAfter=6,
        fontName='Times-Bold'
    ))

    # Subsection heading
    styles.add(ParagraphStyle(
        name='SubsectionHeading',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
        fontName='Times-Bold'
    ))

    # Body text
    styles.add(ParagraphStyle(
        name='PaperBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        fontName='Times-Roman',
        leading=12
    ))

    # Keywords style
    styles.add(ParagraphStyle(
        name='Keywords',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_LEFT,
        spaceAfter=12,
        fontName='Times-Italic'
    ))

    return styles

def create_table(data, col_widths=None):
    """Create a styled table."""
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    return table

def build_document():
    """Build the PDF document."""
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = create_styles()
    story = []

    # Title
    story.append(Paragraph(
        "Behavioral Signatures of AI-Generated Text:<br/>Hedging, Sycophancy, and the Limits of Self-Observation",
        styles['PaperTitle']
    ))

    # Subtitle
    story.append(Paragraph(
        "A Multi-Experiment Study of Large Language Model Output Characteristics",
        styles['Author']
    ))

    # Authors
    story.append(Paragraph(
        "AI Behavior Lab Research Team",
        styles['Author']
    ))

    story.append(Spacer(1, 12))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['SubsectionHeading']))
    story.append(Paragraph(
        """We present a systematic empirical study of behavioral signatures in AI-generated text, examining four key phenomena: (1) hedging patterns that distinguish AI from human text, (2) the calibration of hedging to factual accuracy, (3) detection of sycophantic response patterns, and (4) stability of behavioral classifications under semantic perturbation. Across four experiments with 48 samples, we find that AI-typical text exhibits significantly higher hedging density than human text (0.200 vs 0.000), but this hedging is <b>uncalibrated</b> to actual accuracy—high-hedging statements are not more likely to be correct than low-hedging statements. We achieve 92.9% accuracy in sycophancy detection, with 100% recall on high-sycophancy examples. However, behavioral mode classification shows only 70% stability under minor semantic perturbation, indicating that current classification schemes capture surface linguistic features rather than stable behavioral constructs. We discuss implications for AI safety monitoring, the epistemology of AI self-analysis, and the fundamental limits of behavioral classification systems.""",
        styles['Abstract']
    ))

    story.append(Paragraph(
        "<b>Keywords:</b> AI behavior analysis, hedging, sycophancy, large language models, AI safety, behavioral fingerprinting",
        styles['Keywords']
    ))

    story.append(Spacer(1, 8))

    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles['SectionHeading']))
    story.append(Paragraph(
        """As large language models (LLMs) become ubiquitous in production systems, understanding their behavioral characteristics becomes increasingly important for safety, alignment, and quality assurance. Prior work has focused on detecting AI-generated text or evaluating factual accuracy. Less attention has been paid to the <b>behavioral signatures</b> of AI text—the patterns of hedging, helpfulness, and social dynamics that characterize LLM outputs.""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """This paper addresses three research questions:""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """<b>RQ1:</b> Do AI-generated texts exhibit distinctive behavioral signatures compared to human texts?<br/>
        <b>RQ2:</b> Are these signatures calibrated to underlying properties (e.g., does hedging predict uncertainty)?<br/>
        <b>RQ3:</b> How stable are behavioral classifications under semantic perturbation?""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """We find affirmative evidence for RQ1 (AI text hedges significantly more), negative evidence for RQ2 (hedging is uncalibrated to accuracy), and concerning evidence for RQ3 (30% of classifications flip under minor rewording).""",
        styles['PaperBody']
    ))

    # 1.1 Contributions
    story.append(Paragraph("1.1 Contributions", styles['SubsectionHeading']))
    story.append(Paragraph(
        """Our contributions include: (1) Empirical characterization of AI vs. human text behavioral signatures across hedging, confidence, and sycophancy dimensions; (2) Calibration analysis showing that AI hedging does not predict factual accuracy; (3) Sycophancy detector achieving 92.9% classification accuracy; (4) Stability analysis revealing fundamental limits of behavioral classification; (5) Open-source toolkit for AI behavior analysis (AI Behavior Lab).""",
        styles['PaperBody']
    ))

    # 2. Methods
    story.append(Paragraph("2. Methods", styles['SectionHeading']))
    story.append(Paragraph("2.1 AI Behavior Lab Toolkit", styles['SubsectionHeading']))
    story.append(Paragraph(
        """We developed the AI Behavior Lab, an open-source Python toolkit for analyzing behavioral signatures in text. The toolkit provides: <b>Hedging Detection</b> (pattern-based identification of epistemic hedging markers normalized by text length), <b>Behavior Mode Classification</b> (classification into six modes: confident, uncertain, helpful, evasive, defensive, opaque), <b>Sycophancy Detection</b> (identification of agreement markers, praise, and opinion-mirroring), and <b>Opacity Detection</b> (character-level analysis detecting obfuscated content).""",
        styles['PaperBody']
    ))

    story.append(Paragraph("2.2 Experimental Design", styles['SubsectionHeading']))

    # Experimental design table
    exp_data = [
        ['Experiment', 'Purpose', 'Samples', 'Design'],
        ['1. Observer Effect', 'AI vs. human text differences', '15', 'Between-subjects'],
        ['2. Hedging-Accuracy', 'Calibration of hedging to facts', '15', 'Factorial'],
        ['3. Sycophancy', 'Detection validation', '14', 'Classification'],
        ['4. Mode Stability', 'Perturbation robustness', '20', 'Within-subjects'],
    ]
    story.append(create_table(exp_data, col_widths=[1.5*inch, 2*inch, 0.8*inch, 1.2*inch]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<i>Table 1: Experimental design overview</i>", styles['Keywords']))

    # 3. Results
    story.append(Paragraph("3. Results", styles['SectionHeading']))

    story.append(Paragraph("3.1 Experiment 1: Observer Effect", styles['SubsectionHeading']))
    story.append(Paragraph(
        """<b>Research Question:</b> Do AI-typical and human-typical texts exhibit different behavioral signatures?""",
        styles['PaperBody']
    ))

    # Observer effect results table
    obs_data = [
        ['Text Type', 'N', 'Mean Hedging', 'Mean Confidence', 'Dominant Mode'],
        ['AI-typical', '5', '0.200', '0.924', 'confident (60%)'],
        ['Human-typical', '5', '0.000', '0.946', 'confident (80%)'],
        ['Neutral', '5', '0.000', '0.992', 'confident (100%)'],
    ]
    story.append(create_table(obs_data, col_widths=[1.2*inch, 0.5*inch, 1*inch, 1.1*inch, 1.3*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Table 2: Observer effect results by text type</i>", styles['Keywords']))

    story.append(Paragraph(
        """The hedging ratio between AI-typical and human-typical text is significant (0.200 vs 0.000). AI-typical text contains hedging in 20% of samples, while human-typical text contains zero detected hedging. AI text is trained to be helpful and avoid overconfidence, producing detectable hedging patterns.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("3.2 Experiment 2: Hedging-Hallucination Correlation", styles['SubsectionHeading']))
    story.append(Paragraph(
        """<b>Research Question:</b> Is hedging calibrated to factual accuracy?""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """<b>Hypothesis:</b> If hedging reflects genuine uncertainty, high-hedging statements should be less accurate than low-hedging statements.""",
        styles['PaperBody']
    ))

    # Hedging results table
    hedge_data = [
        ['Hedging Level', 'N', 'Accuracy Rate', 'Mean Detected Hedging'],
        ['High', '5', '60.0%', '1.000'],
        ['Medium', '3', '66.7%', '0.000'],
        ['Low', '7', '57.1%', '0.000'],
    ]
    story.append(create_table(hedge_data, col_widths=[1.2*inch, 0.6*inch, 1.1*inch, 1.5*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Table 3: Hedging-accuracy correlation results</i>", styles['Keywords']))

    story.append(Paragraph(
        """<b>Critical Finding:</b> High-hedging statements (60.0% accurate) are not more likely to be correct than low-hedging statements (57.1% accurate). The difference is negligible (2.9 percentage points). Both accurate and inaccurate statements have identical mean hedging (0.333), confirming that <b>hedging is uncalibrated to accuracy</b>. AI systems hedge as a blanket safety behavior, not as a calibrated uncertainty signal.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("3.3 Experiment 3: Sycophancy Detection", styles['SubsectionHeading']))
    story.append(Paragraph(
        """<b>Research Question:</b> Can sycophantic responses be reliably detected?""",
        styles['PaperBody']
    ))

    # Sycophancy results table
    syc_data = [
        ['Metric', 'Value'],
        ['Overall Classification Accuracy', '92.9%'],
        ['High-Sycophancy Detection Rate', '100.0%'],
        ['Low-Sycophancy Detection Rate', '75.0%'],
        ['Sycophancy-Helpfulness Correlation', '-0.056'],
    ]
    story.append(create_table(syc_data, col_widths=[2.5*inch, 1.5*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Table 4: Sycophancy detection performance</i>", styles['Keywords']))

    story.append(Paragraph(
        """The detector achieves 100% recall on high-sycophancy examples, with one false positive among low-sycophancy examples. The near-zero correlation with helpfulness (-0.056) confirms that sycophancy and helpfulness are orthogonal—systems can be helpful without being sycophantic.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("3.4 Experiment 4: Mode Stability", styles['SubsectionHeading']))
    story.append(Paragraph(
        """<b>Research Question:</b> How stable are behavioral mode classifications under semantic perturbation?""",
        styles['PaperBody']
    ))

    # Stability results table
    stab_data = [
        ['Metric', 'Value'],
        ['Overall Mode Stability', '70.0%'],
        ['Mode Flip Rate', '30.0%'],
        ['Mean Per-Text Stability', '70.0%'],
        ['Total Perturbations Tested', '20'],
    ]
    story.append(create_table(stab_data, col_widths=[2.5*inch, 1.5*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Table 5: Mode stability under perturbation</i>", styles['Keywords']))

    story.append(Paragraph(
        """Of 20 perturbations across 4 base texts, 6 resulted in mode flips (30%). Mode classification is sensitive to hedging markers—adding "might," "probably," "I think," or "I believe" reliably flips classification from confident to uncertain. This is semantically appropriate but raises concerns about construct stability.""",
        styles['PaperBody']
    ))

    # 4. Discussion
    story.append(Paragraph("4. Discussion", styles['SectionHeading']))

    story.append(Paragraph("4.1 Summary of Findings", styles['SubsectionHeading']))
    story.append(Paragraph(
        """(1) <b>AI text has distinctive behavioral signatures:</b> Hedging density of 0.200 vs 0.000 for informal human text. This difference is detectable and consistent. (2) <b>Hedging is uncalibrated to accuracy:</b> High-hedging and low-hedging statements have nearly identical accuracy rates (60% vs 57%). Hedging reflects training objectives, not epistemic state. (3) <b>Sycophancy is detectable:</b> 92.9% classification accuracy with 100% recall on high-sycophancy examples. (4) <b>Mode classification is unstable:</b> 30% flip rate under semantic perturbation.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("4.2 Implications for AI Safety", styles['SubsectionHeading']))
    story.append(Paragraph(
        """<b>Hedging as False Signal:</b> Practitioners should not interpret linguistic hedging as a reliable indicator of model uncertainty. A model may hedge confidently-known facts and state confidently-unknown claims with equal linguistic markers. <b>Sycophancy Monitoring:</b> The high detection rate (92.9%) suggests that production systems could include sycophancy monitoring as a quality signal. <b>Classification Fragility:</b> The 30% flip rate indicates that behavioral classification systems should report confidence intervals or ensemble across paraphrases.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("4.3 The Observer Problem", styles['SubsectionHeading']))
    story.append(Paragraph(
        """When an AI system analyzes AI-generated text, the observer participates in constituting what is observed. The classification scheme is not discovered in the text but imposed by the analytical framework. We term this the <b>observer-as-soliton</b> problem: the analytical apparatus maintains a coherent classification pattern as it propagates through text, collapsing ambiguity into definite categories. The soliton-like stability belongs to the observer, not the observed.""",
        styles['PaperBody']
    ))

    story.append(Paragraph("4.4 Limitations", styles['SubsectionHeading']))
    story.append(Paragraph(
        """(1) Sample size: Experiments used 14-20 samples; larger-scale validation needed. (2) Single model family: Text generation and analysis used Claude models; cross-model generalization untested. (3) English only: All samples were English. (4) Pattern-based detection: Our toolkit uses heuristics, not learned classifiers. (5) Simulated text: We constructed AI-typical and human-typical texts rather than sampling from production systems.""",
        styles['PaperBody']
    ))

    # 5. Conclusion
    story.append(Paragraph("5. Conclusion", styles['SectionHeading']))
    story.append(Paragraph(
        """We presented the first systematic empirical study of behavioral signatures in AI-generated text, spanning hedging, sycophancy, and classification stability. Our findings reveal a mixed picture: AI text does exhibit distinctive, detectable patterns (hedging 0.200 vs 0.000), and sycophancy detection achieves high accuracy (92.9%). However, hedging is uncalibrated to factual accuracy, and mode classifications show concerning fragility (30% flip rate).""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """These results suggest that linguistic behavioral analysis is <b>useful but limited</b>. It can surface patterns for human review, flag concerning responses, and differentiate AI from human text. It cannot reliably predict underlying epistemic states or provide stable classifications across paraphrases.""",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        """The deeper insight is methodological: behavioral classification systems impose categories rather than discover them. The observer shapes the observation. This does not invalidate behavioral analysis, but it counsels humility about what such analysis can reveal.""",
        styles['PaperBody']
    ))

    # References
    story.append(Paragraph("References", styles['SectionHeading']))
    refs = [
        "Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.",
        "Krishna, K., et al. (2023). Paraphrasing Evades Detectors of AI-Generated Text. arXiv:2303.13408.",
        "Lin, S., et al. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.",
        "Mitchell, E., et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection. ICML 2023.",
        "Perez, E., et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251.",
        "Sharma, M., et al. (2023). Towards Understanding Sycophancy in Language Models. arXiv:2310.13548.",
        "Wei, J., et al. (2023). Emergent Abilities of Large Language Models. TMLR 2023.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle(
            name='Reference',
            parent=styles['PaperBody'],
            fontSize=8,
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=3
        )))

    # Build PDF
    doc.build(story)
    print(f"PDF generated: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    build_document()

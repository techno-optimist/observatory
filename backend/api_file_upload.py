"""
File Upload API - Handle file uploads for narrative analysis

Supports:
- Plain text files (.txt)
- PDF files (.pdf)
- Word documents (.docx)
- Markdown files (.md)
- HTML files (.html)

Extracts text content and optionally runs full narrative analysis.
"""

import io
import re
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


def split_into_narratives(text: str, min_length: int = 50) -> List[str]:
    """
    Split text into individual narrative segments for analysis.

    Uses paragraph breaks as primary delimiter, with sentence
    splitting for very long paragraphs.
    """
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)

    narratives = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < min_length:
            continue

        # If paragraph is very long, split into sentences
        if len(para) > 500:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < 400:
                    current_chunk += " " + sent if current_chunk else sent
                else:
                    if len(current_chunk) >= min_length:
                        narratives.append(current_chunk.strip())
                    current_chunk = sent
            if len(current_chunk) >= min_length:
                narratives.append(current_chunk.strip())
        else:
            narratives.append(para)

    return narratives

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


class FileUploadResponse(BaseModel):
    """Response from file upload"""
    success: bool
    filename: str
    file_type: str
    text: str
    word_count: int
    char_count: int
    error: Optional[str] = None


class FileAnalysisResponse(BaseModel):
    """Response from file upload with analysis"""
    success: bool
    filename: str
    file_type: str
    text: str
    word_count: int
    char_count: int
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from Word document"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise ValueError(f"Failed to extract text from Word document: {str(e)}")


def extract_text_from_html(content: bytes) -> str:
    """Extract text from HTML file"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Get text
        text = soup.get_text(separator='\n')

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"Error extracting HTML text: {e}")
        raise ValueError(f"Failed to extract text from HTML: {str(e)}")


def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Replace multiple newlines with double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def get_file_type(filename: str) -> str:
    """Determine file type from filename"""
    suffix = Path(filename).suffix.lower()
    type_map = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.md': 'markdown',
        '.html': 'html',
        '.htm': 'html',
    }
    return type_map.get(suffix, 'unknown')


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Upload a file and extract text content.

    Supported formats: .txt, .pdf, .docx, .md, .html
    """
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    filename = file.filename or "unknown"
    file_type = get_file_type(filename)

    try:
        # Extract text based on file type
        if file_type == 'pdf':
            text = extract_text_from_pdf(content)
        elif file_type == 'docx':
            text = extract_text_from_docx(content)
        elif file_type == 'html':
            text = extract_text_from_html(content)
        elif file_type in ['text', 'markdown']:
            # Try to decode as UTF-8, fallback to latin-1
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_type}. Supported: .txt, .pdf, .docx, .md, .html"
            )

        # Clean the text
        text = clean_text(text)

        if not text:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the file"
            )

        return FileUploadResponse(
            success=True,
            filename=filename,
            file_type=file_type,
            text=text,
            word_count=len(text.split()),
            char_count=len(text)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return FileUploadResponse(
            success=False,
            filename=filename,
            file_type=file_type,
            text="",
            word_count=0,
            char_count=0,
            error=str(e)
        )


@router.post("/upload-and-analyze", response_model=FileAnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    run_full_analysis: bool = Form(default=True),
):
    """
    Upload a file, extract text, and optionally run full narrative analysis.

    This endpoint:
    1. Extracts text from the uploaded file
    2. Splits into narratives (paragraphs/sentences)
    3. Runs narrative analysis on each segment
    4. Returns aggregated results with mode distribution, coordinates, etc.
    """
    # First, extract text
    upload_result = await upload_file(file)

    if not upload_result.success:
        return FileAnalysisResponse(
            success=False,
            filename=upload_result.filename,
            file_type=upload_result.file_type,
            text="",
            word_count=0,
            char_count=0,
            error=upload_result.error
        )

    if not run_full_analysis:
        return FileAnalysisResponse(
            success=True,
            filename=upload_result.filename,
            file_type=upload_result.file_type,
            text=upload_result.text,
            word_count=upload_result.word_count,
            char_count=upload_result.char_count,
            analysis=None
        )

    # Run narrative analysis
    try:
        from api_observer_chat import analyze_text_internal

        # Split text into narratives
        narratives = split_into_narratives(upload_result.text)

        if not narratives:
            return FileAnalysisResponse(
                success=True,
                filename=upload_result.filename,
                file_type=upload_result.file_type,
                text=upload_result.text,
                word_count=upload_result.word_count,
                char_count=upload_result.char_count,
                analysis={"error": "No analyzable narratives found in text"}
            )

        # Analyze each narrative
        results = []
        for narrative in narratives:
            if len(narrative.strip()) < 20:  # Skip very short segments
                continue
            try:
                result = await analyze_text_internal(narrative)
                results.append({
                    "text": narrative[:200] + "..." if len(narrative) > 200 else narrative,
                    "mode": result.get("mode", {}).get("primary_mode", "NEUTRAL"),
                    "coordinates": result.get("vector", {}),
                    "confidence": result.get("mode", {}).get("confidence", 0)
                })
            except Exception as e:
                logger.warning(f"Failed to analyze narrative: {e}")
                continue

        if not results:
            return FileAnalysisResponse(
                success=True,
                filename=upload_result.filename,
                file_type=upload_result.file_type,
                text=upload_result.text,
                word_count=upload_result.word_count,
                char_count=upload_result.char_count,
                analysis={"error": "Analysis failed for all narratives"}
            )

        # Aggregate results
        import numpy as np

        modes = [r["mode"] for r in results]
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        total = len(modes)
        mode_distribution = {m: round(c / total * 100, 1) for m, c in mode_counts.items()}
        dominant_mode = max(mode_counts, key=mode_counts.get)

        # Calculate mean coordinates
        agencies = [r["coordinates"].get("agency", 0) for r in results]
        justices = [r["coordinates"].get("perceived_justice", 0) for r in results]
        belongings = [r["coordinates"].get("belonging", 0) for r in results]

        analysis = {
            "narratives_count": len(results),
            "dominant_mode": dominant_mode,
            "mode_distribution": mode_distribution,
            "coordinates": {
                "agency": {
                    "mean": round(float(np.mean(agencies)), 4),
                    "std": round(float(np.std(agencies)), 4),
                    "min": round(float(np.min(agencies)), 4),
                    "max": round(float(np.max(agencies)), 4)
                },
                "perceived_justice": {
                    "mean": round(float(np.mean(justices)), 4),
                    "std": round(float(np.std(justices)), 4),
                    "min": round(float(np.min(justices)), 4),
                    "max": round(float(np.max(justices)), 4)
                },
                "belonging": {
                    "mean": round(float(np.mean(belongings)), 4),
                    "std": round(float(np.std(belongings)), 4),
                    "min": round(float(np.min(belongings)), 4),
                    "max": round(float(np.max(belongings)), 4)
                }
            },
            "sample_narratives": results[:5],  # First 5 for preview
            "all_narratives": results
        }

        return FileAnalysisResponse(
            success=True,
            filename=upload_result.filename,
            file_type=upload_result.file_type,
            text=upload_result.text,
            word_count=upload_result.word_count,
            char_count=upload_result.char_count,
            analysis=analysis
        )

    except Exception as e:
        logger.error(f"Error analyzing file content: {e}")
        return FileAnalysisResponse(
            success=True,
            filename=upload_result.filename,
            file_type=upload_result.file_type,
            text=upload_result.text,
            word_count=upload_result.word_count,
            char_count=upload_result.char_count,
            analysis={"error": str(e)}
        )

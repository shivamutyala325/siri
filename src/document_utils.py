from typing import List, Tuple
import io

import fitz  # PyMuPDF
import requests
from fastapi import HTTPException


def download_document(url: str, timeout: int = 20) -> Tuple[bytes, str]:
    """
    Download the document from the given URL.

    Returns:
        file_bytes: raw bytes of the document
        content_type: MIME type string (e.g. "application/pdf", "image/png")
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download document from URL: {e}"
        )

    content_type = resp.headers.get("Content-Type", "").lower()
    file_bytes = resp.content
    return file_bytes, content_type


def split_into_pages(
    file_bytes: bytes, content_type: str
) -> List[Tuple[int, bytes, str]]:
    """
    Convert the downloaded document into a list of (page_no, image_bytes, mime_type).

    Supports:
        - PDF (multi-page)
        - Single-page PNG/JPEG

    Returns:
        List of tuples (page_no, image_bytes, mime_type)
    """
    pages: List[Tuple[int, bytes, str]] = []

    if "pdf" in content_type or content_type == "":
        # Treat as PDF if content_type has 'pdf' or unknown but bytes look pdf-ish.
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to open PDF document: {e}"
            )

        for i, page in enumerate(doc):
            pix = page.get_pixmap()  # default resolution; can tweak dpi if needed
            img_bytes = pix.tobytes("png")  # convert to PNG bytes
            pages.append((i + 1, img_bytes, "image/png"))

        if not pages:
            raise HTTPException(status_code=400, detail="No pages found in PDF document.")

    elif "png" in content_type or "jpeg" in content_type or "jpg" in content_type:
        # Single-page image
        pages.append((1, file_bytes, content_type))
    else:
        # Fallback: try as PDF first, then PNG
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                pages.append((i + 1, img_bytes, "image/png"))
        except Exception:
            # Can't parse as PDF, assume single image
            pages.append((1, file_bytes, "image/png"))

    return pages

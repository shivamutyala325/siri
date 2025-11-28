# main.py
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from document_utils import download_document, split_into_pages
from usemodel import UseApiModel

load_dotenv()  # loads GOOGLE_API_KEY / GEMINI_API_KEY if present

app = FastAPI(title="HackRx Bill Extraction API")


class ExtractRequest(BaseModel):
    document: str  # URL of the document (image or multi-page PDF)


@app.post("/extract-bill-data")
async def extract_bill_data(req: ExtractRequest) -> Dict[str, Any]:
    """
    Main endpoint as per problem statement:

    Request:
        {
          "document": "<url to file>"
        }

    Response:
        {
          "is_success": boolean,
          "token_usage": {
            "total_tokens": int,
            "input_tokens": int,
            "output_tokens": int
          },
          "data": {
            "pagewise_line_items": [
              {
                "page_no": "string",
                "page_type": "Bill Detail | Final Bill | Pharmacy",
                "bill_items": [
                  {
                    "item_name": "string",
                    "item_amount": float,
                    "item_rate": float,
                    "item_quantity": float
                  }
                ]
              }
            ],
            "total_item_count": int
          }
        }
    """

    # 1️⃣ Download the document
    file_bytes, content_type = download_document(req.document)

    # 2️⃣ Split into per-page image bytes
    pages = split_into_pages(file_bytes, content_type)
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No pages could be extracted from the provided document.",
        )

    # 3️⃣ Initialize model wrapper
    model = UseApiModel()

    # 4️⃣ For each page, call Gemini and collect results
    pagewise_results: List[Dict[str, Any]] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    for page_no, img_bytes, mime in pages:
        page_data, usage = model.extract_page(page_no, img_bytes, mime)

        # Accumulate token usage
        total_tokens += usage.get("total_tokens", 0)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

        pagewise_results.append(page_data)

    # 5️⃣ Build final schema: pagewise_line_items + total_item_count
    pagewise_line_items: List[Dict[str, Any]] = []
    total_item_count = 0

    for page in pagewise_results:
        page_no_str = str(page.get("page_no", "1"))
        page_type = page.get("page_type", "Bill Detail")
        items = page.get("items", [])

        bill_items: List[Dict[str, Any]] = []

        for item in items:
            # Basic defensive parsing and double-count protection
            name = (item.get("name") or "").strip()

            # Heuristic: avoid obvious total/subtotal rows if model leaked them
            lowered = name.lower()
            if any(
                kw in lowered
                for kw in ["total", "subtotal", "sub total", "grand total", "net amount"]
            ):
                # likely a summary row, skip
                continue

            try:
                rate = float(item.get("rate", 0.0) or 0.0)
            except (ValueError, TypeError):
                rate = 0.0

            try:
                qty = float(item.get("quantity", 0.0) or 0.0)
            except (ValueError, TypeError):
                qty = 0.0

            try:
                amount = float(item.get("amount", 0.0) or 0.0)
            except (ValueError, TypeError):
                amount = 0.0

            bill_items.append(
                {
                    "item_name": name,
                    "item_amount": amount,
                    "item_rate": rate,
                    "item_quantity": qty,
                }
            )

        total_item_count += len(bill_items)

        pagewise_line_items.append(
            {
                "page_no": page_no_str,
                "page_type": page_type,  # "Bill Detail" | "Final Bill" | "Pharmacy"
                "bill_items": bill_items,
            }
        )

    # 6️⃣ Prepare final response
    is_success = True

    response: Dict[str, Any] = {
        "is_success": is_success,
        "token_usage": {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "data": {
            "pagewise_line_items": pagewise_line_items,
            "total_item_count": total_item_count,
        },
    }

    return response

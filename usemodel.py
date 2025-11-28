# use_api_model.py
import json
import re
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types


class UseApiModel:
    """
    Wrapper around Gemini API to extract line items from a single page image.

    Input: image bytes + mime_type
    Output:
        - page_result: {
              "page_no": "1",
              "page_type": "Bill Detail",
              "items": [
                  {"name": "...", "rate": 0.0, "quantity": 0.0, "amount": 0.0}
              ]
          }
        - usage: {
              "total_tokens": int,
              "input_tokens": int,
              "output_tokens": int
          }
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client()
        self.model = model

        # Minimal, multi-page-aware prompt
        self.prompt = """
You are an invoice extraction engine.

For this single page of a bill/invoice:

1. Determine the page type:
   - "Bill Detail"    : pages that list line items in detail
   - "Final Bill"     : overall summary / final total page
   - "Pharmacy"       : medicine / pharmacy billing pages

2. Extract ONLY the line items (rows) from this page.
   A line item is a row with:
   - an item name
   - a quantity
   - a rate (price per unit)
   - an amount (net amount for that row, after discounts)

3. Ignore the following kinds of rows:
   - Subtotal / Total / Grand Total / Net Amount Payable
   - Tax, GST, VAT, CGST, SGST, service charges
   - Round off / rounding adjustment
   - Payment mode, balances, any non-item text.

Return STRICT JSON ONLY in this shape:

{
  "page_no": "string",                       // page number as a string (e.g. "1")
  "page_type": "Bill Detail | Final Bill | Pharmacy",
  "items": [
    {
      "name": "string",                      // item name exactly as in the bill
      "rate": 0.0,                           // rate per unit, as printed
      "quantity": 0.0,                       // quantity/value, as printed
      "amount": 0.0                          // net amount for this line, after discounts
    }
  ]
}

Rules:
- Output MUST be valid JSON, no comments, no trailing commas.
- Do NOT wrap the JSON in markdown code fences.
- If the page has no items, return:
  {
    "page_no": "...",
    "page_type": "...",
    "items": []
  }
"""

    def _clean_json_text(self, text: str) -> str:
        """
        Clean the model output to get pure JSON string.
        Handles cases where model wraps JSON in ``` or ```json fences.
        """
        text = text.strip()

        if "```" in text:
            parts = text.split("```")
            candidates = [p for p in parts if "{" in p and "}" in p]
            if candidates:
                text = candidates[0].strip()
                text = re.sub(r"^json", "", text, flags=re.IGNORECASE).strip()

        return text

    def _extract_usage(self, response) -> Dict[str, int]:
        """
        Extract token usage from the Gemini response in a defensive way,
        since field names may vary slightly across SDK versions.
        """
        usage = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        usage_meta = getattr(response, "usage_metadata", None) or getattr(
            response, "usage", None
        )

        if usage_meta is None:
            return usage

        usage["total_tokens"] = (
            getattr(usage_meta, "total_token_count", None)
            or getattr(usage_meta, "total_tokens", None)
            or 0
        )
        usage["input_tokens"] = (
            getattr(usage_meta, "prompt_token_count", None)
            or getattr(usage_meta, "input_tokens", None)
            or getattr(usage_meta, "input_token_count", None)
            or 0
        )
        usage["output_tokens"] = (
            getattr(usage_meta, "candidates_token_count", None)
            or getattr(usage_meta, "output_tokens", None)
            or 0
        )

        return usage

    def extract_page(
        self, page_no: int, image_bytes: bytes, mime_type: str
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Call Gemini for a single page and parse the minimal JSON.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                self.prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )

        raw_text = response.text or ""
        json_text = self._clean_json_text(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            # Fallback: no items if parsing fails
            data = {
                "page_no": str(page_no),
                "page_type": "Bill Detail",
                "items": [],
            }

        # Ensure page_no is set
        if "page_no" not in data or not data["page_no"]:
            data["page_no"] = str(page_no)

        # Ensure items list exists
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        data["items"] = items

        # Clean page_type: default if missing
        page_type = data.get("page_type", "")
        valid_types = {"Bill Detail", "Final Bill", "Pharmacy"}
        if page_type not in valid_types:
            data["page_type"] = "Bill Detail"

        # Extract usage
        usage = self._extract_usage(response)

        return data, usage

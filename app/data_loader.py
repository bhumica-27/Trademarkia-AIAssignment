import os
import re
from pathlib import Path
from typing import List, Dict

from app.config import DATA_DIR

_KEEP_HEADERS = {"subject"}


def _parse_document(raw: str) -> str:
    lines = raw.split("\n")
    body_start = 0
    subject = ""

    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            # Remove Re: prefixes — they add no semantic value
            subject = re.sub(r"^(Re:\s*)+", "", subject, flags=re.IGNORECASE).strip()

    body_lines = lines[body_start:]

    cleaned: List[str] = []
    for line in body_lines:
        stripped = line.strip()

        # Skip quoted text (replies)
        if stripped.startswith(">") or stripped.startswith("|"):
            continue

        # Stop at signature delimiter
        if stripped in ("--", "-- "):
            break

        # Skip lines that look like attribution lines ("John wrote:" etc.)
        if re.match(r"^[\w\s,.<>@]+writes?:$", stripped, re.IGNORECASE):
            continue

        cleaned.append(line)

    body = "\n".join(cleaned)

    # Collapse whitespace
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = body.strip()

    # Prepend subject for a richer semantic signal
    if subject:
        body = f"{subject}\n\n{body}"

    return body


def load_documents() -> List[Dict]:
    documents: List[Dict] = []
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for category_dir in sorted(DATA_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for doc_path in sorted(category_dir.iterdir()):
            if not doc_path.is_file():
                continue
            try:
                raw = doc_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            text = _parse_document(raw)

            # Skip near-empty documents (see module docstring)
            if len(text) < 20:
                continue

            documents.append(
                {
                    "doc_id": f"{category}_{doc_path.name}",
                    "category": category,
                    "text": text,
                }
            )

    return documents

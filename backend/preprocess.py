# mediquery/backend/preprocess.py
import os
import json
import pdfplumber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_pdfs(pdf_dir="data/pdfs", output_file="data/chunks.json"):
    chunks = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            logger.info(f"Processing {filename}...")
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text:
                            for i in range(0, len(text), 500):
                                chunk = text[i:i+500]
                                chunks.append({
                                    "text": chunk,
                                    "source": filename,
                                    "page": page_num
                                })
                        else:
                            logger.warning(f"No text extracted from {filename}, page {page_num}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    if not chunks:
        logger.error("No chunks extracted from any PDFs.")
        raise ValueError("No text extracted from PDFs. Check PDF content or format.")
    logger.info(f"Saving {len(chunks)} chunks to {output_file}")
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=2)
    return chunks

if __name__ == "__main__":
    preprocess_pdfs()
# mediquery/backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGPipeline
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info("Starting MediQuery backend...")

app = FastAPI(title="MediQuery API")

class QueryRequest(BaseModel):
    query: str
    mode: str
    num_questions: int = 5  # Default to 5 questions for quiz mode

# Initialize RAG pipeline with direct paths
global_pipeline = RAGPipeline(
    index_file="D:\\Company Tasks\\FInal Project\\MedQuery\\data\\index\\faiss.index",
    metadata_file="D:\\Company Tasks\\FInal Project\\MedQuery\\data\\metadata.json"
)

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        result = global_pipeline.process(request.query, request.mode, request.num_questions)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
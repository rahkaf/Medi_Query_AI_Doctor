# MediQuery - Medical Information Assistant

MediQuery is an AI-powered medical information assistant that uses Retrieval-Augmented Generation (RAG) to provide accurate answers to medical questions based on a corpus of medical documents.

## Features

- **Quick Answers**: Get concise responses (100-200 words) to medical questions
- **Detailed Research**: Receive comprehensive information (400-600 words) with citations
- **Quiz Generation**: Create multiple-choice questions based on medical topics
- **History Tracking**: Save and review previous queries and responses

## Architecture

MediQuery uses a RAG (Retrieval-Augmented Generation) architecture with the following components:

- **Frontend**: Streamlit web application
- **Backend**: FastAPI server
- **Embedding Model**: BioBERT for domain-specific text embeddings
- **Generation Model**: BART for answer generation
- **Vector Database**: FAISS for efficient similarity search

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place medical PDF documents in the `data/pdfs/` directory
4. Preprocess the documents:
   ```
   python backend/preprocess.py
   ```
5. Generate embeddings:
   ```
   python backend/embeddings.py
   ```

## Running the Application

1. Start the backend server:
   ```
   python backend/main.py
   ```
2. In a separate terminal, start the frontend:
   ```
   cd frontend
   streamlit run app.py
   ```
3. Open your browser and navigate to http://localhost:8501

## Project Structure

- `backend/`: Contains the API server and RAG pipeline
  - `main.py`: FastAPI server implementation
  - `rag.py`: RAG pipeline implementation
  - `embeddings.py`: Document embedding generation
  - `preprocess.py`: PDF document preprocessing
- `frontend/`: Contains the Streamlit web application
  - `app.py`: Streamlit UI implementation
- `data/`: Stores documents, embeddings, and query history
  - `pdfs/`: Directory for source PDF documents
  - `chunks.json`: Extracted text chunks from documents
  - `index.faiss`: FAISS vector index
  - `metadata.json`: Metadata for indexed chunks
  - `history.json`: Query history

## License

MIT

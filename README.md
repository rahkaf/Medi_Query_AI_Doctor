# MediQuery - AI-Powered Medical Information Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.2.0+-FF4B4B.svg)](https://streamlit.io)

MediQuery is an advanced medical information retrieval system that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers to medical queries. Built specifically for HIV/AIDS information retrieval, it combines the power of BioBERT embeddings with Google's Gemini API for precise and reliable medical information.

## 🌟 Features

- **💨 Quick Mode**: Concise, focused answers (100-200 words)
- **🔍 Research Mode**: Comprehensive responses with citations (400-600 words)
- **📝 Quiz Generation**: Interactive multiple-choice questions for learning
- **📊 History Tracking**: Review past queries and responses
- **🎯 Domain-Specific**: Specialized in HIV/AIDS information
- **🔗 Context-Aware**: Uses RAG for accurate information retrieval

## 🏗️ Architecture


- **Frontend**: Streamlit-based interactive web interface
- **Backend**: FastAPI server with async support
- **Embedding Model**: BioBERT for medical domain embeddings
- **Generation Model**: Google Gemini API for response generation
- **Vector Store**: FAISS for efficient similarity search

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- 4GB+ RAM
- CUDA-compatible GPU (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MediQuery.git
   cd MediQuery
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   copy .env.example .env
   # Edit .env with your Google API key
   ```

5. Initialize the system:
   ```bash
   python backend/preprocess.py
   python backend/embeddings.py
   ```

## 🖥️ Usage

1. Start the backend server:
   ```bash
   python backend/main.py
   ```

2. Launch the frontend (in a new terminal):
   ```bash
   cd frontend
   streamlit run app.py
   ```

3. Access the application at [http://localhost:8501](http://localhost:8501)

## 📁 Project Structure

```
MediQuery/
├── backend/
│   ├── main.py          # FastAPI server
│   ├── rag.py           # RAG pipeline implementation
│   ├── embeddings.py    # Document embedding generation
│   └── preprocess.py    # PDF preprocessing
├── frontend/
│   └── app.py           # Streamlit UI
├── data/
│   ├── pdfs/           # Source documents
│   ├── index/          # FAISS indexes
│   └── metadata.json   # Document metadata
└── requirements.txt
```

## 🛠️ API Reference

### Query Endpoint

```http
POST /query/
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The medical query text |
| `mode` | `string` | `quick`, `research`, or `quiz` |
| `num_questions` | `int` | Number of quiz questions (1-10) |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- BioBERT for biomedical embeddings
- Google Gemini API for text generation
- FastAPI and Streamlit communities

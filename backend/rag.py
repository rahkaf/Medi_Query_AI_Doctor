# mediquery/backend/rag.py
import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import torch
import logging
import os
import re
import dotenv

dotenv.load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, index_file="D:\\Company Tasks\\FInal Project\\MedQuery\\data\\index\\faiss.index", 
                 metadata_file="D:\\Company Tasks\\FInal Project\\MedQuery\\data\\metadata.json"):
        try:
            logger.info("Loading BioBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            logger.info("BioBERT loaded successfully.")
            
            logger.info("Loading Gemini API...")
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key and os.path.exists("models/config.json"):
                with open("models/config.json", "r", encoding='utf-8') as f:
                    config = json.load(f)
                    gemini_api_key = config.get("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment or config.")
            genai.configure(api_key=gemini_api_key)
            self.llm = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini API initialized.")
            
            logger.info("Loading FAISS index and metadata...")
            self.index = faiss.read_index(index_file)
            # Change the encoding to utf-8 when reading the metadata file
            with open(metadata_file, "r", encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"FAISS index and metadata loaded successfully. {len(self.metadata)} metadata entries.")
        except FileNotFoundError:
            logger.error("FAISS index or metadata file not found.")
            raise FileNotFoundError("FAISS index or metadata file not found. Run preprocess.py and embeddings.py first.")
        except Exception as e:
            logger.error(f"Error loading models or files: {str(e)}")
            raise Exception(f"Error loading models or files: {str(e)}")

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embedding.astype("float32")

    def retrieve_context(self, query_embedding, k=10):
        distances, indices = self.index.search(query_embedding, k)
        contexts = []
        logger.info(f"Retrieved indices: {indices[0].tolist()}")
        for idx in indices[0]:
            try:
                contexts.append({
                    "text": self.metadata[idx]["text"],
                    "source": self.metadata[idx]["source"],
                    "page": self.metadata[idx]["page"]
                })
                logger.info(f"Context {idx}: {self.metadata[idx]['text'][:50]}...")
            except IndexError:
                logger.warning(f"Index {idx} out of range in metadata.")
                continue
        logger.info(f"Retrieved {len(contexts)} contexts.")
        return contexts

    def generate_answer(self, query, contexts, mode, num_questions=5):
        if not contexts:
            logger.warning("No contexts retrieved for query.")
            return {"answer": "No relevant context found."}
        context_text = "\n".join([c["text"] for c in contexts])
        
        if mode == "quick":
            prompt = (
                "Using the context below, provide a concise and precise answer (100–200 words) to the question. "
                "Focus only on relevant details, avoid repetition, and be direct.\n"
                f"Question: {query}\n"
                f"Context:\n{context_text}"
            )
        elif mode == "research":
            prompt = (
                "Using the context below, provide a well-structured and informative answer (400–600 words) to the question. "
                "Write in a research-oriented tone. Include clear headings, bullet points, and cite relevant details. "
                "Ensure depth, clarity, and logical flow.\n"
                f"Question: {query}\n"
                f"Context:\n{context_text}"
            )
        else:  # quiz mode
            prompt = (
                f"Generate {num_questions} multiple-choice questions (4 options each, indicate correct answer) based on this context:\n"
                f"{context_text}\n"
                "Format each question as a form question with clear choices:\n"
                "Question: <question>\n"
                "Options:\n"
                "1. <option1>\n"
                "2. <option2>\n"
                "3. <option3>\n"
                "4. <option4>\n"
                "Correct: <number>"
            )

        try:
            response = self.llm.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2000,
                    "temperature": 0.7
                }
            )
            if mode in ["quick", "research"]:
                return {"answer": response.text}
            else:
                quiz_text = response.text
                questions = []
                pattern = r"Question:\s*(.*?)\nOptions:\n1\.\s*(.*?)\n2\.\s*(.*?)\n3\.\s*(.*?)\n4\.\s*(.*?)\nCorrect:\s*(\d+)"
                matches = re.findall(pattern, quiz_text, re.DOTALL)
                
                for i, match in enumerate(matches, 1):
                    question_text, opt1, opt2, opt3, opt4, correct = match
                    questions.append({
                        "question": question_text.strip(),
                        "options": [
                            opt1.strip(),
                            opt2.strip(),
                            opt3.strip(),
                            opt4.strip()
                        ],
                        "correct_answer": int(correct) - 1  # Converting to 0-based index
                    })
                
                if not questions:
                    logger.warning("No questions parsed from quiz response.")
                    return {"answer": "Failed to generate quiz questions."}
                return {"answer": {"questions": questions}}
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {"answer": f"Error generating content: {str(e)}"}

    def process(self, query, mode, num_questions=5):
        query_embedding = self.embed_query(query)
        contexts = self.retrieve_context(query_embedding)
        return self.generate_answer(query, contexts, mode, num_questions)

if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()
    query = "What are the symptoms of diabetes?"
    mode = "research"
    result = pipeline.process(query, mode)
    print(result)

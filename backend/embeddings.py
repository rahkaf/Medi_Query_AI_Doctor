import os
import json
import torch
import numpy as np
import faiss
import warnings
import logging
import gc
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

class ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]["text"], self.chunks[idx]["source"], self.chunks[idx]["page"]

def save_checkpoint(embeddings, metadata, index_file, metadata_file, checkpoint_num):
    """Save intermediate embeddings and metadata."""
    try:
        temp_index_file = f"{index_file}.checkpoint_{checkpoint_num}"
        temp_metadata_file = f"{metadata_file}.checkpoint_{checkpoint_num}"
        embeddings_np = np.array(embeddings).astype("float32")
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        faiss.write_index(index, temp_index_file)
        with open(temp_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved checkpoint {checkpoint_num} to {temp_index_file} and {temp_metadata_file}")
    except Exception as e:
        logger.error(f"Error saving checkpoint {checkpoint_num}: {str(e)}")
        raise

def load_checkpoint(index_file, metadata_file, checkpoint_num):
    """Load latest checkpoint if exists."""
    temp_index_file = f"{index_file}.checkpoint_{checkpoint_num}"
    temp_metadata_file = f"{metadata_file}.checkpoint_{checkpoint_num}"
    if os.path.exists(temp_index_file) and os.path.exists(temp_metadata_file):
        try:
            index = faiss.read_index(temp_index_file)
            with open(temp_metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded checkpoint {checkpoint_num} from {temp_index_file} and {temp_metadata_file}")
            return index, metadata, len(metadata)
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_num}: {str(e)}")
            return None, [], 0
    return None, [], 0

def generate_embeddings(
    chunks_file=CHUNKS_FILE,
    index_file=FAISS_INDEX_PATH,
    metadata_file=METADATA_FILE,
    batch_size=16
):
    logger.info("Loading BioBERT...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    except Exception as e:
        logger.error(f"Error loading BioBERT: {str(e)}")
        raise

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"BioBERT loaded on {device}.")

    logger.info(f"Loading chunks from {chunks_file}...")
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        logger.error(f"{chunks_file} not found.")
        raise

    if not chunks:
        logger.error("No chunks found in the input file.")
        return

    # Initialize empty lists for embeddings and metadata
    embeddings = []
    metadata = []

    dataset = ChunkDataset(chunks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_chunks = len(chunks)

    logger.info(f"Generating embeddings for {total_chunks} chunks...")
    
    # Use a single progress bar for all chunks
    with tqdm(total=total_chunks, desc="Generating embeddings") as pbar:
        for batch_texts, batch_sources, batch_pages in dataloader:
            try:
                # Clear memory before processing each batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get embeddings from the last hidden state
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                for text, source, page in zip(batch_texts, batch_sources, batch_pages):
                    text_value = str(text) if isinstance(text, (torch.Tensor, np.ndarray)) else text
                    source_value = str(source) if isinstance(source, (torch.Tensor, np.ndarray)) else source
                    page_value = int(page.item()) if isinstance(page, torch.Tensor) else int(page)
                    metadata.append({"text": text_value, "source": source_value, "page": page_value})
                
                # Update progress bar
                pbar.update(len(batch_texts))
                
                # Clear memory
                del inputs, outputs, batch_embeddings
                
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

    logger.info("Creating final FAISS index...")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        # Convert embeddings to numpy array and ensure correct shape/type
        embeddings_np = np.array(embeddings).astype('float32')
        
        if len(embeddings_np) == 0:
            raise ValueError("No embeddings generated")
            
        # Create and populate FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings_np)
        
        # Verify index is not empty
        if index.ntotal == 0:
            raise ValueError("Failed to add vectors to index")
            
        # Save the index
        faiss.write_index(index, index_file)
        
        # Save metadata with verification
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # Verify files were created
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not created at {index_file}")
            
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not created at {metadata_file}")
            
        logger.info(f"Successfully saved index with {index.ntotal} vectors to {index_file}")
        logger.info(f"Successfully saved metadata for {len(metadata)} chunks to {metadata_file}")
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        generate_embeddings()
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
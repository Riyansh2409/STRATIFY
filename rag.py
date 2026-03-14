"""
rag.py
======
Handles vector ingestion and RAG querying using processed_data JSONL files.
Backend: ChromaDB + SentenceTransformers.
LLM: Simple open-source model or fallback to a dummy chain for UI demonstration.
"""

import json
from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# Config
CHROMA_DB_DIR = str(Path(__file__).parent / "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fast, small embedding model

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, 
            embedding_function=self.embeddings
        )
        
    def load_processed_data(self, data_dir: str = "processed_data") -> list[Document]:
        """Loads train.jsonl and val.jsonl from the pipeline output into Langchain Documents."""
        documents = []
        data_path = Path(data_dir)
        
        for file_name in ["train.jsonl", "val.jsonl"]:
            file_path = data_path / file_name
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        # We extract the main text for vectorisation and keep the rest as metadata
                        text = data.pop("text")
                        documents.append(Document(page_content=text, metadata=data))
        return documents

    def ingest(self, data_dir: str = "processed_data"):
        """Embeds and saves processed pipeline chunks into ChromaDB."""
        logger.info("Loading processed JSONL data...")
        documents = self.load_processed_data(data_dir)
        
        if not documents:
            logger.warning("No documents found in processed_data! Run the pipeline first.")
            return False

        logger.info(f"Ingesting {len(documents)} document chunks into ChromaDB. This might take a minute...")
        
        # We delete the old collection to prevent duplicates on re-runs
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR, 
                embedding_function=self.embeddings
            )
        except Exception:
            pass # Collection might not exist yet
            
        self.vector_store.add_documents(documents)
        logger.info("Ingestion complete!")
        return True

    def query(self, question: str, k: int = 3) -> tuple[str, list]:
        """Runs a retrieval query, returning an answer and source documents."""
        # 1. Retrieve most relevant chunks
        docs = self.vector_store.similarity_search(question, k=k)
        
        if not docs:
            return "I couldn't find any relevant information in the uploaded documents.", []
            
        # 2. Augment / Generate
        # In a real production app, you would pass `context` to Llama-3 or OpenAI here.
        # Since we might not have a GPU/API key in this local demo space, we will 
        # simulate the LLM's capability by summarizing the exact retrieved chunks 
        # to show the UI functionality working. 
        
        context = "\n\n".join([f"Excerpt {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        answer = (
            "**[Simulated RAG Answer]**\n\n"
            "Based on the documents you uploaded, here is what I found:\n\n"
            f"{context}\n\n"
            "---\n*Note: To answer questions naturally like ChatGPT, you would connect an LLM "\
            "(like OpenAI or a local Llama-3 model) here. Currently, this shows the direct 'Retrieved' context.*"
        )
        
        return answer, [doc.metadata for doc in docs]

if __name__ == "__main__":
    # Quick CLI test
    rag = RAGPipeline()
    print("Ingesting data...")
    rag.ingest()
    print("Asking a question...")
    ans, sources = rag.query("What is machine learning?")
    print(ans)

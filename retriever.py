# retriever.py
import os
import pickle
from typing import List, Tuple, Dict
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

# model name for embeddings (light & good)
EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_dir="vectorstore", embed_model_name=EMBED_MODEL):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.index = None
        self.metadatas: List[Dict] = []
        self.docs: List[str] = []

    def extract_text_from_pdf(self, path: str) -> List[Tuple[int, str]]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
        return pages

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        text = text.replace("\n", " ").strip()
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            start = max(0, end - overlap)
            if end == len(text):
                break
        return chunks

    def index_pdfs(self, pdf_paths: List[str], chunk_size: int = 800, overlap: int = 200):
        self.docs = []
        self.metadatas = []
        embeddings = []
        for path in pdf_paths:
            pages = self.extract_text_from_pdf(path)
            for page_no, text in pages:
                if not text.strip(): 
                    continue
                chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for ci, chunk in enumerate(chunks):
                    self.docs.append(chunk)
                    self.metadatas.append({"source": os.path.basename(path), "page": page_no, "chunk": ci})
        if len(self.docs) == 0:
            raise ValueError("No text found in provided PDFs.")
        # compute embeddings in batches
        embeddings = self.embed_model.encode(self.docs, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        # persist
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)
        with open(os.path.join(self.index_dir, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)
        print(f"Indexed {len(self.docs)} chunks. Index saved to {self.index_dir}")

    def load_index(self):
        idx_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "metadatas.pkl")
        docs_path = os.path.join(self.index_dir, "docs.pkl")
        if not os.path.exists(idx_path):
            raise FileNotFoundError("Index not found. Run index_pdfs first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            self.metadatas = pickle.load(f)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        print(f"Loaded index with {len(self.docs)} chunks")

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.load_index()
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            results.append({"text": self.docs[idx], "meta": self.metadatas[idx]})
        return results

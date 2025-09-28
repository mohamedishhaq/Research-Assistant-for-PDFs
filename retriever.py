from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
import streamlit as st

@st.cache_resource
def get_query_engine(pdf_dir: str = "sample_papers"):
    """
    Load PDFs from a directory, build a vector index, 
    and return a RetrieverQueryEngine for semantic search.
    """
    # HuggingFace embeddings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load PDFs from directory
    docs = SimpleDirectoryReader(pdf_dir).load_data()

    # Create index
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    # Create retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    # Return query engine
    return RetrieverQueryEngine(retriever=retriever)

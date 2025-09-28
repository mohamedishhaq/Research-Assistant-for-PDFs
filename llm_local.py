# llm_local.py
from transformers import pipeline
import os

def get_generator(model_name: str = "google/flan-t5-small", device: int = -1, max_length:int=512):
    """
    device=-1 uses CPU; device=0 would use GPU if available.
    Returns a transformers pipeline object for text2text-generation.
    """
    # If you have limited RAM, consider "google/flan-t5-small" over base versions.
    gen = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=max_length
    )
    return gen

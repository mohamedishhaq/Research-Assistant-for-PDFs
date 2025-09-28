# llm_gemini.py
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def generate_answer_with_gemini(prompt: str, max_output_tokens: int = 512):
    """
    Use Gemini to produce the final answer. Prompt should include context and instructions.
    """
    if not API_KEY:
        return "[No Gemini key — generation disabled]"
    try:
        model = genai.GenerativeModel("gemini-1.5-mini")
        resp = model.generate_content(prompt, max_output_tokens=max_output_tokens)
        return resp.text
    except Exception as e:
        print("⚠️ Gemini generation error:", e)
        return "[Generation failed]"

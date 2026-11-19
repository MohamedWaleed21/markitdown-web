import base64
import io
import os
import tempfile
from urllib.parse import parse_qs, urlparse
import fitz  # PyMuPDF
import pytesseract
import requests
import streamlit as st
from PIL import Image
from markitdown import MarkItDown
from openai import OpenAI

st.set_page_config(
    page_title="MarkItDown AI Converter",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📄 File, Image & URL to Markdown")
st.caption("🤖 Built with AI • Powered by Microsoft MarkItDown & Streamlit — Convert documents, images, and URLs into clean, LLM-ready Markdown.")

with st.expander("💡 Why convert files to Markdown (.md) for AI?"):
    st.markdown(
        """
        * **Saves 30%–60% of AI Tokens:** Raw HTML, XML, and Office files contain heavy formatting tags that waste your token limit. Markdown strips out the clutter and keeps pure content.
        * **Preserves Structure & Tables:** Unlike plain copy-paste (which flattens tables and breaks headings), `.md` keeps titles (`#`), lists, and table columns intact so Gemini, ChatGPT, and Claude read your data accurately.
        * **Reduces AI Hallucinations:** Structured text helps LLMs distinguish between section headers, table rows, and body paragraphs, leading to more reliable answers and summaries.
        * **Universal AI Format:** Every modern AI assistant, coding agent, and RAG pipeline reads `.md` files natively as a lightweight plain-text format.
        """
    )

with st.expander("🚀 User Guide: Images (OCR vs. AI Vision) & Maximum Value"):
    st.markdown(
        """
        #### 🖼️ How Image Conversion Works
        * **Free Offline OCR (No API Key Needed):** Automatically extracts printed Arabic & English text from screenshots and scanned PDFs. *Note: Standard OCR only reads isolated text labels—it cannot interpret circuit schematics, wiring, or scientific plots.*
        * **AI Vision Mode (Open Sidebar `>` to add a Free Gemini Key):** Uses multimodal AI to actually "see" and describe diagrams, circuit schematics, charts, complex tables, and handwriting in structured Markdown. *(Tip: Choose a `flash` model for fast responses and higher free limits).*
        * **Direct Chat Shortcut:** If you are already inside an AI chat (like Gemini or ChatGPT) and only need to ask about a single image, upload the image directly into that chat. Use this converter when you want reusable `.md` notes or need to process multi-page documents.

        #### ⚡ How to Get Maximum Benefit
        * **Bypass Free Chat Upload Limits:** Convert multiple PDFs, Word docs, or Excel sheets into lightweight `.md` files and combine them cleanly into a single AI prompt.
        * **Keep Spreadsheets & Reports Accurate:** Convert `.xlsx` and `.docx` files before feeding them to an LLM so rows, columns, and formulas stay aligned.
        * **Turn YouTube & Articles into Study Notes:** Paste any webpage or YouTube link in the **Convert URL** tab to extract clean text and transcripts ready for summarization.
        """
    )

@st.cache_data(ttl=300, show_spinner=False)
def fetch_gemini_models(key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key.strip()}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text
            return [], f"HTTP {resp.status_code}: {err_msg}"

        data = resp.json()
        models = []
        for m in data.get("models", []):
            methods = m.get("supportedGenerationMethods", [])
            name = m.get("name", "").replace("models/", "")
            if "generateContent" in methods and "gemini" in name.lower():
                if not any(skip in name.lower() for skip in ["embedding", "aqa", "tts", "robotics", "image-generation", "2.5"]):
                    models.append(name)

        def model_priority(name: str):
            n = name.lower()
            if "3.8-flash" in n:
                return (0, n)
            if "3.6-flash" in n:
                return (1, n)
            if "flash" in n and "lite" not in n:
                return (2, n)
            if "flash" in n:
                return (3, n)
            return (4, n)

        models.sort(key=model_priority)
        return models, None
    except Exception as e:
        return [], str(e)

def convert_image_with_gemini(image_path: str, key: str, model_name: str) -> str:
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key.strip()}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Extract all text, tables, equations, and diagrams from this image and format the output as clean, structured Markdown. "
                            "Preserve the original language (Arabic or English), headings, lists, and tables accurately. "
                            "If the image contains a diagram, plot, or circuit schematic, describe its components, connections, and labels clearly in Markdown."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }
        ]
    }

    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        try:
            err_detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err_detail = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {err_detail}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned an empty response.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()

def convert_image_with_openai(image_path: str, key: str, model

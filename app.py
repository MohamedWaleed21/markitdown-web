import base64
import io
import os
import tempfile
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

def convert_image_with_openai(image_path: str, key: str, model_name: str) -> str:
    client = OpenAI(api_key=key.strip(), timeout=30.0, max_retries=1)
    md_vision = MarkItDown(llm_client=client, llm_model=model_name.strip())
    result = md_vision.convert(image_path)
    return getattr(result, "markdown", None) or result.text_content or ""

@st.cache_resource
def get_basic_converter():
    return MarkItDown()

def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang="ara+eng").strip()

def ocr_scanned_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    extracted_pages = []
    progress_bar = st.progress(0, text="Running OCR on scanned PDF pages...")

    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = ocr_image(img)
            if page_text:
                extracted_pages.append(f"## Page {i + 1}\n\n{page_text}")
            progress_bar.progress((i + 1) / total_pages, text=f"Reading page {i + 1} of {total_pages}...")
    finally:
        doc.close()
        progress_bar.empty()

    return "\n\n---\n\n".join(extracted_pages)

# Collapsible sidebar for AI Vision integration
with st.sidebar:
    st.header("⚙️ AI Vision Options")
    provider = st.selectbox(
        "AI Provider (Optional)",
        ["Google Gemini (Free Tier)", "OpenAI"]
    )
    api_key = st.text_input(
        f"{provider.split(' ')[0]} API Key",
        type="password",
        help="Paste your API key to use AI Vision for diagrams, equations, and complex images. Leave empty for free offline OCR."
    )

    selected_model = None
    if "Gemini" in provider:
        if api_key and api_key.strip():
            available_models, fetch_err = fetch_gemini_models(api_key)
            if fetch_err:
                st.error(f"Key check failed: {fetch_err}")
            elif available_models:
                st.success(f"✅ Connected! {len(available_models)} models available.")
                selected_model = st.selectbox("Available Gemini Models", available_models)
            else:
                st.warning("Connected, but no text/vision models were listed.")
        else:
            st.caption("Paste your free Google AI Studio key above to enable AI diagram & image understanding.")
    else:
        selected_model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])

    st.info("🔒 Privacy First: Your API key and files stay in temporary session memory and are deleted immediately after conversion.")

if "markdown_output" not in st.session_state:
    st.session_state.markdown_output = None
if "output_filename" not in st.session_state:
    st.session_state.output_filename = "converted.md"
if "warning_msg" not in st.session_state:
    st.session_state.warning_msg = None

tab_file, tab_url = st.tabs(["📁 Upload File or Image", "🌐 Convert URL"])

with tab_file:
    uploaded_file = st.file_uploader(
        "Choose a document or image",
        type=["pdf", "docx", "xlsx", "xls", "pptx", "png", "jpg", "jpeg", "webp", "html", "htm", "csv", "json", "xml", "zip", "txt", "msg"]
    )
    if uploaded_file is not None:
        if st.button("Convert File", type="primary", use_container_width=True):
            st.session_state.warning_msg = None
            st.session

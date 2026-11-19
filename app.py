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
    page_title="MarkItDown Converter",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📄 File, Image & URL to Markdown")
st.caption("Convert PDF, Word, Excel, PowerPoint, Images, Scanned PDFs, and URLs into clean, AI-ready Markdown.")

with st.expander("💡 Why convert files to Markdown (.md) for AI?"):
    st.markdown(
        """
        * **Saves 30%–60% of AI Tokens:** Raw HTML, XML, and Office files contain heavy formatting tags that waste your token limit. Markdown strips out the clutter and keeps pure content.
        * **Preserves Structure & Tables:** Unlike plain copy-paste (which flattens tables and breaks headings), `.md` keeps titles (`#`), lists, and table columns intact so Gemini, ChatGPT, and Claude read your data accurately.
        * **Reduces AI Hallucinations:** Structured text helps LLMs distinguish between section headers, table rows, and body paragraphs, leading to more reliable answers and summaries.
        * **Universal AI Format:** Every modern AI assistant, coding agent, and RAG pipeline reads `.md` files natively as a lightweight plain-text format.
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
                if not any(skip in name.lower() for skip in ["embedding", "aqa", "tts", "robotics", "image-generation"]):
                    models.append(name)

        models.sort(key=lambda x: (0 if "flash" in x else 1, x))
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
                            "If the image contains a diagram, plot, or circuit schematic, describe its structure and labels clearly in Markdown."
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
            st.caption("Paste your AIza... key above to load your available Gemini models automatically.")
    else:
        selected_model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])

    st.info("🔒 Your API key and files stay in temporary session memory and are never stored.")

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
            st.session_state.markdown_output = None
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            base_name = os.path.splitext(uploaded_file.name)[0]
            st.session_state.output_filename = f"{base_name}.md"
            md_basic = get_basic_converter()

            with st.spinner("Converting to Markdown..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Case 1: Image files
                    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                        text = ""
                        if api_key and api_key.strip() and selected_model:
                            try:
                                if "Gemini" in provider:
                                    text = convert_image_with_gemini(tmp_path, api_key, selected_model)
                                else:
                                    text = convert_image_with_openai(tmp_path, api_key, selected_model)
                            except Exception as vision_err:
                                st.session_state.warning_msg = (
                                    f"⚠️ AI Vision error ({vision_err}). Switched to free offline OCR automatically."
                                )
                                with Image.open(tmp_path) as img:
                                    text = ocr_image(img)
                        else:
                            with Image.open(tmp_path) as img:
                                text = ocr_image(img)

                    # Case 2: Documents & Archives
                    else:
                        result = md_basic.convert(tmp_path)
                        text = getattr(result, "markdown", None) or result.text_content or ""

                        # Fallback to OCR if PDF is scanned images (like CamScanner)
                        if not text.strip() and suffix == ".pdf":
                            text = ocr_scanned_pdf(tmp_path)

                    if text.strip():
                        st.session_state.markdown_output = text
                    elif not st.session_state.warning_msg:
                        st.session_state.warning_msg = "⚠️ No readable text was detected in this file or image."
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

with tab_url:
    url_input = st.text_input("Enter a webpage or YouTube URL", placeholder="https://example.com/article")
    if st.button("Convert URL", type="primary", use_container_width=True) and url_input.strip():
        st.session_state.warning_msg = None
        st.session_state.markdown_output = None
        st.session_state.output_filename = "webpage.md"
        md_basic = get_basic_converter()

        with st.spinner("Fetching and converting URL..."):
            try:
                result = md_basic.convert(url_input.strip())
                text = getattr(result, "markdown", None) or result.text_content or ""
                if text.strip():
                    st.session_state.markdown_output = text
                else:
                    st.session_state.warning_msg = "⚠️ No readable text content could be extracted from this URL."
            except Exception as e:
                st.error(f"URL conversion failed: {e}")

if st.session_state.warning_msg:
    st.warning(st.session_state.warning_msg)

if st.session_state.markdown_output:
    st.divider()
    markdown_output = st.session_state.markdown_output
    char_count = len(markdown_output)
    est_tokens = max(1, char_count // 4)

    col1, col2 = st.columns(2)
    col1.metric("Characters", f"{char_count:,}")
    col2.metric("Estimated AI Tokens", f"~{est_tokens:,}")

    st.download_button(
        label="⬇️ Download .md File",
        data=markdown_output,
        file_name=st.session_state.output_filename,
        mime="text/markdown",
        use_container_width=True
    )

    st.text_area("Markdown Preview", markdown_output, height=350)

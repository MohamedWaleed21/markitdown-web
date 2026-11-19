import io
import os
import tempfile
import fitz  # PyMuPDF
import pytesseract
import streamlit as st
from PIL import Image
from markitdown import MarkItDown
from openai import OpenAI

st.set_page_config(page_title="MarkItDown Converter", page_icon="📄", layout="centered")

st.title("📄 File, Image & URL to Markdown")
st.caption("Convert PDF, Word, Excel, PowerPoint, Images, Scanned PDFs, and URLs into clean, AI-ready Markdown.")

with st.expander("💡 Why convert files to Markdown (.md) for AI?"):
    st.markdown(
        """
        * **Saves 30%–60% of AI Tokens:** Raw HTML, XML, and Office files contain heavy formatting tags that waste your token limit. Markdown strips out the clutter and keeps pure content.
        * **Preserves Structure & Tables:** Unlike plain copy-paste (which flattens tables and breaks headings), `.md` keeps titles (`#`), lists, and table columns intact so ChatGPT, Gemini, and Claude read your data accurately.
        * **Reduces AI Hallucinations:** Structured text helps LLMs distinguish between section headers, table rows, and body paragraphs, leading to more reliable answers and summaries.
        * **Universal AI Format:** Every modern AI assistant, coding agent, and RAG pipeline reads `.md` files natively as a lightweight plain-text format.
        """
    )

# Optional AI Vision key in sidebar
with st.sidebar:
    st.header("⚙️ Advanced Options")
    api_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Add your own key only if you want GPT-4o-mini Vision to describe diagrams and complex images. Free OCR works without a key."
    )
    st.info("🔒 Files are processed temporarily in memory and deleted immediately after conversion.")

@st.cache_resource
def get_basic_converter():
    return MarkItDown()

def get_converter(key: str):
    if key and key.strip():
        client = OpenAI(api_key=key.strip())
        return MarkItDown(llm_client=client, llm_model="gpt-4o-mini")
    return get_basic_converter()

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
            md = get_converter(api_key)

            with st.spinner("Converting to Markdown..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Case 1: Image files
                    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                        if api_key and api_key.strip():
                            result = md.convert(tmp_path)
                            text = getattr(result, "markdown", None) or result.text_content or ""
                        else:
                            with Image.open(tmp_path) as img:
                                text = ocr_image(img)

                    # Case 2: Documents & Archives
                    else:
                        result = md.convert(tmp_path)
                        text = getattr(result, "markdown", None) or result.text_content or ""

                        # Fallback to OCR if PDF is scanned images (like CamScanner)
                        if not text.strip() and suffix == ".pdf":
                            text = ocr_scanned_pdf(tmp_path)

                    if text.strip():
                        st.session_state.markdown_output = text
                    else:
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
        md = get_converter(api_key)

        with st.spinner("Fetching and converting URL..."):
            try:
                result = md.convert(url_input.strip())
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

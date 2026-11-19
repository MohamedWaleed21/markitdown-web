import io
import os
import tempfile
import fitz  # PyMuPDF
import pytesseract
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

# Collapsible sidebar for AI Vision integration (Gemini or OpenAI)
with st.sidebar:
    st.header("⚙️ AI Vision Options")
    provider = st.selectbox(
        "AI Provider (Optional)",
        ["Google Gemini (Free Tier)", "OpenAI"]
    )

    if "Gemini" in provider:
        model_choice = st.selectbox(
            "Model Name",
            ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "Custom..."]
        )
    else:
        model_choice = st.selectbox(
            "Model Name",
            ["gpt-4o-mini", "gpt-4o", "Custom..."]
        )

    if model_choice == "Custom...":
        selected_model = st.text_input("Enter exact model ID", value="gemini-2.0-flash")
    else:
        selected_model = model_choice

    api_key = st.text_input(
        f"{provider.split(' ')[0]} API Key",
        type="password",
        help="Paste your key to use AI Vision for describing diagrams, equations, and complex images. Leave empty to use free offline OCR."
    )
    st.info("🔒 Your API key and files stay in temporary session memory and are never stored.")

@st.cache_resource
def get_basic_converter():
    return MarkItDown()

def build_vision_converter(selected_provider: str, key: str, model_name: str):
    if "Gemini" in selected_provider:
        client = OpenAI(
            api_key=key.strip(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        client = OpenAI(api_key=key.strip())
    return MarkItDown(llm_client=client, llm_model=model_name.strip())

def convert_image_with_vision(image_path: str, selected_provider: str, key: str, chosen_model: str) -> str:
    models_to_try = [chosen_model.strip()]
    if "Gemini" in selected_provider:
        for fallback in ["gemini-2.0-flash", "gemini-2.5-flash"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

    last_err = None
    for m in models_to_try:
        try:
            md_vision = build_vision_converter(selected_provider, key, m)
            result = md_vision.convert(image_path)
            return getattr(result, "markdown", None) or result.text_content or ""
        except Exception as e:
            last_err = e
            if "404" in str(e) or "NotFound" in str(e):
                continue
            raise e
    raise last_err

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
            md_basic = get_basic_converter()

            with st.spinner("Converting to Markdown..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Case 1: Image files
                    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                        text = ""
                        if api_key and api_key.strip():
                            try:
                                text = convert_image_with_vision(tmp_path, provider, api_key, selected_model)
                            except Exception as vision_err:
                                st.session_state.warning_msg = (
                                    f"⚠️ AI Vision returned an error ({vision_err}), so offline OCR was used automatically instead."
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

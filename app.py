import os
import tempfile
import streamlit as st # type: ignore
import markitdown

st.set_page_config(page_title="MarkItDown Converter", page_icon="📄", layout="centered")

st.title("📄 File & URL to Markdown Converter")
st.caption("Convert PDF, Word, Excel, PowerPoint, HTML, CSV, JSON, ZIP, and URLs into clean, LLM-ready Markdown.")

@st.cache_resource
def get_converter():
    return markitdown.MarkItDown()

md = get_converter()

tab_file, tab_url = st.tabs(["📁 Upload File", "🌐 Convert URL"])

markdown_output = None
output_filename = "converted.md"

with tab_file:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "xls", "pptx", "html", "htm", "csv", "json", "xml", "zip", "txt", "msg"]
    )
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        base_name = os.path.splitext(uploaded_file.name)[0]
        output_filename = f"{base_name}.md"

        with st.spinner("Converting file to Markdown..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                result = md.convert(tmp_path)
                markdown_output = getattr(result, "markdown", None) or result.text_content
            except Exception as e:
                st.error(f"Conversion failed: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

with tab_url:
    url_input = st.text_input("Enter a webpage or YouTube URL", placeholder="https://example.com/article")
    if st.button("Convert URL") and url_input.strip():
        output_filename = "webpage.md"
        with st.spinner("Fetching and converting URL..."):
            try:
                result = md.convert(url_input.strip())
                markdown_output = getattr(result, "markdown", None) or result.text_content
            except Exception as e:
                st.error(f"URL conversion failed: {e}")

if markdown_output:
    st.divider()
    char_count = len(markdown_output)
    est_tokens = max(1, char_count // 4)

    col1, col2 = st.columns(2)
    col1.metric("Characters", f"{char_count:,}")
    col2.metric("Estimated AI Tokens", f"~{est_tokens:,}")

    st.download_button(
        label="⬇️ Download .md File",
        data=markdown_output,
        file_name=output_filename,
        mime="text/markdown",
        use_container_width=True
    )

    st.text_area("Markdown Preview", markdown_output, height=350)
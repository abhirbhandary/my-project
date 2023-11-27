import base64

import streamlit as st
from streamlit_chat import message
from streamlit_javascript import st_javascript

st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")
st.header("pdf-GPT")

def clear_submit():
    st.session_state["submit"] = False

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def displayPDF(upl_file, width):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

def displayPDFpage(upl_file, page_nr):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_nr}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="file_qa_api_key", type="password"
    )

    model_name = st.radio(
        "Select the model", ("gpt-3.5-turbo", "text-davinci-003", "gpt-4")
    )

    approach = st.radio(
        "Choose an approach", ("1", "2"))

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf"],
        help="Only PDF files are supported",
        on_change=clear_submit,
    )


col1, col2 = st.columns(spec=[2, 1], gap="small")


if uploaded_file:
    with col1:
        ui_width = st_javascript("window.innerWidth")
        displayPDF(uploaded_file, ui_width -10)

    with col2:
        question = st.text_input(
            "Ask something about the article",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
            on_change=clear_submit,
        )
        



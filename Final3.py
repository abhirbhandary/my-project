import base64
from streamlit_chat import message
from streamlit_javascript import st_javascript


from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader 
#to take from PDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():

    load_dotenv()
    st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")
    st.header("pdf-GPT")



    def clear_submit():
        st.session_state["submit"] = False


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
        pdf = st.file_uploader(
        "Upload file",
        type=["pdf"],
        help="Only PDF files are supported",
        on_change=clear_submit,
        )
        

    col1, col2 = st.columns(spec=[2, 1], gap="small")


    if pdf:
        with col1:
            ui_width = st_javascript("window.innerWidth")
            displayPDF(pdf, ui_width -10)

        with col2:
            if pdf is not None: 
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        
                # split into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                
                chunks = text_splitter.split_text(text)
      
                # create embeddings
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
      
                # show user input
                user_question = st.text_input("Ask something about the article",
                placeholder="Can you give me a short summary?",
                disabled=not pdf,
                on_change=clear_submit,
                )

                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
        
                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb: #To see how much i have been using the model
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)
                    
                    st.write(response)
if __name__ == '__main__':
    main()





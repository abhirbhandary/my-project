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


import random
import itertools
from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatOpenAI


import os

def main():

    load_dotenv()
    st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")
    st.header("pdf-GPT")

    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }   
        </style>
        """,
        unsafe_allow_html=True,
    )

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

        openai_api_key= st.text_input(
        "OpenAI API Key", key="file_qa_api_key", type="password"
        )

        os.environ["OPENAI_API_KEY"] = openai_api_key

        questions = st.slider("Number of questions", 0, 10, 3) 

        search = st.button('Search')

        #Intialize session state
        if "search_state" not in st.session_state:
            st.session_state.search_state = False

        #st.sidebar.image("img/logo1.jpg", use_column_width=True)
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
                st.info("`Reading doc ...`")
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                st.write("Documents uploaded and processed.")

                # split into chunks
                st.info("`Splitting doc ...`")
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                
                chunks = text_splitter.split_text(text)

                # Display the number of text chunks
                num_chunks = len(chunks)
                st.write(f"Number of text chunks: {num_chunks}")
      
                # create embeddings
                embeddings = OpenAIEmbeddings()

               

                knowledge_base = FAISS.from_texts(chunks, embeddings)


                if search or st.session_state.search_state:
                    st.session_state.search_state = True
                    st.info("`Generating sample questions ...`")
                    n = len(text)
                    chunk = 50
                    starting_indices = [random.randint(0, n-chunk) for _ in range(questions)]
                    sub_sequences = [text[i:i+chunk] for i in starting_indices]
                    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
                    eval_set = []
                    for i, b in enumerate(sub_sequences):
                        try:
                            qa = chain.run(b)
                            eval_set.append(qa)
                            st.write("Creating Question:",i+1)
                        except:
                            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
                    eval_set_full = list(itertools.chain.from_iterable(eval_set))

                    st.session_state.eval_set = eval_set_full

                    for i, qa_pair in enumerate(st.session_state.eval_set):
                        st.sidebar.markdown(
                        f"""
                        <div class="css-card">
                        <span class="card-tag">Question {i + 1}</span>
                        <p style="font-size: 12px;">{qa_pair['question']}</p>
                        <p style="font-size: 12px;">{qa_pair['answer']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                        )
                    st.session_state.search_state = False
                    # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
                    # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>

                st.write("Ready to answer questions.")
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

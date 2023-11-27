import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

#One PDF
from PyPDF2 import PdfReader 
#to take from PDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

#CSV
from langchain.agents import create_csv_agent


from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="MIA QA",
        page_icon="🤖"
    )


def main():
    init()

    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Ask your PDF 💬")

    tab1, tab2, tab3 = st.tabs(["PDF", "CSV", "Database"])
    
    with tab1:

        # upload file
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        
        # extract the text
        #Single PDF
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
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
        
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb: #To see how much i have been using the model
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
           
                st.write(response)

    with tab2:
        
        
        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        if csv_file is not None:

            agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

            user_question = st.text_input("Ask a question about your CSV: ")

            if user_question is not None and user_question != "":
                with st.spinner(text="In progress..."):
                    st.write(agent.run(user_question))



    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your own ChatGPT 🤖", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))

        # display message history
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages[1:]):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=str(i) + '_user')
            else:
                message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()
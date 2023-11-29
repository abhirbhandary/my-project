import streamlit as st
from PyPDF2 import PdfReader 
#to take from PDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT



import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub

def main():
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    HUGGINGFACEHUB_API_TOKEN = st.text_input(
        "Hugging Face Access Tokens", key="api_key", type="password"
        )

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
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
      #embeddings = OpenAIEmbeddings()

      embeddings = HuggingFaceEmbeddings()


      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        #google/flan-t5-xxl

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.write(response)
    

if __name__ == '__main__':
    main()
## Prompt Template
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain


import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title("Celebrity Search Results")
input_text=st.text_input("Search the topic you want")


#Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    #template="Tell me about celebrity {name}"
    template="Tell me {name} age"
)


## OpenAI LLMS - Temperature is how much control the agent has. Its range is from 0 to 1.
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True)


# pydantic error install a lower version and also for lower version of OPEN AI use [pip install openai==0.28][pip install pydantic==1.10.9]

if input_text: #giving my input to LLM
    st.write(chain.run(input_text))

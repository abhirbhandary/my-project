## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title("Langchain Demo with OpenAI AI")
input_text=st.text_input("Search the topic you want")

## OpenAI LLMS - Temperature is how much control the agent has. Its range is from 0 to 1.
llm=OpenAI(temperature=0.8)


# pydantic error install a lower version and also for lower version of OPEN AI use [pip install openai==0.28][pip install pydantic==1.10.9]

if input_text: #giving my input to LLM
    st.write(llm(input_text))

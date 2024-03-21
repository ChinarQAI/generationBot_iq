# importing necessary libraries
import os
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
import streamlit as st
# import boto3
from data_extractor import *


# API keys
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = "654b8bee0c754d8a87403bc292cffb8c2296190e6bf88f86f96a7f4d67d118cf"


# File upload
uploaded_file = extract_text("file injest")

temp = st.sidebar.selectbox("choose the option for how creative your chatbot should be",("creative", "balanced", "imitative"))

# three tempertature modes
if temp == "creative":
    llm = OpenAI(temperature=0.9)
if temp == "balanced":
    llm = OpenAI(temperature=0.6)
if temp == "imitative":
    llm = OpenAI(temperature=0.2)



memory = ConversationBufferMemory()

tools = load_tools(['serpapi', 'llm-math'], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory = memory,
    verbose = False
)
col1, col2 = st.columns(2)

prompt = "you are a chatbot who writes response to the below Request to Proposal documents"+"\n"+uploaded_file

if prompt:
    result = agent.run(prompt)
    st.write(result)

else:
    None
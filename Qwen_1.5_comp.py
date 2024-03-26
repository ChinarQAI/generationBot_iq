import torch
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import prompts
from langchain.chains import RetrievalQA
from data_extractor import *
from Rag import *

device = 'cpu'  # switch this option to cuda to use the GPU

model = AutoModelForCausalLM.from_pretrained('Qwen_1.5', device_map='auto') # loading the pretrained model here

tokenizer = AutoTokenizer.from_pretrained('Qwen_1.5', device_map='auto')
# loading the tokenizer from the model itself

# batching the prompts

prompt = extract_text("file injest")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
    )


llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

prompt = "what is a polygon ?"
messages = [
        {'role': 'system', 'content': 'you are helpful chatbot'},
        {'role': 'user','content': prompt}
    ]

qa = RetrievalQA.from_chain_type(llm=llm, chain_type='refine', retriever=retriever, return_source_documents=False)

result = qa.invoke({'query': 'what a sample response for a request for proposal with budget of $30,000 for machine building?'})
print(result['result'])




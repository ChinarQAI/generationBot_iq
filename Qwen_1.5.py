import torch
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from data_extractor import *
from Rag import *

device = 'cpu'  # switch this option to cuda to use the GPU

model = AutoModelForCausalLM.from_pretrained('Qwen_1.5', device_map='auto') # loading the pretrained model here

tokenizer = AutoTokenizer.from_pretrained('Qwen_1.5', device_map='auto')
# loading the tokenizer from the model itself

# batching the prompts

prompt = extract_text("file injest")

# prompt template

def chatbot_pipeline(prompt):
    messages = [
        {'role': 'system', 'content': 'you are chatbot who writes the response for given request for proposal documents'},
        {'role': 'user','content': prompt}
    ]


    # tokenize the input prompt
    text = tokenizer.apply_chat_template(messages,
                                        tokenize=False,
                                        add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors='pt').to(device)


    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = 500
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# pipelines = pipeline(
#     model=model,
#     tokenizer=tokenizer
#     )

# print(pipeline.generate(messages = [
#         {'role': 'system', 'content': 'you are chatbot who writes the response for given request for proposal documents'},
#         {'role': 'user','content': 'create a sample rpf'}
#     ]))



# pipe = HuggingFacePipeline(pipeline==pipelines)



# print(chatbot('create a sample request for proposal'))


# qa = RetrievalQA.from_chain_type(llm=pipe, chain_type="refine", retriever=retriever, return_source_documents=False)

# results = qa.run(messages)
# print(results)
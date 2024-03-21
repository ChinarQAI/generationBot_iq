from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers 
import torch


# model = "tinyllama1.1_gguf"
model = AutoModelForCausalLM.from_pretrained("tinyllama1.1_gguf")
tokenizer = AutoTokenizer.from_pretrained("tinyllama1.1_gguf")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


prompt = 'what is a polygon'

message = [
    {
        'role':'system',
        'content': 'You are a friendly chatbot who always responds in a style of a pirate'
    },
    {'role':'user',
     'content': prompt}
]

message = tokenizer.apply_chat_template(message)

sequences = pipeline(
    message,
    do_sample=True,
    top_k=50,
    top_p = 0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=60,
    temperature=0.1
)


print(sequences)
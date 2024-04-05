# This repo contains code for RFP Response Generation Bot.

## The file structure for the above is defined below.


|-> Qwen_1.5          This directory contains all the config and tokenizer files with model file itself.

|-> file injest       This directory contains the RFP document which is gonna be passed as a query in the prompt.

|-> sample_dataset    This directory contains sample RFP documents with there responses.

|-> tinyllama1.1_gguf This directory contains all all the config and tokenizer files with model file itself.


### AIM:
##### The Response bot aims to generate the response for the given RFP(request for proposal document)

The repository contains Three models as of now.
1. Tinyllama 1.1
2. Qwen 1.5
3. gpt 3.5     Running inference using API.

#### Please Intall all the requirements before using the requirements.txt file.
<pip install -r requirements.txt>


The above Model directories doesn't have the saved model files beacuse of the large size of the model files
But every model directory has a readme.md file which has link to the file to be downloaded.
Please go through it before running any inference.





### The Bot is under development


#### Iteration 1.

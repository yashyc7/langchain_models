from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os 
"""by default this program will download the model in the c drive i wanted to store the model in d drive then change the environment path to the d drive """

os.environ['HF_HOME'] = 'D:/Huggingface_cache'


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100},
) #it will download and run model 

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("what is the capital of india? ")

print(result)

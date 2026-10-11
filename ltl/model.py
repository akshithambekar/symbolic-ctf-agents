# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/model.py
import os
from langchain_openai import ChatOpenAI
import yaml

def llm_init(gpt_api,model_name="gpt-4o",
             temperature=0.0,
             max_tokens=4000,
             frequency_penalty=0,
            presence_penalty=0):

    llm = ChatOpenAI(model_name=model_name,
                     temperature=temperature,
                     max_tokens=max_tokens,
                     #max_completion_tokens = max_tokens,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                     openai_api_key=gpt_api)

    return llm

if __name__ == "__main__":
    
    with open("/config/config_openai.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    openai_key = config['openai_api_key']
    
    LLM = llm_init(gpt_api=openai_key)
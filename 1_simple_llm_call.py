from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatMistralAI()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of India?"})

print(result)
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

chain1 = prompt1 | model | parser

chain2 = (
    RunnableLambda(lambda text: {"text": text})
    | prompt2
    | model
    | parser
)

result1 = chain1.invoke({"topic": "Artificial Intelligence in India"})

result2 = chain2.invoke(result1)

print(result2)
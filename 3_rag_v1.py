import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'RAG APP' 

# Load API key from .env
load_dotenv()


# 1) Load PDF
PDF_PATH = "islr.pdf"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


# 2) Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)


# 3) Create Mistral embeddings + FAISS index
emb = MistralAIEmbeddings(
    model="mistral-embed"
)

vs = FAISS.from_documents(splits, emb)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not found in the context, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


# 5) Mistral LLM
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)


def format_docs(docs):
    return "\n\n".join(
        d.page_content for d in docs
    )


parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


chain = parallel | prompt | llm | StrOutputParser()


# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

q = input("\nQ: ")

ans = chain.invoke(q.strip())

print("\nA:", ans)
import os
from dotenv import load_dotenv

from langsmith import traceable

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
os.environ['LANGCHAIN_PROJECT'] = 'RAG App' 


# Load environment variables
load_dotenv()


PDF_PATH = "islr.pdf"


# ---------- traced setup steps ----------

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits):

    # Mistral embeddings
    emb = MistralAIEmbeddings(
        model="mistral-embed"
    )

    # Create FAISS vector store
    vs = FAISS.from_documents(
        splits,
        emb
    )

    return vs


# Trace the complete setup
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vs = build_vectorstore(splits)

    return vs


# ---------- Mistral LLM ----------

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)


# ---------- Prompt ----------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If not found, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


# ---------- Format documents ----------

def format_docs(docs):
    return "\n\n".join(
        d.page_content for d in docs
    )


# ---------- Build vector store ----------

vectorstore = setup_pipeline(PDF_PATH)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ---------- RAG chain ----------

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})


chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# ---------- Run query ----------

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

q = input("\nQ: ").strip()


# LangSmith configuration
config = {
    "run_name": "pdf_rag_query"
}


# Invoke chain
ans = chain.invoke(
    q,
    config=config
)

print("\nA:", ans)
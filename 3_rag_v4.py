import os
import json
import hashlib
from pathlib import Path

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


load_dotenv()

PDF_PATH = "islr.pdf"

INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# ----------------- helpers (traced) -----------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):

    emb = MistralAIEmbeddings(
        model=embed_model_name
    )

    return FAISS.from_documents(
        splits,
        emb
    )


# ----------------- cache key / fingerprint -----------------

def _file_fingerprint(path: str) -> dict:

    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime)
    }


def _index_key(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str
) -> str:

    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }

    return hashlib.sha256(
        json.dumps(
            meta,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


# ----------------- load existing index -----------------

@traceable(name="load_index", tags=["index"])
def load_index_run(
    index_dir: Path,
    embed_model_name: str
):

    emb = MistralAIEmbeddings(
        model=embed_model_name
    )

    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )


# ----------------- build new index -----------------

@traceable(name="build_index", tags=["index"])
def build_index_run(
    pdf_path: str,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str
):

    # Load PDF
    docs = load_pdf(pdf_path)

    # Split documents
    splits = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create vector store
    vs = build_vectorstore(
        splits,
        embed_model_name
    )

    # Save FAISS index
    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    vs.save_local(
        str(index_dir)
    )

    # Save metadata
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "pdf_path": os.path.abspath(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embed_model_name,
            },
            indent=2
        )
    )

    return vs


# ----------------- dispatcher -----------------

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "mistral-embed",
    force_rebuild: bool = False,
):

    key = _index_key(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name
    )

    index_dir = INDEX_ROOT / key

    cache_hit = (
        index_dir.exists()
        and not force_rebuild
    )

    if cache_hit:

        print("Loading existing FAISS index...")

        return load_index_run(
            index_dir,
            embed_model_name
        )

    else:

        print("Building new FAISS index...")

        return build_index_run(
            pdf_path,
            index_dir,
            chunk_size,
            chunk_overlap,
            embed_model_name
        )


# ----------------- Mistral model -----------------

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)


# ----------------- prompt -----------------

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


def format_docs(docs):

    return "\n\n".join(
        d.page_content
        for d in docs
    )


# ----------------- setup pipeline -----------------

@traceable(
    name="setup_pipeline",
    tags=["setup"]
)
def setup_pipeline(
    pdf_path: str,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name="mistral-embed",
    force_rebuild=False
):

    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )


# ----------------- complete RAG run -----------------

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "mistral-embed",
    force_rebuild: bool = False,
):

    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name,
        force_rebuild
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

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

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {
                "k": 4
            }
        }
    )


# ----------------- CLI -----------------

if __name__ == "__main__":

    print(
        "PDF RAG ready. Ask a question "
        "(or Ctrl+C to exit)."
    )

    q = input("\nQ: ").strip()

    ans = setup_pipeline_and_query(
        PDF_PATH,
        q
    )

    print("\nA:", ans)
"""Load html from files, clean up, split, ingest into Weaviate."""

import re
from parse import langchain_docs_extractor
from typing import List

from bs4 import BeautifulSoup, SoupStrainer

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders.directory import DirectoryLoader

from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain.vectorstores.faiss import FAISS


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",  # type: ignore
        "language": html.get("lang", "") if html else "",  # type: ignore
        **meta,
    }


def load_langchain_docs():
    print(f"Loading docs from LangChain site")
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    print(f"Loading docs from LangChain API documentation")
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def load_local_docs(path):
    print(f"Loading local clinical docs from {path}")
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
        recursive=True,
    )
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} documents")
    return raw_documents


def load_huggingface_nlp_course_docs():
    print(f"Loading sites from HuggingFace NLP course")
    loader = RecursiveUrlLoader(
        url="https://huggingface.co/learn/nlp-course/en/",
        max_depth=3,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    )
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} documents")
    return raw_documents


def get_embeddings_model() -> Embeddings:
    # using default model: sentence-transformers/all-mpnet-base-v2
    return HuggingFaceEmbeddings()


def ingest_langchain_docs(
    docs: List[Document],
    vectorstore_name: str = "langchain_index",
    save_local: bool = True,
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(docs)

    # Add metadata fields if they don't exist.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    if save_local:
        embedding = get_embeddings_model()
        vectorstore = FAISS.from_documents(docs_transformed, embedding)
        vectorstore.save_local(f"index/{vectorstore_name}")

    return docs_transformed


def ingest_local_docs(
    docs: List[Document],
    vectorstore_name: str = "clinical_index",
    save_local: bool = True,
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(docs)

    # Add metadata fields if they don't exist.
    for doc in docs_transformed:
        if "title" not in doc.metadata and "source" in doc.metadata:
            # Use the filename as the title if it doesn't exist.
            doc.metadata["title"] = doc.metadata["source"].split("/")[-1]
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    if save_local:
        embedding = get_embeddings_model()
        vectorstore = FAISS.from_documents(docs_transformed, embedding)
        vectorstore.save_local(f"index/{vectorstore_name}")

    return docs_transformed


if __name__ == "__main__":
    path = "docs"
    docs = load_local_docs(path)
    ingest_local_docs(docs, "clinical_index")
    # docs = load_langchain_docs() + load_api_docs()
    # ingest_docs(docs, "langchain_index")

    # docs = load_huggingface_nlp_course_docs()
    # ingest_langchain_docs(docs, "huggingface_index")
    # path = "docs/Essentials of Anatomy and Physiology ( PDFDrive ).pdf"
    # docs = PyPDFLoader(path).load()
    # ingest_local_docs(docs, "clinical_textbook_index")

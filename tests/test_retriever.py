import os
import random
from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_random_docs_from_patient_name(patient_name, frequency):
    directory = Path(__file__).parent.parent / "docs" / patient_name
    files = [file for file in os.listdir(directory) if file.split(".")[-1] == "txt"]
    if frequency > len(files):
        raise ValueError
    random_files = random.sample(files, frequency)

    contents = []
    for file in random_files:
        path = os.path.join(directory, file)
        with open(path, "r", encoding="utf-8") as file:
            contents.append(file.read())
    return contents


def get_docs_from_patient_names(patient_names):
    docs = []
    for name, freq in patient_names:
        docs += [
            Document(page_content=content, metadata={"title": name})
            for content in get_random_docs_from_patient_name(name, freq)
        ]
    return docs


patient_names = [("fake_patient1", 10), ("fake_patient2", 9), ("fake_patient3", 8)]
retriever = BM25Retriever.from_documents(get_docs_from_patient_names(patient_names))
if __name__ == "__main__":
    from vectorstore import get_vectorstore
    from langchain_community.vectorstores.faiss import FAISS
    from ingest import get_embeddings_model

    num_docs = sum([x[1] for x in patient_names])
    docs = get_docs_from_patient_names(patient_names)
    print("num docs", num_docs)
    query = "What is Fake Patient1 diagnosed with?"
    # vectorstore = FAISS.from_documents(docs, get_embeddings_model())
    # docs = vectorstore.search(query, search_type="similarity")[: num_docs // 2]
    retriever = BM25Retriever.from_documents(docs)
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc.metadata)

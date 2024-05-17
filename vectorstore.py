from ingest import get_embeddings_model
from pathlib import Path
from langchain.vectorstores.faiss import FAISS


def get_vectorstore(index_name: str):
    embeddings = get_embeddings_model()

    index_name = "clinical_index"

    vectorstore = FAISS.load_local(
        str(Path(__file__).parent / f"index/{index_name}"),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

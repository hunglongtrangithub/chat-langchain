from langchain.vectorstores.faiss import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever


class CustomRetriever:
    def __init__(self, vectorstore: FAISS, num_docs: int = 5):
        self.num_docs = num_docs
        self.semantic_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=dict(
                k=self.num_docs * 2,
            ),
        )

    def get_relevant_documents(self, query: str):
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)
        bm25_retriever = BM25Retriever.from_documents(semantic_docs, k=self.num_docs)
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        return bm25_docs

    async def aget_relevant_documents(self, query: str):
        semantic_docs = await self.semantic_retriever.aget_relevant_documents(query)
        bm25_retriever = BM25Retriever.from_documents(semantic_docs, k=self.num_docs)
        bm25_docs = await bm25_retriever.aget_relevant_documents(query)
        return bm25_docs


if __name__ == "__main__":
    from vectorstore import get_vectorstore

    vectorstore = get_vectorstore("clinical_index")
    retriever = CustomRetriever(vectorstore)
    query = "What is Fake Patient1 diagnosed with?"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.metadata)

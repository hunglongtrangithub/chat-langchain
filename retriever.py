from pydantic import BaseModel
import meilisearch
from meilisearch.errors import MeilisearchApiError
from typing import List
from logger_config import get_logger
from _index import EMBEDDER_NAME, INDEX_NAME, SEARCH_API_KEY, MEILI_API_URL

logger = get_logger(__name__)


class Document(BaseModel):
    page_content: str
    title: str
    source: str


class CustomRetriever:
    def __init__(self, num_docs: int = 5, semantic_ratio: float = 0.1):
        self.index_name = INDEX_NAME
        self.embedder_name = EMBEDDER_NAME
        self.num_docs = num_docs
        self.semantic_ratio = semantic_ratio
        self.client = meilisearch.Client(
            url=MEILI_API_URL,
            api_key=SEARCH_API_KEY,
        )
        self.index = self.client.index(self.index_name)

    def get_relevant_documents(self, query: str) -> List[Document]:
        opt_params = {
            "hybrid": {
                "semanticRatio": self.semantic_ratio,
                "embedder": self.embedder_name,
            },
            "limit": self.num_docs,
        }
        try:
            result = self.index.search(query, opt_params)
        except MeilisearchApiError as e:
            logger.error(f"Error searching documents: {e}")
            return []

        docs = result["hits"]
        logger.info(f"Found {len(docs)} documents.")
        return [
            Document(
                page_content=doc["page_content"],
                title=doc["title"],
                source=doc["source"],
            )
            for doc in docs
        ]

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)


if __name__ == "__main__":
    retriever = CustomRetriever(num_docs=5)
    query = "What is Fake Patient1 diagnosed with?"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.title)

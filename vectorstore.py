from ingest import get_embeddings_model
from pathlib import Path
from langchain.vectorstores.faiss import FAISS

from logger_config import get_logger

logger = get_logger(__name__)

embeddings = get_embeddings_model()

vectorstore_name = "faiss_index"

vectorstore = FAISS.load_local(
    str(Path(__file__).parent / f"index/{vectorstore_name}"),
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
logger.info("FAISS index loaded: ", vectorstore_name)

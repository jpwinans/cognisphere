import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data_ingestion import WebIngestor
from text_processing import DocumentIndexer

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
SERVICE_HOST = os.getenv("SERVICE_HOST")
URL_INDEXER_SERVICE_PORT = os.getenv("URL_INDEXER_SERVICE_PORT")
INDEX_COLLECTION_NAME = os.getenv("INDEX_COLLECTION_NAME")

url_index_service = FastAPI()
web_ingestor = WebIngestor()
document_indexer = DocumentIndexer(
    collection_name=INDEX_COLLECTION_NAME, host=QDRANT_HOST, port=QDRANT_PORT
)


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    category: str = "webpage"
    traverse: bool = False


@url_index_service.post("/scrape_urls")
async def scrape_urls(request: ScrapeUrlsRequest):
    try:
        documents = web_ingestor.ingest_documents(
            urls=request.urls, category=request.category, traverse=request.traverse
        )
        document_ids = document_indexer.index_documents(documents)
        return {"document_ids": document_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(url_index_service, host=SERVICE_HOST, port=URL_INDEXER_SERVICE_PORT)

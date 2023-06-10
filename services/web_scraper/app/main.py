import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.data_ingestion import WebIngestor
from shared.text_processing import DocumentIndexer

load_dotenv()

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = os.environ["QDRANT_PORT"]
SERVICE_HOST = os.environ["SERVICE_HOST"]
URL_INDEXER_SERVICE_PORT = int(os.environ["URL_INDEXER_SERVICE_PORT"])
INDEX_COLLECTION_NAME = os.environ["INDEX_COLLECTION_NAME"]

app = FastAPI()
web_ingestor = WebIngestor()
document_indexer = DocumentIndexer(
    collection_name=INDEX_COLLECTION_NAME, host=QDRANT_HOST, port=QDRANT_PORT
)


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    category: str = "webpage"
    traverse: bool = False


@app.post("/scrape_urls")
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

    uvicorn.run("app", host=SERVICE_HOST, port=URL_INDEXER_SERVICE_PORT, reload=True)

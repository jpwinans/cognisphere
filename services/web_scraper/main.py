import os
import traceback
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from common.data_ingestion.web_ingestor import WebIngestor
from common.text_processing.document_indexer import DocumentIndexer
from models import ScrapeUrlsRequest

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


@app.post("/scrape_urls")
async def scrape_urls(request: ScrapeUrlsRequest):
    try:
        documents = web_ingestor.ingest_documents(
            urls=request.urls, categories=request.categories, traverse=request.traverse
        )
        document_ids = document_indexer.index_documents(documents)
        return {"document_ids": document_ids}
    except Exception as e:
        stacktrace = traceback.format_exc()
        detail = f"{str(e)}\n\nStacktrace:\n{stacktrace}"
        raise HTTPException(status_code=500, detail=detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=URL_INDEXER_SERVICE_PORT)

from text_processing import DocumentProcessor
from data_ingestion import TextIngestor
document = TextIngestor().ingest_document('transcript.txt')
document.sections[0].keywords = DocumentProcessor().extract_keywords(document.sections[0].section_text) 
document.sections[0].named_entities = DocumentProcessor().extract_named_entities(document.sections[0].section_text)
document = DocumentProcessor().summarize_document(document, with_sections=True)
# import pickle
# with open('my_doc.pkl', 'rb') as f:
#     document = pickle.load(f)

import pymongo
from schema import Document, Section
from llama_index import (
    GPTVectorStoreIndex,
    GPTListIndex,
    GPTSimpleKeywordTableIndex,
)
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SimpleNodeParser
index_name="documents"
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["cognisphere"]
index_name = index_name
collection = db[index_name]
docstore = MongoDocumentStore.from_uri(
    uri="mongodb://localhost:27017/", db_name="cognisphere"
)
index_store = MongoIndexStore.from_uri(
    uri="mongodb://localhost:27017/", db_name="cognisphere"
)
storage_context = StorageContext.from_defaults(
    docstore=docstore, index_store=index_store
)
document_data = document.dict()
document_data["vector_representation"] = document.vector_representation
sections_data = [section.dict() for section in document.sections]
result = collection.insert_one({**document_data, "sections": sections_data})
document_id = str(result.inserted_id)


llama_document = document.to_llama_format()
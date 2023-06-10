import os
from typing import List

import openai
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

from text_processing import Preprocessor
from utils import open_file

openai.api_key = os.getenv("OPENAI_API_KEY")


class TextIngestor:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 0

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP
        )

    def extract_text(self, file_path: str, category: str) -> List[Document]:
        """
        Extracts text from txt file at the given path.
        """
        loader = TextLoader(file_path, encoding="utf8")
        documents = loader.load()

        chunks = self.text_splitter.split_documents(documents)

        # Populate the metadata
        source = os.path.basename(file_path)
        for i, doc in enumerate(chunks):
            doc.page_content = doc.page_content.strip()
            doc.metadata = {"source": source, "category": category, "doc_index": i}

        return chunks

    def process_text(self, text: str) -> str:
        processed_text = Preprocessor().clean_text(text)
        processed_text = Preprocessor().normalize_text(processed_text)
        return processed_text

    def ingest_documents(
        self, file_paths: List[str], category: str = "document"
    ) -> List[Document]:
        """
        Ingests a document from a file at the given path.
        """
        documents = []
        for file_path in file_paths:
            documents.extend(self.extract_text(file_path, category))

        return documents

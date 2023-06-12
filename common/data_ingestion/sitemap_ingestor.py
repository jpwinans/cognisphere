from .text_ingestor import TextIngestor
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import TokenTextSplitter
from schema import Document
from typing import List


class SiteMapIngestor(TextIngestor):

    def __init__(self):
        super().__init__()

    def extract_text(self, sitemap_url: str) -> List[Document]:
        """
        Extracts text from all pages in the sitemap at the given URL.
        """
        # Load the text from the documents in the sitemap and process
        loader = SitemapLoader(sitemap_url)
        documents = loader.load()

        # Process each document
        processed_documents = []
        for document in documents:
            # Divide the text into sections of 2048 tokens
            text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=0)
            texts = text_splitter.split_documents([document])

            # Populate the Section and Document data structures
            sections = []
            for i, text in enumerate(texts):
                section = self.create_section(i, text.page_content, document.metadata['loc'])
                sections.append(section)

            processed_document = self.create_document(document.metadata['loc'], sections)
            processed_documents.append(processed_document)

        return processed_documents

    def ingest_document(self, sitemap_url: str) -> List[Document]:
        """
        Ingests documents from a sitemap at the given URL.
        """
        documents = self.extract_text(sitemap_url)
        return documents

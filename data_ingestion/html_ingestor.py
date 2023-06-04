from .text_ingestor import TextIngestor
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import TokenTextSplitter
from schema import Document


class HTMLIngestor(TextIngestor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url: str) -> Document:
        """
        Extracts text from HTML document at the given URL.
        """
        # Load the text from the HTML document and process
        loader = BSHTMLLoader(url)
        documents = loader.load()

        # Divide the text into sections of 2048 tokens
        text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Populate the Section and Document data structures
        document_id = url
        sections = []
        for i, text in enumerate(texts):
            section = self.create_section(i, text.page_content, document_id)
            sections.append(section)

        document = self.create_document(url, sections)

        return document

    def ingest_document(self, url: str) -> Document:
        """
        Ingests a document from an HTML document at the given URL.
        """
        document = self.extract_text(url)
        return document

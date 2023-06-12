from .text_ingestor import TextIngestor
from langchain.document_loaders import PyMuPDFLoader
from google.cloud import documentai_v1beta3 as documentai
from typing import List
from schema import Document, Section


class PDFIngestor(TextIngestor):
    def __init__(self):
        super().__init__()
        self.client = documentai.DocumentProcessorServiceClient()

    def extract_text(self, file_path: str) -> Document:
        """Extracts text from PDF at the given path."""

        # Try parsing the PDF with PyMuPDF
        try:
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            sections = self.pages_to_sections(pages)
            document = self.create_document(file_path, sections)
            return document
        except Exception as e:
            print(f"Unable to parse PDF file: {e}")

        # Detect if the PDF contains scanned images using Google Cloud Document AI
        pdf_type = self.detect_document_type(file_path)
        if pdf_type == "IMAGE_BASED_PDF":
            print("Performing OCR on scanned PDF...")
            sections = self.ocr_scanned_pdf(file_path)
            document = self.create_document(file_path, sections)
            return document
        else:
            raise ValueError("Unable to extract text from PDF file.")

    def pages_to_sections(self, pages):
        sections = []
        for page in pages:
            section = Section(section_text=page.get_text(), metadata=page.metadata)
            sections.append(section)
        return sections

    def ocr_scanned_pdf(self, file_path: str) -> List[Section]:
        """Performs OCR on the scanned PDF at the given path and returns extracted text."""

        # Initialize the Document AI client
        client = documentai.DocumentProcessorServiceClient()

        # Read the file into memory
        with open(file_path, "rb") as pdf_file:
            content = pdf_file.read()

        # Build the Document AI OCR request
        ocr_request = {
            "name": "OCR",
            "document": {"content": content, "mime_type": "application/pdf"},
            "features": {"text_extraction": {}},
        }

        # Perform OCR on the PDF
        response = client.async_batch_process_documents(request=ocr_request)
        result = response.responses[0]

        # Extract text from the OCR response
        sections = []
        for page in result.document.pages:
            page_text = ""
            for par in page.paragraphs:
                page_text += par.text + " "

            section = self.create_section(
                index_id=page.page_number,
                text=page_text,
                document_id=file_path,
                metadata={
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                },
            )
            sections.append(section)

        return sections

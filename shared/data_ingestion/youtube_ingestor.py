from .text_ingestor import TextIngestor
from langchain.document_loaders import YoutubeLoader
from schema import Document
from typing import List


class YoutubeIngestor(TextIngestor):
    def __init__(self):
        super().__init__()

    def extract_text(self, youtube_url: str, category: str) -> List[Document]:
        """
        Extracts text from YouTube video at the given URL.
        """
        # Load the transcript from the YouTube video and process
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        documents = loader.load()

        chunks = self.text_splitter.split_documents(documents)

        # Populate the metadata
        for i, doc in enumerate(chunks):
            doc.page_content = self.process_text(doc.page_content)
            doc.metadata["category"] = category
            doc.metadata["doc_index"] = i

        return chunks

    def ingest_documents(
        self, youtube_urls: List[str], category: str = "video"
    ) -> List[Document]:
        """
        Ingests a document from a YouTube video at the given URL.
        """
        documents = []
        for youtube_url in youtube_urls:
            documents.extend(self.extract_text(youtube_url, category))

        return documents


# Provide Main Functionality to test the module
if __name__ == "__main__":
    docs = YoutubeIngestor().ingest_documents(
        ["https://www.youtube.com/watch?v=tds4_3LeaVI"], category="wisdom"
    )
    for doc in docs:
        print(doc.page_content)

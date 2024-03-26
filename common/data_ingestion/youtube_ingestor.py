import json
from typing import List
import requests
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from .text_ingestor import TextIngestor
from common.text_processing.document_indexer import DocumentIndexer

class YoutubeIngestor(TextIngestor):
    def __init__(self):
        super().__init__()
        self.indexer = DocumentIndexer()

    def extract_text(self, youtube_url: str, categories: List[str]) -> List[Document]:
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
            doc.metadata["categories"] = categories
            doc.metadata["doc_index"] = i

        return chunks

    def ingest_documents(
        self, youtube_urls: List[str], categories: List[str] = ["youtube"], with_index: bool = False
    ) -> List[Document]:
        """
        Ingests a document from a YouTube video at the given URL.
        """
        documents = []
        for youtube_url in youtube_urls:
            try:
                chunks = self.extract_text(youtube_url, categories)
                documents.extend(chunks)
                if with_index:
                    id = self.indexer.index_documents(chunks)
                    print(f"Indexed {youtube_url} with ID {id}")
            except Exception as e:
                print(f"Failed to ingest {youtube_url}: {e}")
                continue

        return documents

    def ingest_playlist(
        self, playlist_id: str, categories: List[str] = ["youtube"], with_index: bool = False
    ) -> List[Document]:
        """
        Ingests all documents from a YouTube playlist given its ID.
        """
        # Load YouTube Data API key from file
        with open("/Users/jameswinans/.gcp_api_key", "r") as file:
            api_key = file.read().strip()

        # Get the playlist items (limit 50)
        request_url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId={playlist_id}&key={api_key}"
        response = requests.get(request_url)
        playlist_data = json.loads(response.text)

        # Extract video URLs
        youtube_urls = [
            f"https://www.youtube.com/watch?v={item['snippet']['resourceId']['videoId']}"
            for item in playlist_data["items"]
        ]

        # Ingest documents
        return self.ingest_documents(youtube_urls, categories, with_index)


# Provide Main Functionality to test the module
if __name__ == "__main__":
    docs = YoutubeIngestor().ingest_documents(
        ["https://www.youtube.com/watch?v=8R7QOJgGyIQ"],
        categories=["youtube", "writing"],
    )
    for doc in docs:
        print(doc.page_content)

from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

from .text_ingestor import TextIngestor
from .web_scrapers import TrafilaturaWebReader


def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    base_url = url
    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".html"):
            full_url = urljoin(base_url, href)
            links.append(full_url)
    return links


class WebIngestor(TextIngestor):
    def __init__(self):
        super().__init__()
        self.reader = TrafilaturaWebReader()

    def ingest_documents(
        self, urls: List[str], categories: List[str] = ["document"], traverse: bool = False
    ) -> List[Document]:
        documents = []
        for url in urls:
            if traverse:
                links = extract_links(url)
            else:
                links = [url]
            docs = self.reader.load_langchain_documents(urls=links)
            for i, doc in enumerate(docs):
                chunks = self.text_splitter.split_documents(docs)
                # Populate the metadata
                for j, doc in enumerate(chunks):
                    doc.page_content = doc.page_content.strip()
                    doc.metadata = {
                        "source": links[i],
                        "categories": categories,
                        "doc_index": j,
                    }
                    documents.append(doc)
        return documents


# Provide Main Functionality to test the module
if __name__ == "__main__":
    docs = WebIngestor().ingest_documents(
        ["https://python.langchain.com/en/latest/getting_started/concepts.html"],
        categories=["api-documentation"], traverse=False
    )
    for doc in docs:
        print(doc.page_content)

import requests
url = "https://www.rigpawiki.org/index.php?title=Dzogchen"
from typing import List
service_host = "localhost"
service_port = 8000
urls_to_scrape: List[str] = [url]
request_data = {
     "urls": urls_to_scrape,
     "categories": ["dzogchen", "buddhism","webpage"],
     "traverse": False,
}
response = requests.post(
  f"http://{service_host}:{service_port}/scrape_urls", json=request_data
)

from common.text_processing.document_indexer import DocumentIndexer
indexer = DocumentIndexer()
client = indexer.get_client()
from qdrant_client.http import models
scroll_filter=models.Filter(
  must=[
    models.FieldCondition(
      key="metadata.author",
      match=models.MatchValue(value="Lama Lena Teachings"),
    )
  ]
)
docs, point = client.scroll(limit=1000, scroll_filter=scroll_filter, collection_name='documents')
sorted = sorted(docs, key=lambda x: x.payload['metadata']['doc_index'])
texts = []
for doc in sorted:
    texts.append(doc.payload['page_content'])

page_text = '\n'.join(texts)
from common.text_processing.text_tiler import TextTiler
tiles = TextTiler().get_tiles(page_text)

from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models as qc_models
from qdrant_client.conversions import common_types as types

EMBEDDING_SIZE = 1536


class DocumentIndexer:
    def __init__(self, collection_name: str = "documents", host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

    def get_vector_store(self) -> Qdrant:
        """Returns the vector store."""
        return self.vector_store

    def get_client(self) -> QdrantClient:
        """Returns the client."""
        return self.client

    def index_documents(self, documents: List[Document]) -> str:
        """Indexes the given Document in Qdrant and returns the document ID."""
        try:
            document_ids = self.vector_store.add_documents(documents)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                self.create_collection(self.collection_name, documents[0])
                document_ids = self.vector_store.add_documents(documents)
        return document_ids

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Performs a vector similarity search on the document index using the given query
        vector. Returns matching Documents."""
        documents = self.vector_store.similarity_search(query, k=k)
        return documents

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: int = 0.5
    ) -> List[Document]:
        """Maximal marginal relevance optimizes for similarity to query AND diversity among
        selected documents."""
        documents = self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        return documents

    def create_collection(self, collection_name: str, document: Document) -> None:
        """Creates a collection in Qdrant."""

        # Just do a single quick embedding to get vector size
        partial_embeddings = self.embeddings.embed_documents(
            document.page_content[:100]
        )
        vector_size = len(partial_embeddings[0])

        collection_name = collection_name
        distance_func = "COSINE"

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qc_models.VectorParams(
                size=vector_size,
                distance=qc_models.Distance[distance_func],
            ),
        )

    def find_by_categories(
        self, categories: List[str], k: int = 5,
        offset: Optional[types.PointId] = None
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Performs a search on the document index using the given category.

        Returns:
            A pair of (List of points) and (optional offset for the next scroll request).
        """

        # create the filter for the search iterating over each category to create the filter
        conditions = [
            qc_models.FieldCondition(
                key="metadata.categories", match=qc_models.MatchValue(value=category)
            )
            for category in categories
        ]
        conditions.append(
            qc_models.FieldCondition(
                key="metadata.doc_index", match=qc_models.MatchValue(value=0)
            )
        )
        filter = qc_models.Filter(must=conditions)

        documents, point_id = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter,
            limit=k,
            offset=offset,
        )

        return documents, point_id
    
    def find_by_source(
            self, source: str, k: int = 5, offset: Optional[types.PointId] = None
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Perform a search to return all documents from a given source.

        Returns:
            A pair of (List of points) and (optional offset for the next scroll request).
        """

        # create the filter for the search source
        conditions = [
            qc_models.FieldCondition(
                key="metadata.source", match=qc_models.MatchValue(value=source)
            )
        ]
        filter = qc_models.Filter(must=conditions)

        documents, point_id = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter,
            limit=k,
            offset=offset,
        )

        # sort documents by doc.payload['metadata']['doc_index']
        documents = sorted(documents, key=lambda doc: doc.payload['metadata']['doc_index'])

        return documents, point_id
    
    def get_content_by_source(
            self, source: str
    ) -> str:
        """Perform search of documents from a given source and return the content."""

        # call find_by_source to get the documents, then extract the page_content
        documents, _ = self.find_by_source(source=source, k=1000)
        page_content = [doc.payload['page_content'] for doc in documents]

        return "/n".join(page_content)

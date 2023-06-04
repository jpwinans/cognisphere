import pymongo
from llama_index import (
    GPTVectorStoreIndex,
    GPTListIndex,
    GPTSimpleKeywordTableIndex,
)
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext


class Interrogator:
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "cognisphere"

    def __init__(self, index_name="documents"):
        self.client = pymongo.MongoClient(self.MONGO_URI)
        self.db = self.client[self.DB_NAME]
        self.collection = self.db[index_name]
        self.storage_context = StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(
                uri=self.MONGO_URI, db_name=self.DB_NAME
            ),
            index_store=MongoIndexStore.from_uri(
                uri=self.MONGO_URI, db_name=self.DB_NAME
            ),
        )

    def initialize_indexes(self, nodes):
        self.list_index = GPTListIndex(nodes, storage_context=self.storage_context)
        self.vector_index = GPTVectorStoreIndex(
            nodes, storage_context=self.storage_context
        )
        self.keyword_table_index = GPTSimpleKeywordTableIndex(
            nodes, storage_context=self.storage_context
        )

    def load_all_nodes_from_mongodb(self):
        docstore = MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME)
        nodes = list(docstore.docs.values())
        self.initialize_indexes(nodes)

    def search_query(self, query: str):
        vector_response = self.vector_index.as_query_engine(
            retriever_mode="embedding"
        ).query(query)
        return vector_response

    def search_summarize(self, query: str):
        list_response = self.list_index.as_query_engine().query(query)
        return list_response

    # def scratch():
    #     import tiktoken

    #     enc = tiktoken.get_encoding("gpt2")
    #     tokenizer = lambda text: enc.encode(text, allowed_special={"<|endoftext|>"})
    #     from llama_index.embeddings.openai import OpenAIEmbedding

    #     embed_model = OpenAIEmbedding()
    #     embed_model._tokenizer = tokenizer
    #     node_parser = SimpleNodeParser(
    #         text_splitter=TokenTextSplitter(tokenizer=tokenizer)
    #     )
    #     service_context = ServiceContext.from_defaults(
    #         embed_model=embed_model,
    #         node_parser=node_parser,
    #         llm_predictor=llm_predictor,
    #         prompt_helper=prompt_helper,
    #     )
    #     nodes = node_parser.get_nodes_from_documents(documents)

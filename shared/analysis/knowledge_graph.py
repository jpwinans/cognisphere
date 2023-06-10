import pymongo
from schema import Document


class KnowledgeGraph:
    def __init__(self, index_name="documents"):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["cognisphere"]
        self.index_name = index_name
        self.collection = self.db[self.index_name]
        
    def build_graph(self):
        """Builds the knowledge graph from the indexed document data."""
        
        # Construct the graph nodes from documents
        document_nodes = self._documents_to_nodes()
        
        # Find connections between documents based on similarities and references 
        # (in summaries, keywords, named entities, citations, etc.)
        edge_specs = self._find_document_relationships() 
        
        # Add all nodes and edges to the MongoDB knowledge graph 
        self.collection.insert_many(document_nodes + edge_specs) 

    def update_graph(self, new_documents):
        """Updates the knowledge graph by adding just the given new_documents 
        and any edges that connect them to existing nodes."""
        
        # Get IDs of all existing documents already in the knowledge graph
        existing_ids = [d["_id"] for d in self.collection.find({"labels": "Document"})]
        
        # Filter to only new documents not already in the graph
        new_documents = [d for d in new_documents if str(d.id) not in existing_ids]
        
        # Add new document nodes 
        new_nodes = self._documents_to_nodes(new_documents)
        self.collection.insert_many(new_nodes)
        
        # Find edges between new and existing documents 
        edge_specs = self._find_connections(new_documents)
        self.collection.insert_many(edge_specs)

    def query_graph(self, query):
        """Executes a query on the knowledge graph. The `query` should specify which
        nodes, edges, paths, etc. to retrieve from the graph."""

        return

    def _documents_to_nodes(self):
        """Converts documents into node objects for the knowledge graph."""
        pipeline = [
            {"$project": {
                "id": 1, 
                "titles": "$title",
                "summaries": "$summary",
                "keywords": "$keywords",
                "named_entities": "$named_entities"
            }}
        ]

        documents = list(self.collection.aggregate(pipeline))

        # Construct node objects from documents 
        nodes = [{
            "id": str(d["id"]),
            "labels": ["Document"],
            "properties": {
                "titles": d["titles"],
                "summaries": d["summaries"],
                "keywords": d["keywords"],
                "named_entities": d["named_entities"]
            }
        } for d in documents]

        return nodes

    def _find_document_relationships(self):
        """Determines connections between documents and constructs edge objects to represent
        the relationships in the knowledge graph."""

        return

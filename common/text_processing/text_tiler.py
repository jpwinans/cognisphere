from typing import List
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np


class TextTiler:
    def __init__(self, sentence_window_size: int = 7):
        self.sentence_window_size = sentence_window_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text to prepare it for tiling. This includes cleaning and normalizing
        the text, and converting it into an array of sentences.
        """
        sentences = sent_tokenize(text)
        return sentences

    def compute_semantic_vectors(self, sentences: List[str]) -> List[List[float]]:
        """
        Compute the semantic vectors for each sentence using BERT. Each semantic vector
        represents the meaning of a sentence as a vector of weights over concepts from
        a large knowledge base.
        """
        semantic_vectors = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            outputs = self.model(**inputs)
            semantic_vectors.append(
                outputs.last_hidden_state[0].mean(dim=0).detach().numpy()
            )
        return semantic_vectors

    def compute_similarity_matrix(
        self, semantic_vectors: List[List[float]]
    ) -> List[List[float]]:
        """
        Compute a similarity matrix that represents the semantic similarity between each
        pair of sentences. Only compute similarity between each sentence and the K subsequent ones.
        """
        similarity_matrix = np.zeros((len(semantic_vectors), len(semantic_vectors)))
        for i in range(len(semantic_vectors)):
            for j in range(
                i + 1, min(i + self.sentence_window_size, len(semantic_vectors))
            ):
                similarity_matrix[i][j] = cosine_similarity(
                    [semantic_vectors[i]], [semantic_vectors[j]]
                )[0][0]
        return similarity_matrix

    def segment_into_tiles(
        self, similarity_matrix: List[List[float]], sentences: List[str]
    ) -> List[List[str]]:
        """
        Segment the sentences into tiles based on the similarity matrix. Each tile should
        contain sentences that are semantically similar to each other. Ensure that the final tiles are coherent.
        """
        G = nx.Graph()
        for i in range(len(sentences)):
            for j in range(i + 1, min(i + self.sentence_window_size, len(sentences))):
                G.add_edge(i, j, weight=similarity_matrix[i][j])
        partition = community_louvain.best_partition(G)
        tiles = [[] for _ in range(max(partition.values()) + 1)]
        for i, tile_id in partition.items():
            tiles[tile_id].append(sentences[i])

        # Post-processing step to ensure that the final tiles are coherent
        tiles = [tile for tile in tiles if len(tile) > 1]
        return tiles

    def join_tiles_sentences(
        self, tiles: List[List[str]], sentences: List[str]
    ) -> List[str]:
        """
        Join the sentences in each tile into a single string. Ensure that the tiles are in the same order as the original text.
        """
        # Create a dictionary where the key is the index of the first sentence of each tile in the original text, and the value is the tile
        tiles_dict = {sentences.index(tile[0]): tile for tile in tiles}

        # Sort the dictionary by key to ensure that the tiles are in the correct order
        tiles_dict = dict(sorted(tiles_dict.items()))

        # Join the sentences in each tile into a single string
        tiles_sentences = []
        for tile in tiles_dict.values():
            tile_sentences = " ".join(tile)
            tiles_sentences.append(tile_sentences)
        return tiles_sentences

    def get_tiles(self, text: str) -> List[str]:
        """
        Given a large block of text, output an array of tiles, tiled with multi-sentence-level
        tiling using BERT and Louvain method.
        """
        sentences = self.preprocess_text(text)
        semantic_vectors = self.compute_semantic_vectors(sentences)
        similarity_matrix = self.compute_similarity_matrix(semantic_vectors)
        tiles = self.segment_into_tiles(similarity_matrix, sentences)
        tiles = self.join_tiles_sentences(tiles, sentences)
        return tiles

from typing import List

import nltk
import numpy as np
from gensim import corpora, models
from gensim.utils import simple_preprocess
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics.pairwise import cosine_similarity

from common.text_processing.preprocessor import Preprocessor


class SimpleTextTiler:
    def __init__(
        self,
        sentence_window_size: int = 5,
        boundary_threshold: float = 0.005,
        min_tile_length: int = 500,
    ):
        self.sentence_window_size = sentence_window_size
        self.boundary_threshold = boundary_threshold
        self.min_tile_length = min_tile_length
        self.preprocessor = Preprocessor()

    def preprocess_text(self, sentences: List[str]) -> List[str]:
        """
        Preprocess the text to prepare it for tiling. This includes cleaning and normalizing
        the text, and converting it into an array of sentences.
        """
        cleaned_sentences = [
            self.preprocessor.clean_text(sentence) for sentence in sentences
        ]
        return cleaned_sentences

    def compute_semantic_vectors(self, sentences: List[str]) -> List[List[float]]:
        """
        Compute the semantic vectors for each sentence using ESA. Each semantic vector
        represents the meaning of a sentence as a vector of weights over concepts from
        a large knowledge base.
        """
        # Tokenize the sentences
        texts = [simple_preprocess(sentence) for sentence in sentences]

        # Create a dictionary from the tokenized texts
        dictionary = corpora.Dictionary(texts)

        # Convert the tokenized texts into a document-term matrix (Bag of Words)
        BoW_corpus = [dictionary.doc2bow(text, allow_update=True) for text in texts]

        # Initialize a TF-IDF model from the corpus
        tfidf = models.TfidfModel(BoW_corpus)

        # Compute the TF-IDF weights for each sentence
        tfidf_weights = tfidf[BoW_corpus]

        # Convert the weights into a list of lists format
        num_terms = len(dictionary)
        semantic_vectors = [
            [next((freq for id, freq in doc if id == i), 0) for i in range(num_terms)]
            for doc in tfidf_weights
        ]

        return semantic_vectors

    def compute_similarity_matrix(
        self, semantic_vectors: List[List[float]]
    ) -> List[List[float]]:
        """
        Compute a similarity matrix that represents the semantic similarity between each
        pair of sentences.
        """
        similarity_matrix = cosine_similarity(semantic_vectors)

        return similarity_matrix

    def join_tiles_sentences(self, tiles: List[List[str]]) -> List[str]:
        """
        Join the sentences in each tile into a single string.
        """
        tiles_sentences = []
        for tile in tiles:
            tile_sentences = " ".join(tile)
            tiles_sentences.append(tile_sentences)
        merged_tiles = self.merge_short_tiles(tiles_sentences)
        return merged_tiles

    def merge_short_tiles(self, tiles: List[str]) -> List[str]:
        merged_tiles = []
        i = 0
        while i < len(tiles):
            current_tile = tiles[i]
            if len(current_tile) < self.min_tile_length and i < len(tiles) - 1:
                next_tile = tiles[i + 1]
                merged_tile = current_tile + next_tile
                if len(merged_tile) < self.min_tile_length:
                    tiles.pop(i + 1)
                    tiles[i] = merged_tile
                    continue
            merged_tiles.append(tiles[i])
            i += 1

        return merged_tiles

    def segment_into_tiles(
        self, similarity_matrix: List[List[float]], sentences: List[str]
    ) -> List[str]:
        """
        Segment the sentences into tiles based on the similarity matrix. Each tile should
        contain sentences that are semantically similar to each other.
        """
        # Compute the average similarity of each sentence with all other sentences
        avg_similarities = np.mean(similarity_matrix, axis=1)

        # Compute the local average similarity within the sliding window at each point
        local_averages = uniform_filter1d(
            avg_similarities, size=self.sentence_window_size
        )

        # Identify the boundaries as points where the local average changes significantly
        boundaries = np.where(np.diff(local_averages) > self.boundary_threshold)[0]

        # Segment the sentences into tiles based on the boundaries
        tiles = []
        start = 0
        for end in boundaries:
            tiles.append(sentences[start:end])
            start = end
        tiles.append(sentences[start:])

        return tiles

    def get_tiles(self, text: str) -> List[str]:
        """
        Given a large block of text, output an array of tiles, tiled with multi-sentence-level
        tiling using Explicit Semantic Analysis (ESA).
        """
        sentences = nltk.sent_tokenize(text)
        cleaned_sentences = self.preprocess_text(sentences)
        semantic_vectors = self.compute_semantic_vectors(cleaned_sentences)
        similarity_matrix = self.compute_similarity_matrix(semantic_vectors)
        tiles = self.segment_into_tiles(similarity_matrix, sentences)
        tiles = self.join_tiles_sentences(tiles)
        return tiles

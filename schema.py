import llama_index
from pydantic import BaseModel
from typing import List
from typing import Optional


class Section(BaseModel):
    section_id: str
    section_text: str
    vector_representation: Optional[List[float]]
    keywords: Optional[List[str]]
    named_entities: Optional[List[str]]
    summary: Optional[str]
    sentiment: Optional[float]
    document_id: str

    def to_llama_format(self):
        """Converts the CogniSphere's Section into a llama's Document format."""
        extra_info = {
            "section_id": self.section_id or "",
            "document_id": self.document_id or "",
            "summary": self.summary or "",
            "sentiment": self.sentiment or "",
            "keywords": ", ".join(self.keywords) if self.keywords else "",
            "named_entities": ", ".join(self.named_entities)
            if self.named_entities
            else "",
        }

        return llama_index.Document(
            text=self.section_text or "",
            doc_id=f"{self.document_id}-{self.section_id}"
            if self.document_id and self.section_id
            else "",
            extra_info=extra_info,
            embedding=self.vector_representation or [],
        )


class Document(BaseModel):
    id: str
    document_id: str
    title: Optional[str]
    author: Optional[str]
    publication_date: Optional[str]
    genre: Optional[str]
    publisher: Optional[str]
    language: Optional[str]
    isbn: Optional[str]
    summary: Optional[str]
    vector_representation: Optional[List[float]]
    sections: List[Section]

    def get_section_keywords(self):
        section_keywords = [
            keyword
            for section in self.sections
            if section.keywords
            for keyword in section.keywords
        ]

        # remove duplicates
        return list(set(section_keywords))

    def get_section_named_entities(self):
        section_named_entities = [
            entity
            for section in self.sections
            if section.named_entities
            for entity in section.named_entities
        ]
        # remove duplicates
        return list(set(section_named_entities))

    def get_text(self):
        return " ".join(section.section_text for section in self.sections)

import re
from typing import List, Tuple

from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm

from common.text_processing.document_indexer import DocumentIndexer
from common.text_processing.preprocessor import Preprocessor
from schema import Document, Section

CHUNK_SIZE = 5000
CHUNK_OVERLAP = 0


class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.openai_chat_4 = ChatOpenAI(
            model_name="gpt-4", temperature=0.3, max_tokens=2048
        )
        self.openai_chat_3 = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=150
        )

    def generate_response(
        self, template: str, text: str, model_name: str = "gpt-3.5-turbo"
    ) -> List[str]:
        message_prompt = HumanMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages([message_prompt])
        response = ""

        if model_name == "gpt-4":
            response = self._generate_chat_responses(
                chat_prompt, text, self.openai_chat_4, 5000
            )
        elif model_name == "gpt-3.5-turbo":
            response = self._generate_chat_responses(
                chat_prompt, text, self.openai_chat_3, 2000
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        return response

    def _generate_chat_responses(self, chat_prompt, text, chat_model, chunk_size):
        response = ""
        if len(text) > chunk_size:
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                response += chat_model(
                    chat_prompt.format_prompt(text=chunk).to_messages()
                ).content
                response += "/n/n"
        else:
            response.append(
                chat_model(chat_prompt.format_prompt(text=text).to_messages()).content
            )

        return response

    def extract_keywords(self, text: str) -> str:
        template = """Let's define 'Juicy Words' as: 
        Relevant: A Juicy Word must be relevant to the context of the overall text. 
        Recurrent: Juicy Word should be terms or phrases that appear repeatedly throughout the text. 
        Indicative: A Juicy Word should be indicative of larger themes, patterns, or concepts within the text. It could point to significant underlying ideas or relationships that are important to understanding the text.
        Evocative: Juicy Word should evoke or suggest meaning beyond their literal definitions. They often carry connotations and implications that can provide further insights into the data.

        Extract top 25 significant Juicy Words appearing in this text. Do not repeat words. Sort descending by frequency of the word used: {text}
        
        - Keyword
        """
        keywords = self.generate_response(template, text)
        return Preprocessor().text_to_array(keywords)

    def extract_named_entities(self, text: str) -> str:
        template = """Let's define 'Juicy Names' as:
        Specific: Juicy Names are specific instances of a particular category. They are typically proper nouns, but not always.
        Classifiable: A Juicy Name should belong to a defined category or class. Common categories include Person, Location, Organization, Date/Time, but depending on the application, there could be other categories as well, such as Medical Terms, Legal Terms, Product Names, etc.
        Distinct: Juicy Names are distinct from the surrounding text in that they refer to unique entities. They represent real-world objects that can be distinguished from others.
        Informative: Juicy Names often contain key information in a text. Recognizing them allows for the extraction of structured information from unstructured text data.
        Relevant: Juicy Names are words that appear in the text and are relevant to the overall context of the text. They are not random words that appear in the text but are not relevant to the overall context.
        
        Extract top Juicy Names appearing in this text. Do not repeat names. Sort descending by frequency of the name used: {text}

        - Juicy Name
        """
        entities = self.generate_response(template, text)
        return Preprocessor().text_to_array(entities)

    def summary_completion(self, text: str, model_name: str = "gpt-3.5-turbo") -> str:
        template = "Summarize this text, while preserving key concepts and technical keywords: {text}"
        return self.generate_response(template, text, model_name)

    def process_document(
        self, document: Document, index_name: str = "documents"
    ) -> Document:
        tasks = [
            ("Summarizing document", self.summarize_document, [document, True]),
            ("Extracting keywords", self.extract_keywords),
            ("Extracting named entities", self.extract_named_entities),
            ("Indexing document", self.index_document, [index_name]),
        ]

        # Add each summarize_sections as a task
        for i, section in enumerate(document.sections):
            tasks.append((f"Summarizing section {i+1}", self.summarize_sections))

        for desc, func, args in tqdm(tasks, desc="Processing document", unit="task"):
            print(desc)
            if args:
                document = func(document, *args)
            else:
                document = func(document)

        return document

    def split_questions(self, text) -> List[str]:
        # Split the text by the newline character
        lines = text.split('\n')
        
        # Remove any empty strings from the list
        lines = [line for line in lines if line]
        
        # Remove the numbers at the beginning of each line
        questions = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
        
        return questions

    def extract_questions_and_answers(
        self, text: str, model_name: str = "gpt-4"
    ) -> Tuple[List[str], List[str]]:
        questions_template = """
        Extract a list of each question asked by students to the teacher in this transcript:
        
        {text}

        """
        questions_blob = self.generate_response(questions_template, text, model_name)

        questions = self.split_questions(questions_blob)

        answers = []
        for question in questions:
            answers_template = f"Provide the teacher's exact answer (word for word) from this transcript to the question '{question}' If you can't find the exact answer from this transcript, output NOT FOUND.:/n/n" + "{text}"
            answers.append(self.generate_response(answers_template, text, model_name))

        return questions, answers




# To provide a better generated script, you could have extracted more specific information from the original transcript, such as:

# 1. Key points or tips discussed in the transcript
# 2. The speaker's emphasized concepts
# 3. The mention of specific examples or anecdotes
# 4. Any recurring phrases or terms that are unique to the original transcript

# By providing these specific details from the original transcript, the generated script would have been more accurate and closer to the original content, even if not a perfect match.

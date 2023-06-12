from tqdm import tqdm
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from schema import Document, Section
from common.text_processing.document_indexer import DocumentIndexer
from common.text_processing.preprocessor import Preprocessor

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
    ) -> list:
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

        Extract top 25 significant Juicy Words appearing in this text: {text}
        
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
        
        Extract top Juicy Names appearing in this text: {text}

        - Juicy Name
        """
        entities = self.generate_response(template, text)
        return Preprocessor().text_to_array(entities)

    def summary_completion(self, text: str, model_name: str = "gpt-3.5-turbo") -> str:
        template = "Summarize this text, while preserving key concepts and technical keywords: {text}"
        return self.generate_response(template, text, model_name)

    def summarize_document(
        self, document: Document, with_sections: bool = False
    ) -> Document:
        # Split the full text into chunks
        get_text = document.get_text()
        text_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(get_text)

        # Summarize each chunk and concatenate the summaries
        summaries = [
            self.summary_completion(chunk, model_name="gpt-4") for chunk in chunks
        ]

        # Set the Document's summary attribute
        document.summary = "\n".join(summaries)

        # Vectorize the summary and set Document vector_representation attribute
        document.vector_representation = self.embeddings.embed_documents(
            [document.summary]
        )[0]

        if with_sections:
            if len(document.sections) > 1:
                document.sections = [
                    self.summarize_section(section) for section in document.sections
                ]
            else:
                document.sections[0].summary = document.summary

        return document

    def summarize_section(self, section: Section) -> Section:
        section.summary = self.summary_completion(
            section.section_text, model_name="gpt-4"
        )

        # Vectorize the summary and set Section vector_representation attribute
        section.vector_representation = self.embeddings.embed_documents(
            [section.summary]
        )[0]

        return section

    def index_document(self, document: Document, index_name: str = "documents") -> str:
        indexer = DocumentIndexer(index_name=index_name)
        return indexer.index_document(document)

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


# To provide a better generated script, you could have extracted more specific information from the original transcript, such as:

# 1. Key points or tips discussed in the transcript
# 2. The speaker's emphasized concepts
# 3. The mention of specific examples or anecdotes
# 4. Any recurring phrases or terms that are unique to the original transcript

# By providing these specific details from the original transcript, the generated script would have been more accurate and closer to the original content, even if not a perfect match.

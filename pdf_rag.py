from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.utils.auth import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.routers import ConditionalRouter

import os
import warnings
from dotenv import load_dotenv, find_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())


def create_indexing_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    """
    The indexing pipeline purpose is to fetch a PDF document and write it's embeddings into the document store (in memory)
    Components:
    - PyPDFToDocument: Converts PDF to Document
    - DocumentCleaner: Cleans the document
    - DocumentSplitter: Splits the document into smaller chunks
    - OpenAIDocumentEmbedder: Embeds the document using OpenAI
    - DocumentWriter: Writes the document into the document store
    """

    print("Creating indexing pipeline...")

    pipeline = Pipeline()

    converter = PyPDFToDocument()
    cleaner = DocumentCleaner()

    splitter = DocumentSplitter(
        split_by="sentence", split_length=10, split_overlap=2)
    embedder = OpenAIDocumentEmbedder()
    writer = DocumentWriter(document_store=document_store)

    pipeline.add_component("converter", converter)
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    print("Index pipeline created...")

    return pipeline


def create_query_pipeline(template: str, document_store: InMemoryDocumentStore) -> Pipeline:
    """
    Creates a query pipeline for retrieving and generating responses based on a given template and document store.
    The query pipeline is responsible for:
    1. Embedding the query using OpenAITextEmbedder.
    2. Retrieving relevant documents from the in-memory document store using the query embedding.
    3. Building a prompt using the retrieved documents and the provided template.
    4. Generating a response based on the constructed prompt using OpenAI.
    """

    print("Creating query pipeline...")

    query_embedder = OpenAITextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator()

    rag = Pipeline()
    rag.add_component("query_embedder", query_embedder)
    rag.add_component("retriever", retriever)
    rag.add_component("prompt", prompt_builder)
    rag.add_component("generator", generator)

    rag.connect("query_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever.documents", "prompt.documents")
    rag.connect("prompt", "generator")

    print("Query pipeline created...")
    return rag


def main():

    warnings.filterwarnings('ignore')

    print("Loading environment variables...")
    load_env()

    document_store = InMemoryDocumentStore()
    indexing_pipeline = create_indexing_pipeline(document_store=document_store)
    indexing_pipeline.run(
        {"converter": {"sources": ['data/travel-guide.pdf']}})

    template = """
    Answer the question based on the provided context. Do it in italian.
    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}
    Question: {{ query }}
    """

    query_pipeline = create_query_pipeline(template, document_store)
    # print(indexing_pipeline.dumps())
    # print(query_pipeline.dumps())

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting...")
            break

        result = query_pipeline.run(
            {
                "query_embedder": {"text": question},
                "retriever": {"top_k": 3},
                "prompt": {"query": question},
            }
        )

        print("Answer: ", result["generator"]["replies"][0])


if __name__ == "__main__":
    main()

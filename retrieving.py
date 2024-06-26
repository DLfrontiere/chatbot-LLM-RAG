from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.embeddings.embeddings import Embeddings

import uuid

class Retriever:

    def __init__(self, docs, embedding_function, splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, add_start_index=True)):
        self.docs = docs
        self.embedding_function = embedding_function
        self.splitter = splitter

    def get_retriever(self):
        texts = self.splitter.split_documents(self.docs)
        db = Chroma.from_documents(self.docs, self.embedding_function)
        #retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.3})
        retriever = db.as_retriever(search_type="similarity",search_kwargs={'k': 3})
        return retriever

class ParentRetriever:

    def __init__(self, docs, vectorstore, parent_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=125, add_start_index=True), child_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=25, add_start_index=True)):
        self.docs = docs
        self.vectorstore = vectorstore
        self.child_splitter = parent_splitter
        self.parent_splitter = child_splitter

    def get_retriever(self):
        docs = self.docs
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        retriever.add_documents(docs)
        return retriever

class CompressionExtractorRetriever:

    def __init__(self, base_retriever,model):
        self.base_retriever = base_retriever
        self.model = model

    def get_compression_retriever(self):
        llm = self.model
        compressor = LLMChainExtractor.from_llm(llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )
        return self.retriever
    
    def get_retriever(self):
        return self.get_compression_retriever()

class CompressionFilterRetriever:

    def __init__(self, base_retriever,model):
        self.base_retriever = base_retriever
        self.model = model

    def get_compression_retriever(self):
        llm = self.model
        compressor = LLMChainFilter.from_llm(llm)
        compression_retriever = compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )
        return compression_retriever

    def get_retriever(self):
        return self.get_compression_retriever()

class CompressionEmbeddingRetriever:

    def __init__(self, base_retriever,docs,embedding_function):
        self.embedding_function = embedding_function
        self.db = Chroma.from_documents(docs,self.embedding_function)
        self.embeddings = self.embedding_function.embed_documents([d.page_content for d in docs])
        self.base_retriever = base_retriever

    def get_compression_embedding_retriever(self):

        embeddings_filter = EmbeddingsFilter(embeddings=self.embedding_function, similarity_threshold=0.76)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=self.base_retriever
        )
        return compression_retriever

    def get_retriever(self):
        return self.get_compression_embedding_retriever()



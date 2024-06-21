import os
from langchain_community.document_loaders import PyPDFLoader


class Loader:
    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.all_docs = []

    def load_documents(self):
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdfloader = PyPDFLoader(os.path.join(self.pdf_directory, filename))
                docs = pdfloader.load()
                print(len(docs),docs)
                self.all_docs.extend(docs)
        return self.all_docs





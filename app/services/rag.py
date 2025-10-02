from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.utils import get_template
from app.core import config



class rag_chain:
    def __init__(self, template: str):
        self.rag_chain = self._initialize(template)
        
    def __call__(self, template: str):
        return self.rag_chain()

    def _initialize(self, template: str):
        """Initializes and returns the RAG chain."""
        
        # Configuration
        DB_DIR = '../db'
        COLLECTION_NAME = 'customer_reviews'
        EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
 
        retriever = self._get_vector_store()
        template = get_template(template)
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name=config["llm"]["model"])
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    
    def _get_retriever(self):
        """get retriever from vector store"""
        embedding_function = SentenceTransformerEmbeddings(model_name=config["embedding"]["model"])
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
    @property
    def chain(self):
        return self.rag_chain
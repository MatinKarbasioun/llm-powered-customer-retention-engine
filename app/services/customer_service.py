from kink import inject

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.utils import get_template
from app.core import config, env

from app.repositories import ICustomerRepository


@inject
class CustomerService:
    def __init__(self, customer_repository: ICustomerRepository):
        self._customer_repo = customer_repository
        self._rag_chain = self._initialize()
    
    def _initialize(self):
        """Initializes the RAG and ready to interact with the user."""
        
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
    
    def ask(self, query: str):
        return self._rag_chain.invoke(query)

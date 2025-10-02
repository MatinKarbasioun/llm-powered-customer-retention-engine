from contextlib import asynccontextmanager
from fastapi import FastAPI
from .utils import env
import os

# Import all the RAG components from our notebook
from .routers import customer_v1_router, root_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """_summary_
    lifespan of FastAPI application for graceful shutdown and bootstrap
    
    Args:
        app (FastAPI): the fastapi application instance
                    
    """
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    yield
    

app = FastAPI(
    lifespan=lifespan,
    title="Customer Insights Engine API",
    description="An API for querying customer data using a RAG pipeline.",
    version="1.0.0",
)

app.include_router(customer_v1_router, prefix="/v1/customers")
app.include_router(root_router, prefix="/")

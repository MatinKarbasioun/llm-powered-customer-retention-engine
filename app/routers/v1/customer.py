from fastapi import Depends
from fastapi.routing import APIRouter
from pydantic import BaseModel
from functools import lru_cache
from app.services import RagChain

import os


customer_router = APIRouter(
    tags=["customer"],
    responses={404: {"description": "Not found"}},
)


@customer_router.post("/ask")
async def ask_question(query: Query, rag_chain: Depends()):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ry):
    """
    Accepts a user's question and returns the RAG chain's answer.
    """
    answer = rag_chain.invoke(query.question)
    return {"answer": answer}
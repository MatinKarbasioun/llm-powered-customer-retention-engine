from fastapi import APIRouter


root_router = APIRouter(
    tags=["root"],
    responses={404: {"description": "Not found"}},
)


@root_router.get("/")
async def read_root():
    return {"message": "Welcome to the Customer Insights Engine API."}
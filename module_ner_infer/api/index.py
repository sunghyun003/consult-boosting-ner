
from fastapi import APIRouter, Response
from module_ner_infer.api.routers.inference import router as inference_router

api_router = APIRouter()
api_router.include_router(
    inference_router,
    tags=["inference"],
)


@api_router.get("/favicon.ico")
async def ignore_favicon():
    return Response(status_code=204)


@api_router.get("/healthcheck")
async def healthcheck():
    return Response(status_code=200)

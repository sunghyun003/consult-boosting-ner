
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
)
from pydantic import BaseModel
from module_ner_infer.services.inference import NERInferenceService

router = APIRouter()


class NERRequest(BaseModel):
    text: str


@router.post("/infer")
async def infer(request: Request, body: NERRequest):
    try:
        service: NERInferenceService = request.app.state.ner_service
        result = service.infer(body.text)
        return {
            "input": body.text,
            "result": result
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"[{request.url.path}][{request.method}]: {exc}")

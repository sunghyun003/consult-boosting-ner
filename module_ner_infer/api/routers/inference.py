
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
)
from pydantic import BaseModel
from module_ner_infer.services.inference import (
    NERInferenceService,
    MeCabInferenceService,
)

router = APIRouter()


class NERRequest(BaseModel):
    text: str


@router.post("/infer")
async def infer(request: Request, body: NERRequest):
    try:
        ner_service: NERInferenceService = request.app.state.ner_service
        mecab_service: MeCabInferenceService = request.app.state.mecab_service

        ner_result = ner_service.infer(body.text)
        mecab_result = mecab_service.infer(body.text)

        return {
            "result": {
                "ner": ner_result,
                "mecab": mecab_result,
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"[{request.url.path}][{request.method}]: {exc}")

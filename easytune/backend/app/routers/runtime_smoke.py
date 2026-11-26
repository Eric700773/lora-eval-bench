from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.qwen_runtime_service import run_qlora_smoke, run_single_inference_smoke


router = APIRouter(prefix="/api/runtime-smoke", tags=["runtime-smoke"])


class QwenInferenceSmokeRequest(BaseModel):
    prompt: str = Field(default="请只回答数字：2+3等于多少？", min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    load_in_4bit: bool = False


class QwenQLoRASmokeRequest(BaseModel):
    prompt: str = Field(default="Answer with only the number: What is 7 times 6?", min_length=1)
    answer: str = Field(default="42", min_length=1)
    rank: int = Field(default=8, ge=1, le=64)


@router.post("/qwen-inference")
def qwen_inference_smoke(payload: QwenInferenceSmokeRequest) -> dict[str, object]:
    return run_single_inference_smoke(
        prompt=payload.prompt,
        max_new_tokens=payload.max_new_tokens,
        load_in_4bit=payload.load_in_4bit,
    )


@router.post("/qwen-qlora")
def qwen_qlora_smoke(payload: QwenQLoRASmokeRequest) -> dict[str, object]:
    return run_qlora_smoke(payload.prompt, payload.answer, payload.rank)

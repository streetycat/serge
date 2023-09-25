import os

from typing import Optional, List
from fastapi import APIRouter
from langchain.memory import RedisChatMessageHistory
from langchain.schema import SystemMessage, messages_to_dict, AIMessage, HumanMessage
from llama_cpp import Llama
from loguru import logger
from redis import Redis
from sse_starlette.sse import EventSourceResponse

from serge.models.chat import Chat, ChatParameters
from serge.utils.stream import get_prompt

completion_router = APIRouter(
    prefix="/completion",
    tags=["completion"],
)


@completion_router.post("/")
async def completion(
    prompts: List[str],
    model: str = "7B",
    temperature: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 2048,
    context_window: int = 2048,
    gpu_layers: Optional[int] = None,
    repeat_last_n: int = 64,
    repeat_penalty: float = 1.3,
):
    final_prompt = ""
    for prompt in prompts:
        final_prompt += prompt + "\n"
    
    try:
        client = Llama(
            model_path="/usr/src/app/weights/" + model + ".bin",
            n_ctx=context_window,
            n_threads=len(os.sched_getaffinity(0)),
            n_gpu_layers=gpu_layers,
            last_n_tokens_size=repeat_last_n,
        )
        answer = client(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            max_tokens=max_length,
        )
    except Exception as e:
        error = e.__str__()
        logger.error(error)
        return error

    if not isinstance(answer, str):
        answer = str(answer)

    return answer

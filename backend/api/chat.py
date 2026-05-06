"""Chat API endpoint with SSE streaming."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.models.schemas import ChatRequest
from backend.services.ocr import extract_text_from_base64_image
from backend.services.graph import GraphState, astream_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(req: ChatRequest):
    """Send a message and receive a streaming response via SSE.

    The response uses ``text/event-stream`` content type. Each SSE event
    carries a JSON payload ``{"token": "..."}`` for incremental tokens,
    or ``{"done": true}`` when the stream ends.
    """
    logger.info(
        "POST /api/chat: message='%s', kb_name=%s, model=%s/%s, has_image=%s",
        req.message[:80] + ("..." if len(req.message) > 80 else ""),
        req.kb_name,
        req.model_provider,
        req.model_name,
        bool(req.image),
    )

    image_text = req.image_text or ""
    if req.image and not image_text:
        image_text = extract_text_from_base64_image(req.image)

    state: GraphState = {
        "question": req.message,
        "image": req.image,
        "image_text": image_text,
        "kb_name": req.kb_name,
        "model_provider": req.model_provider,
        "model_name": req.model_name,
        "retrieved_context": "",
        "answer": "",
    }

    async def event_generator():
        try:
            async for token in astream_answer(state):
                data = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
            logger.info("Chat SSE stream completed successfully")
        except Exception as e:
            logger.error("Error during chat streaming: %s", e, exc_info=True)
            error_data = json.dumps(
                {"error": str(e)}, ensure_ascii=False
            )
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

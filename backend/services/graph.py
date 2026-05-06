"""LangGraph RAG workflow."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from backend.services.knowledge_base import query_knowledge_base
from backend.services.llm import get_chat_model

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State that flows through the graph."""

    question: str
    image: Optional[str]
    image_text: Optional[str]
    kb_name: Optional[str]
    model_provider: str
    model_name: str
    retrieved_context: str
    answer: str


def retrieve(state: GraphState) -> dict:
    """Retrieve relevant context from the selected knowledge base."""

    kb_name = state.get("kb_name")
    question = state["question"]

    if not kb_name:
        logger.info("No knowledge base selected, skipping retrieval")
        return {"retrieved_context": ""}

    logger.info("Retrieving context from KB '%s' for question: '%s'", kb_name, question[:80])
    try:
        docs = query_knowledge_base(kb_name, question, n_results=5)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        logger.info(
            "Retrieved %d documents from KB '%s', total context length=%d chars",
            len(docs),
            kb_name,
            len(context),
        )
    except Exception as exc:
        logger.error("Error retrieving from KB '%s': %s", kb_name, exc, exc_info=True)
        context = ""

    return {"retrieved_context": context}


def generate(state: GraphState) -> dict:
    """Generate an answer using the selected LLM."""

    logger.info(
        "Generating answer: provider=%s, model=%s",
        state["model_provider"],
        state["model_name"],
    )
    llm = get_chat_model(
        state["model_provider"],
        state["model_name"],
        streaming=False,
    )

    messages = _build_messages(state)
    logger.debug("Sending %d messages to LLM", len(messages))
    response = llm.invoke(messages)
    logger.info("LLM response received, length=%d chars", len(response.content))
    return {"answer": response.content}


def _build_messages(state: GraphState) -> list:
    """Construct the message list for the LLM."""

    messages: list = []

    system_text = (
        "You are a professional AI assistant. Answer clearly and helpfully. "
        "When knowledge-base context is provided, use it to answer the user. "
        "If the context does not contain the answer, answer from general knowledge "
        "and mention that it was not found in the knowledge base. "
        "Reply in English unless the user asks for another language."
    )
    messages.append(SystemMessage(content=system_text))

    context = state.get("retrieved_context", "")
    if context:
        logger.debug("Including retrieved context (%d chars) in messages", len(context))
        messages.append(
            SystemMessage(
                content=f"Relevant content retrieved from the knowledge base:\n\n{context}"
            )
        )

    image_text = (state.get("image_text") or "").strip()
    if image_text:
        logger.debug("Including OCR image text (%d chars) in messages", len(image_text))
        messages.append(
            SystemMessage(
                content=(
                    "Text extracted from the uploaded image with OCR:\n\n"
                    f"{image_text}\n\n"
                    "Use this OCR text when answering questions about the image. "
                    "If the OCR may be incomplete or unclear, say so."
                )
            )
        )

    image = state.get("image")
    provider = state.get("model_provider")

    if image and provider != "groq":
        logger.info("Including image in user message (base64 length=%d)", len(image))
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": state["question"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ]
            )
        )
    else:
        question = state["question"]
        if image:
            logger.info(
                "Image omitted for provider=%s because this model expects text content",
                provider,
            )
            if image_text:
                question = f"{question}\n\nUse the OCR text extracted from the attached image."
            else:
                question = (
                    f"{question}\n\n"
                    "Note: An image was attached, but no readable text was extracted and "
                    "the selected Groq text model cannot inspect image pixels directly."
                )
        messages.append(HumanMessage(content=question))

    logger.debug("Built %d messages for LLM", len(messages))
    return messages


def _build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow


graph = _build_graph().compile()
logger.info("LangGraph RAG workflow compiled: retrieve -> generate -> END")


async def astream_answer(state: GraphState) -> AsyncIterator[str]:
    """Run retrieval, then stream the LLM answer token by token."""

    logger.info(
        "astream_answer started: question='%s', kb=%s, model=%s/%s",
        state["question"][:80],
        state.get("kb_name"),
        state["model_provider"],
        state["model_name"],
    )

    retrieve_result = retrieve(state)
    state = {**state, **retrieve_result}

    llm = get_chat_model(
        state["model_provider"],
        state["model_name"],
        streaming=True,
    )

    messages = _build_messages(state)
    logger.info("Starting LLM streaming generation")

    token_count = 0
    async for chunk in llm.astream(messages):
        if chunk.content:
            token_count += 1
            yield chunk.content

    logger.info("Streaming complete, yielded %d token chunks", token_count)


import os
import logging
import asyncio

from dotenv import load_dotenv

# Load environment variables BEFORE importing model.py
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from model import generate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Bedrock Product Discovery API",
    description="AI-powered product discovery using AWS Bedrock",
    version="1.0.0"
)


# CORS configuration

origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://continuos-learning-chatbot.vercel.app"
)

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in origins.split(",")
    if origin.strip()
]

logger.info("CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request models

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):

    message: str | None = None
    prompt: str | None = None
    query: str | None = None

    history: list[ChatMessage] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_message(cls, values):

        if isinstance(values, dict):

            values = values.copy()

            if not values.get("message"):
                values["message"] = (
                    values.get("prompt") or values.get("query")
                )

        return values

    @model_validator(mode="after")
    def validate_message(self):

        if not self.message or not self.message.strip():
            raise ValueError("A chat message is required.")

        return self


# Health endpoints

@app.get("/")
async def root():

    return {
        "service": "Bedrock Product Discovery API",
        "status": "running"
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "bedrock-chat"
    }


@app.get("/ready")
async def ready():

    return {
        "status": "ready"
    }


# Main chat endpoint

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        logger.info("Received chat request.")

        history = [
            item.model_dump()
            for item in request.history
        ]

        result = await asyncio.wait_for(
            run_in_threadpool(
                generate_response,
                request.message.strip(),
                history
            ),
            timeout=150
        )

        logger.info("Chat response generated successfully.")

        return {
            "success": True,
            "response": result["response"],
            "answer": result["response"],
            "sources": result["sources"],
            "model": result["model"]
        }

    except asyncio.TimeoutError:

        logger.error("Chat request timed out.")

        raise HTTPException(
            status_code=504,
            detail="The AI response took too long. Please try again."
        )

    except RuntimeError:

        logger.exception("Bedrock service error.")

        raise HTTPException(
            status_code=502,
            detail="Unable to generate a response using AWS Bedrock."
        )

    except Exception:

        logger.exception("Unexpected chat API error.")

        raise HTTPException(
            status_code=500,
            detail="Internal chat service error."
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )

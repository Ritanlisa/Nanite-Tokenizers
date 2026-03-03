import argparse
import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from agent.agent import SmartAgent
from main import health_check, setup_logging
from monitoring import start_metrics_server

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field("default", min_length=1)
    stream: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class ResetRequest(BaseModel):
    session_id: str = Field("default", min_length=1)


SESSION_STORE: dict[str, SmartAgent] = {}
SESSION_PARAMS: dict[str, dict[str, Optional[object]]] = {}


def get_or_create_agent(session_id: str, model: Optional[str], temperature: Optional[float]) -> SmartAgent:
    params = {"model": model, "temperature": temperature}
    if session_id not in SESSION_STORE or SESSION_PARAMS.get(session_id) != params:
        SESSION_STORE[session_id] = SmartAgent(
            session_id=session_id,
            model=model,
            temperature=temperature,
        )
        SESSION_PARAMS[session_id] = params
    return SESSION_STORE[session_id]


def clear_session(session_id: str) -> None:
    SESSION_STORE.pop(session_id, None)
    SESSION_PARAMS.pop(session_id, None)


def update_settings(args: argparse.Namespace) -> None:
    config.settings = config.settings.update(
        DATA_DIR=args.data_dir,
        PERSIST_DIR=args.persist_dir,
        AGENT_VERBOSE=args.verbose or config.settings.AGENT_VERBOSE,
        LOG_LEVEL="DEBUG" if args.verbose else config.settings.LOG_LEVEL,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Nanite Agent API")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        agent = get_or_create_agent(request.session_id, request.model, request.temperature)
        try:
            answer = await agent.achat(request.message)
        except Exception as exc:
            logging.getLogger(__name__).exception("Chat failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if request.stream:
            async def streamer():
                for start in range(0, len(answer), 24):
                    yield answer[start : start + 24]

            return StreamingResponse(streamer(), media_type="text/plain")

        return {"answer": answer}

    @app.post("/api/reset")
    async def reset(request: ResetRequest):
        clear_session(request.session_id)
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nanite Agent Web API")
    parser.add_argument("--data-dir", default=config.settings.DATA_DIR)
    parser.add_argument("--persist-dir", default=config.settings.PERSIST_DIR)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--metrics-port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_settings(args)
    setup_logging()
    start_metrics_server(args.metrics_port)

    try:
        asyncio.run(health_check())
    except Exception as exc:
        logging.getLogger(__name__).error("Health check failed: %s", exc)
        raise SystemExit(1) from exc

    import uvicorn

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

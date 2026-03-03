import argparse
import asyncio
import logging
import uuid

import gradio as gr

import config
from agent.agent import SmartAgent
from main import health_check, setup_logging
from monitoring import start_metrics_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG + MCP Agent (Gradio GUI)")
    parser.add_argument("--data-dir", default=config.settings.DATA_DIR)
    parser.add_argument("--persist-dir", default=config.settings.PERSIST_DIR)
    parser.add_argument("--session", default="default")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--metrics-port", type=int, default=8000)
    return parser.parse_args()


def update_settings(args: argparse.Namespace) -> None:
    config.settings = config.settings.update(
        DATA_DIR=args.data_dir,
        PERSIST_DIR=args.persist_dir,
        AGENT_VERBOSE=args.verbose or config.settings.AGENT_VERBOSE,
        LOG_LEVEL="DEBUG" if args.verbose else config.settings.LOG_LEVEL,
    )


def build_ui(session_prefix: str) -> gr.Blocks:
    def new_agent() -> SmartAgent:
        suffix = uuid.uuid4().hex[:8]
        session_id = f"{session_prefix}-{suffix}" if session_prefix else suffix
        return SmartAgent(session_id=session_id)

    async def respond(message: str, history: list[tuple[str, str]], agent_state: SmartAgent | None):
        if not message.strip():
            return "", history, agent_state
        if agent_state is None:
            agent_state = new_agent()
        answer = await agent_state.achat(message)
        history = history + [(message, answer)]
        return "", history, agent_state

    def reset_session():
        return [], new_agent()

    with gr.Blocks(title="Nanite Agent") as demo:
        gr.Markdown("# Nanite Agent GUI")
        gr.Markdown("Chat with the RAG + MCP agent using a local Gradio UI.")

        chatbot = gr.Chatbot(label="Conversation")
        agent_state = gr.State(None)

        with gr.Row():
            msg = gr.Textbox(label="Message", placeholder="Type a message and press Enter", scale=8)
            send = gr.Button("Send", scale=1)

        new_session = gr.Button("New Session")

        msg.submit(respond, [msg, chatbot, agent_state], [msg, chatbot, agent_state])
        send.click(respond, [msg, chatbot, agent_state], [msg, chatbot, agent_state])
        new_session.click(reset_session, outputs=[chatbot, agent_state])

    return demo


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

    demo = build_ui(args.session)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

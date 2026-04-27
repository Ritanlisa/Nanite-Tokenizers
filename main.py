import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import sys
import time
from typing import Optional
from logging.handlers import RotatingFileHandler
import colorlog
import config
from capabilities import set_capabilities
from monitoring import start_metrics_server

_ONE_BY_ONE_JPEG_DATA_URL = (
    "data:image/png;base64,"
    "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"
)


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            try:
                from tqdm import tqdm

                tqdm.write(message)
            except Exception:
                stream = self.stream
                stream.write(f"{message}{self.terminator}")
                self.flush()
        except Exception:
            self.handleError(record)


async def _check_tool_calling_support(client, model: str, logger: logging.Logger) -> bool:
    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Return OK.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    def _has_tool_calls(response) -> bool:
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return False
            message = getattr(choices[0], "message", None)
            tool_calls = getattr(message, "tool_calls", None)
            return bool(tool_calls)
        except Exception:
            return False

    def _is_connection_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "connection error" in text
            or "server disconnected" in text
            or "connection reset" in text
            or "remoteprotocolerror" in text
        )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=tool_schema,
            tool_choice={"type": "function", "function": {"name": "ping"}},
            max_tokens=8,
        )
        if _has_tool_calls(response):
            logger.info("Model supports tool calling: %s", model)
            return True

        logger.info(
            "Tool-call probe accepted tools payload without tool_calls (%s), treating as supported",
            model,
        )
        return True
    except Exception as exc:
        if not _is_connection_error(exc):
            logger.warning("Model does not support tool calling (%s): %s", model, exc)
            return False

        logger.info(
            "Tool-call probe hit connection issue for %s, retrying with compatibility payload",
            model,
        )

    try:
        fallback_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Use ping once and return done."}],
            tools=tool_schema,
            max_tokens=32,
            temperature=0,
        )
        if _has_tool_calls(fallback_response):
            logger.info("Model supports tool calling (compat mode): %s", model)
            return True

        logger.info(
            "Tool-call compat probe accepted tools payload without tool_calls (%s), treating as supported",
            model,
        )
        return True
    except Exception as exc:
        logger.warning("Model does not support tool calling (%s): %s", model, exc)
        return False


async def _check_multimodal_support(
    client, model: str, logger: logging.Logger
) -> Optional[bool]:
    try:
        await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image in one word."},
                        {"type": "image_url", "image_url": {"url": _ONE_BY_ONE_JPEG_DATA_URL}},
                    ],
                }
            ],
            max_tokens=1,
        )
        logger.info("Model supports multimodal input: %s", model)
        return True
    except Exception as exc:
        error_text = str(exc)
        if "too much pixel data" in error_text or "invalid format" in error_text:
            logger.warning(
                "Multimodal probe failed for %s due to image decoding limits: %s",
                model,
                exc,
            )
            return None
        logger.warning("Model does not support multimodal input (%s): %s", model, exc)
        return False


async def _await_with_progress(
    coro,
    timeout: int,
    label: str,
    logger: logging.Logger,
):
    try:
        from tqdm import tqdm
    except Exception as exc:
        logger.warning("tqdm unavailable, waiting without progress: %s", exc)
        return await asyncio.wait_for(coro, timeout=timeout)

    task = asyncio.create_task(coro)
    start = time.monotonic()
    with tqdm(
        total=timeout,
        desc=label,
        unit="s",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as bar:
        while True:
            done, _ = await asyncio.wait({task}, timeout=1)
            elapsed = time.monotonic() - start
            bar.n = min(int(elapsed), timeout)
            bar.refresh()
            if done:
                return await task
            if elapsed >= timeout:
                task.cancel()
                raise asyncio.TimeoutError(f"{label} timed out after {timeout}s")


def setup_logging() -> None:
    configured_level_name = str(config.settings.LOG_LEVEL).upper()
    configured_level = getattr(logging, configured_level_name, logging.INFO)

    # 基础格式（用于文件）
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # 带颜色的格式（用于控制台）
    color_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"

    # ---------- 控制台 Handler（带颜色）----------
    console = TqdmLoggingHandler()  # 您已有的与 tqdm 兼容的 handler
    console.setLevel(configured_level)
    color_formatter = colorlog.ColoredFormatter(
        color_format,
        datefmt=None,  # 使用默认日期格式，也可自定义如 "%Y-%m-%d %H:%M:%S"
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'  # 使用 % 风格，与 logging 默认一致
    )
    console.setFormatter(color_formatter)

    # ---------- 文件 Handler（无颜色）----------
    os.makedirs(os.path.dirname(config.settings.LOG_FILE), exist_ok=True)
    file_handler = RotatingFileHandler(
        config.settings.LOG_FILE,
        maxBytes=config.settings.LOG_MAX_BYTES,
        backupCount=config.settings.LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(configured_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # ---------- 配置 Root Logger ----------
    root_logger = logging.getLogger()
    root_logger.setLevel(configured_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    # ---------- 统一 Uvicorn 日志 ----------
    from uvicorn.config import LOGGING_CONFIG
    LOGGING_CONFIG["loggers"]["uvicorn"] = {"handlers": [], "level": configured_level_name, "propagate": True}
    LOGGING_CONFIG["loggers"]["uvicorn.error"] = {"handlers": [], "level": configured_level_name, "propagate": True}
    LOGGING_CONFIG["loggers"]["uvicorn.access"] = {"handlers": [], "level": configured_level_name, "propagate": True}

def parse_args():
    parser = argparse.ArgumentParser(description="RAG + MCP Agent")
    parser.add_argument("--data-dir", default=config.settings.DATA_DIR)
    parser.add_argument("--persist-dir", default=config.settings.PERSIST_DIR)
    parser.add_argument("--session", default="default")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-file", help="Process inputs from a file")
    parser.add_argument("--output-file", help="Write batch results (.csv or .json)")
    parser.add_argument("--metrics-port", type=int, default=8000)
    return parser.parse_args()


async def health_check(include_mcp: bool = True) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running startup health check")

    openai_client = None
    try:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(
            api_key=config.settings.OPENAI_API_KEY,
            base_url=config.settings.OPENAI_API_URL,
        )
        await openai_client.models.list()
        logger.info("OpenAI API reachable")

        tool_supported = await _check_tool_calling_support(
            openai_client, config.settings.LLM_MODEL, logger
        )
        multimodal_supported = await _check_multimodal_support(
            openai_client, config.settings.LLM_MODEL, logger
        )
        set_capabilities(
            tool_calling_supported=tool_supported,
            multimodal_supported=multimodal_supported,
        )
    except Exception as exc:
        logger.error("OpenAI API check failed: %s", exc)
        sys.exit(1)
    finally:
        if openai_client is not None:
            await openai_client.close()

    if include_mcp:
        from mcp_client.client import get_mcp_client

        mcp_client = get_mcp_client()
        try:
            await _await_with_progress(
                mcp_client.initialize(),
                config.settings.MCP_INIT_TIMEOUT,
                "MCP init",
                logger,
            )
            if getattr(mcp_client, "_fallback_mode", False):
                logger.warning("MCP server unavailable or disabled; using direct HTTP fallback")
            else:
                logger.info("MCP server reachable")
        except asyncio.TimeoutError as exc:
            logger.warning("MCP init timed out: %s", exc)
        except Exception as exc:
            logger.warning("MCP init failed: %s", exc)

    persist_dir = config.get_rag_persist_dir()
    os.makedirs(persist_dir, exist_ok=True)
    test_file = os.path.join(persist_dir, ".write_test")
    try:
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("test")
        os.remove(test_file)
        logger.info("Persist dir is writable")
    except Exception as exc:
        logger.error("Persist dir not writable: %s", exc)
        sys.exit(1)


async def batch_process(input_file: str, output_file: str | None = None):
    from agent.agent import Agent

    logger = logging.getLogger(__name__)
    agent = Agent()
    with open(input_file, "r", encoding="utf-8") as handle:
        queries = [line.strip() for line in handle if line.strip()]

    total = len(queries)
    logger.info("Batch processing %s queries with concurrency %s", total, config.settings.BATCH_CONCURRENCY)

    semaphore = asyncio.Semaphore(config.settings.BATCH_CONCURRENCY)

    async def process_one(query: str, idx: int):
        async with semaphore:
            try:
                answer = await agent.achat(query)
                return {"query": query, "answer": answer, "status": "success"}
            except Exception as exc:
                logger.exception("Batch query %s failed", idx)
                return {"query": query, "error": str(exc), "status": "failed"}

    tasks = [process_one(query, idx + 1) for idx, query in enumerate(queries)]
    results = await asyncio.gather(*tasks)

    success = sum(1 for result in results if result.get("status") == "success")
    logger.info("Batch complete: %s/%s", success, total)

    if output_file:
        if output_file.endswith(".json"):
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
        else:
            with open(output_file, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["query", "answer", "error", "status"])
                writer.writeheader()
                writer.writerows(results)
        logger.info("Batch results written to %s", output_file)
    return results


async def shutdown() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Shutting down")
    from mcp_client.client import get_mcp_client

    await get_mcp_client().close()
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    args = parse_args()
    new_settings = config.settings.update(
        DATA_DIR=args.data_dir,
        PERSIST_DIR=args.persist_dir,
        AGENT_VERBOSE=args.verbose or config.settings.AGENT_VERBOSE,
        LOG_LEVEL="DEBUG" if args.verbose else config.settings.LOG_LEVEL,
    )
    config.settings = new_settings

    setup_logging()
    start_metrics_server(args.metrics_port)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(shutdown()))

    await health_check()

    if args.batch_file:
        await batch_process(args.batch_file, args.output_file)
        return

    from agent.agent import Agent

    agent = Agent(session_id=args.session)
    print("Agent ready (type 'exit' to quit)")
    while True:
        try:
            user = await asyncio.get_event_loop().run_in_executor(None, input, "\nYou: ")
            if user.lower() in {"exit", "quit"}:
                break
            if not user.strip():
                continue
            response = await agent.achat(user)
            print(f"Assistant: {response}")
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as exc:
            logging.exception("Main loop error")
            if config.settings.ENV == "prod":
                print("An error occurred.")
            else:
                print(f"Error: {exc}")

    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())

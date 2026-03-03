import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

import config
from monitoring import start_metrics_server


def setup_logging() -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console = logging.StreamHandler()
    console.setLevel(
        logging.INFO if config.settings.ENV != "prod" else logging.WARNING
    )
    console.setFormatter(logging.Formatter(log_format))

    os.makedirs(os.path.dirname(config.settings.LOG_FILE), exist_ok=True)
    file_handler = RotatingFileHandler(
        config.settings.LOG_FILE,
        maxBytes=config.settings.LOG_MAX_BYTES,
        backupCount=config.settings.LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(
        logging.DEBUG if config.settings.ENV == "dev" else logging.INFO
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(config.settings.LOG_LEVEL)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


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


async def health_check() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running startup health check")

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.settings.OPENAI_API_KEY)
        await client.models.list()
        logger.info("OpenAI API reachable")
    except Exception as exc:
        logger.error("OpenAI API check failed: %s", exc)
        sys.exit(1)

    from mcp.client import get_mcp_client

    client = get_mcp_client()
    try:
        await client.initialize()
        logger.info("MCP server reachable")
    except Exception as exc:
        logger.warning("MCP init failed: %s", exc)

    os.makedirs(config.settings.PERSIST_DIR, exist_ok=True)
    test_file = os.path.join(config.settings.PERSIST_DIR, ".write_test")
    try:
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("test")
        os.remove(test_file)
        logger.info("Persist dir is writable")
    except Exception as exc:
        logger.error("Persist dir not writable: %s", exc)
        sys.exit(1)


async def batch_process(input_file: str, output_file: str | None = None):
    from agent.agent import SmartAgent

    logger = logging.getLogger(__name__)
    agent = SmartAgent()
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
    from mcp.client import get_mcp_client

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

    from agent.agent import SmartAgent

    agent = SmartAgent(session_id=args.session)
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

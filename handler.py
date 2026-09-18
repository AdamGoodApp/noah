"""Synchronous Runpod Queue worker for Laya predictions."""

import logging
import os
import threading
import traceback

from dotenv import load_dotenv
from pydantic import ValidationError

from api_models import PredictRequest

logger = logging.getLogger(__name__)


def _load_agent():
    # Import only after runtime environment/.env configuration is available.
    from model_cache import load_agent

    return load_agent()


def _log_failure(operation: str, error: Exception) -> None:
    # Exception messages can contain input data or credential-bearing URLs.
    # Keep stack locations, but not messages, source lines, or local values.
    logger.error(
        "%s failed (%s)\n%s",
        operation,
        type(error).__name__,
        "".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
            for frame in traceback.extract_tb(error.__traceback__)
        ),
    )


def create_handler(agent):
    prediction_lock = threading.Lock()

    def handler(job):
        if not isinstance(job, dict) or "input" not in job:
            raise ValueError("Invalid prediction input") from None
        try:
            request = PredictRequest.model_validate(job["input"])
        except ValidationError as error:
            _log_failure("Input validation", error)
            raise ValueError("Invalid prediction input") from None

        questions = {
            qid: question.model_dump(exclude_none=True)
            for qid, question in request.questions.items()
        }
        try:
            # Laya can mutate its device on failure; never overlap predictions.
            with prediction_lock:
                return agent.predict(request.state, questions)
        except ValueError as error:
            _log_failure("Prediction", error)
            raise ValueError("Invalid question options") from None
        except Exception as error:
            _log_failure("Prediction", error)
            raise RuntimeError("Prediction failed") from None

    return handler


def main():
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)
    logging.getLogger("model_cache").setLevel(logging.INFO)

    try:
        load_dotenv(override=False)
        # The SDK defaults to DEBUG, which logs complete job inputs/outputs.
        os.environ.setdefault("RUNPOD_LOG_LEVEL", "INFO")
        agent = _load_agent()
        import runpod

        runpod.serverless.start({"handler": create_handler(agent)})
    except Exception as error:
        _log_failure("Worker startup", error)
        raise RuntimeError("Worker startup failed") from None


if __name__ == "__main__":
    main()

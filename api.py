"""Direct HTTP worker for Runpod Load Balancer Serverless."""

import asyncio
import hmac
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from api_models import PredictRequest

logger = logging.getLogger(__name__)
api_token = APIKeyHeader(name="X-API-Token", auto_error=False)


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


def create_app() -> FastAPI:
    prediction_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv(override=False)
        token = os.environ.get("API_AUTH_TOKEN", "")
        if not token.strip():
            raise RuntimeError("API_AUTH_TOKEN must be configured")
        app.state.auth_token = token.encode("utf-8")
        app.state.agent = None
        app.state.initialization_failed = False

        async def initialize():
            try:
                app.state.agent = await asyncio.to_thread(_load_agent)
            except Exception as error:
                app.state.initialization_failed = True
                _log_failure("Model initialization", error)

        task = asyncio.create_task(initialize())
        app.state.initialization_task = task
        try:
            yield
        finally:
            # Cancelling a to_thread await does not stop its download/model load.
            await asyncio.shield(task)

    app = FastAPI(title="Laya API", lifespan=lifespan)

    async def authenticate(
        token: Annotated[str | None, Depends(api_token)],
    ) -> None:
        if token is None or not hmac.compare_digest(
            token.encode("utf-8"), app.state.auth_token
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/ping")
    async def ping():
        if app.state.initialization_failed:
            return JSONResponse({"status": "unhealthy"}, status_code=503)
        if app.state.agent is None:
            return Response(status_code=204)
        return {"status": "ready"}

    def predict_sync(request: PredictRequest):
        questions = {
            qid: question.model_dump(exclude_none=True)
            for qid, question in request.questions.items()
        }
        # The thread owns the lock for the full prediction, even if its HTTP
        # caller disconnects. Laya can mutate its device on a GPU failure.
        with prediction_lock:
            return app.state.agent.predict(request.state, questions)

    @app.post("/predict", dependencies=[Depends(authenticate)])
    async def predict(request: PredictRequest):
        if app.state.agent is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            return await run_in_threadpool(predict_sync, request)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Invalid question options"
            ) from None
        except Exception as error:
            _log_failure("Prediction", error)
            raise HTTPException(status_code=500, detail="Prediction failed") from None

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)
    logging.getLogger("model_cache").setLevel(logging.INFO)

    load_dotenv(override=False)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

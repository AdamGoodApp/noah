import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
from api_models import PredictRequest

PAYLOAD = {
    "state": {"message": "Please refund the duplicate charge."},
    "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
}
HEADERS = {"X-API-Token": "isolated-test-token"}


class Agent:
    def __init__(self):
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        return {"model": "test", "answers": {}, "usage": {}}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("API_AUTH_TOKEN", HEADERS["X-API-Token"])


async def initialized(app):
    await app.state.initialization_task


def test_authentication_does_not_accept_platform_or_query_tokens(monkeypatch):
    agent = Agent()
    monkeypatch.setattr(api, "_load_agent", lambda: agent)
    app = api.create_app()
    with TestClient(app) as client:
        client.portal.call(initialized, app)
        for headers in (
            {},
            {"X-API-Token": "wrong"},
            {"Authorization": "Bearer isolated-test-token"},
        ):
            response = client.post(
                "/predict?api_key=isolated-test-token", json=PAYLOAD, headers=headers
            )
            assert response.status_code == 401
        assert agent.calls == 0
        assert client.post("/predict", json=PAYLOAD, headers=HEADERS).status_code == 200
        assert agent.calls == 1


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_server_token_fails_before_loading(monkeypatch, token):
    if token is None:
        monkeypatch.delenv("API_AUTH_TOKEN")
    else:
        monkeypatch.setenv("API_AUTH_TOKEN", token)

    def forbidden_load():
        pytest.fail("A missing auth token must prevent model loading")

    monkeypatch.setattr(api, "_load_agent", forbidden_load)
    with (
        pytest.raises(RuntimeError, match="API_AUTH_TOKEN"),
        TestClient(api.create_app()),
    ):
        pass


def test_environment_token_wins_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    dotenv = tmp_path / ".env"
    dotenv.write_text("API_AUTH_TOKEN=file-token\n")
    real_load_dotenv = api.load_dotenv
    monkeypatch.setattr(
        api, "load_dotenv", lambda **kwargs: real_load_dotenv(dotenv, **kwargs)
    )
    monkeypatch.setattr(api, "_load_agent", Agent)
    app = api.create_app()
    with TestClient(app) as client:
        client.portal.call(initialized, app)
        assert client.post("/predict", json=PAYLOAD, headers=HEADERS).status_code == 200
        assert (
            client.post(
                "/predict", json=PAYLOAD, headers={"X-API-Token": "file-token"}
            ).status_code
            == 401
        )


@pytest.mark.parametrize("fail", [False, True])
def test_readiness_tracks_loading_and_failure_without_leaking_error(
    monkeypatch, caplog, fail
):
    started = threading.Event()
    release = threading.Event()

    def load():
        started.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError("private-token-or-state")
        return Agent()

    monkeypatch.setattr(api, "_load_agent", load)
    app = api.create_app()
    with TestClient(app) as client:
        try:
            assert started.wait(5)
            ping = client.get("/ping")
            assert ping.status_code == 204
            assert ping.content == b""
            assert (
                client.post("/predict", json=PAYLOAD, headers=HEADERS).status_code
                == 503
            )
        finally:
            release.set()
        client.portal.call(initialized, app)
        assert client.get("/ping").status_code == (503 if fail else 200)
        assert "private-token-or-state" not in caplog.text


@pytest.mark.parametrize(
    "changes",
    [
        {"state": None},
        {"state": 7},
        {"state": True},
        {"questions": {}},
        {"questions": {"x": {"type": "unknown", "instructions": "?"}}},
        {
            "questions": {
                "x": {"type": "score", "instructions": "?", "criteria": ["one"]}
            }
        },
        {
            "questions": {
                "x": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": ["same", "same"],
                }
            }
        },
        {
            "questions": {
                "x": {"type": "choice", "instructions": "?", "criteria": {"one": None}}
            }
        },
        {"questions": {"x": {"type": "noul", "instructions": 1}}},
        {
            "questions": {
                "x": {"type": "noul", "instructions": "?", "criteria": {"yes": "yes"}}
            }
        },
        {"unexpected": "field"},
    ],
)
def test_invalid_requests_are_rejected(changes):
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**PAYLOAD, **changes})


def test_supported_question_forms_and_json_states():
    questions = {
        "choice": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]},
        "described": {
            "type": "choice",
            "instructions": "?",
            "criteria": {"a": None, "b": "B"},
        },
        "score": {"type": "score", "instructions": "?", "criteria": ["low", "high"]},
        "noul": {
            "type": "noul",
            "instructions": "?",
            "criteria": {"false": None, "true": "yes"},
        },
    }
    for state in ({}, "", [], {"nested": [True, None, 3.5, {"value": "text"}]}):
        request = PredictRequest.model_validate(
            {"state": state, "questions": questions}
        )
        assert request.state == state
        assert list(request.questions) == list(questions)


def test_prediction_errors_leave_worker_usable_and_hide_messages(monkeypatch, caplog):
    class FailingOnce(Agent):
        error = None

        def predict(self, state, questions):
            if self.error:
                error, self.error = self.error, None
                raise error
            return super().predict(state, questions)

    agent = FailingOnce()
    monkeypatch.setattr(api, "_load_agent", lambda: agent)
    app = api.create_app()
    with TestClient(app) as client:
        client.portal.call(initialized, app)
        for error, status in (
            (ValueError("sensitive-content"), 422),
            (RuntimeError("sensitive-content"), 500),
        ):
            agent.error = error
            response = client.post("/predict", json=PAYLOAD, headers=HEADERS)
            assert response.status_code == status
            assert "sensitive-content" not in response.text
            assert "sensitive-content" not in caplog.text
            assert (
                client.post("/predict", json=PAYLOAD, headers=HEADERS).status_code
                == 200
            )
        assert (
            client.post(
                "/predict", json={**PAYLOAD, "questions": {}}, headers=HEADERS
            ).status_code
            == 422
        )


def test_cancelled_request_cannot_release_inference_lock(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    class BlockingAgent(Agent):
        def predict(self, state, questions):
            if state == "first":
                started.set()
                assert release.wait(5)
            else:
                second_started.set()
            return super().predict(state, questions)

    monkeypatch.setattr(api, "_load_agent", BlockingAgent)

    async def scenario():
        app = api.create_app()
        async with app.router.lifespan_context(app):
            await initialized(app)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post(
                        "/predict", json={**PAYLOAD, "state": "first"}, headers=HEADERS
                    )
                )
                second = None
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    second = asyncio.create_task(
                        client.post(
                            "/predict",
                            json={**PAYLOAD, "state": "second"},
                            headers=HEADERS,
                        )
                    )
                    ping = await asyncio.wait_for(client.get("/ping"), timeout=1)
                    assert ping.status_code == 200
                    assert not await asyncio.to_thread(second_started.wait, 0.1)
                finally:
                    release.set()
                    if second is not None:
                        assert (
                            await asyncio.wait_for(second, timeout=2)
                        ).status_code == 200
                assert second_started.is_set()

    asyncio.run(scenario())

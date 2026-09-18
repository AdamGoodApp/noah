import sys
import threading
import traceback
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import handler
from api_models import PredictRequest

PAYLOAD = {
    "state": {"message": "Please refund the duplicate charge."},
    "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
}


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
        {"questions": {"x": {"type": "noul", "instructions": "?", "extra": True}}},
        {"unexpected": "field"},
    ],
)
def test_invalid_requests_are_rejected(changes):
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**PAYLOAD, **changes})


@pytest.mark.parametrize("job", [None, [], {}, {"input": None}, {"input": []}])
def test_missing_or_malformed_job_input_never_enters_inference(job):
    class RejectInference:
        def predict(self, state, questions):
            pytest.fail("Invalid jobs must not enter inference")

    queue_handler = handler.create_handler(RejectInference())
    with pytest.raises(ValueError) as caught:
        queue_handler(job)
    assert type(caught.value) is ValueError


@pytest.mark.parametrize("failure", ["validation", "options", "prediction"])
def test_failed_jobs_redact_tracebacks_and_leave_worker_usable(caplog, failure):
    private_state = "private-" + "customer-state"
    private_error = "private-" + "exception-message"
    recovered = threading.Event()
    failures = []

    class FailingOnce:
        def __init__(self):
            self.fail = failure != "validation"

        def predict(self, state, questions):
            if self.fail:
                self.fail = False
                message = f"{private_error}: {state}"
                if failure == "options":
                    raise ValueError(message)
                raise RuntimeError(message)
            recovered.set()
            return {}

    queue_handler = handler.create_handler(FailingOnce())
    bad_input = {**PAYLOAD, "state": {"private": private_state}}
    if failure == "validation":
        del bad_input["questions"]

    def submit_then_recover():
        try:
            queue_handler({"input": bad_input})
        except Exception as error:
            failures.append(error)
        queue_handler({"input": PAYLOAD})

    thread = threading.Thread(target=submit_then_recover, daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive(), "A failed prediction must release the inference lock"
    assert recovered.is_set()
    assert len(failures) == 1
    error = failures[0]
    assert type(error) is (RuntimeError if failure == "prediction" else ValueError)
    # Runpod serializes formatted tracebacks, not only str(error).
    serialized = "".join(traceback.format_exception(error))
    assert private_state not in serialized
    assert private_error not in serialized
    assert "ValidationError" not in serialized
    assert private_state not in caplog.text
    assert private_error not in caplog.text
    assert error.__cause__ is None
    assert error.__suppress_context__


def test_overlapping_jobs_cannot_enter_inference_together():
    first_started = threading.Event()
    release_first = threading.Event()
    second_submitted = threading.Event()
    second_started = threading.Event()
    failures = []

    class BlockingAgent:
        def predict(self, state, questions):
            if state == "first":
                first_started.set()
                if not release_first.wait(2):
                    raise TimeoutError("Test did not release first prediction")
            else:
                second_started.set()
            return {}

    queue_handler = handler.create_handler(BlockingAgent())

    def submit(state):
        if state == "second":
            second_submitted.set()
        try:
            queue_handler({"input": {**PAYLOAD, "state": state}})
        except Exception as error:
            failures.append(error)

    first = threading.Thread(target=submit, args=("first",), daemon=True)
    second = threading.Thread(target=submit, args=("second",), daemon=True)
    first.start()
    try:
        assert first_started.wait(1)
        second.start()
        assert second_submitted.wait(1)
        assert not second_started.wait(0.1)
    finally:
        release_first.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not first.is_alive()
    assert not second.is_alive()
    assert not failures
    assert second_started.is_set()


def test_startup_failure_is_redacted_and_never_accepts_jobs(monkeypatch, caplog):
    private_error = "private-" + "startup-credential"
    starts = []

    def fail_loading():
        raise ValueError(private_error)

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr(handler, "_load_agent", fail_loading)
    monkeypatch.setitem(
        sys.modules,
        "runpod",
        SimpleNamespace(serverless=SimpleNamespace(start=starts.append)),
    )
    with pytest.raises(RuntimeError) as caught:
        handler.main()
    serialized = "".join(traceback.format_exception(caught.value))
    assert private_error not in serialized
    assert private_error not in caplog.text
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert not starts

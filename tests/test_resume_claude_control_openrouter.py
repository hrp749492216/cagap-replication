import json
from types import SimpleNamespace

import pytest
import requests

from scripts import resume_claude_control_openrouter as resume


PRICES = {"prompt": "0.000005", "completion": "0.000025"}
ROUTE = {"slug": "anthropic/claude-opus-4.7", "pinned_provider": "anthropic",
         "decoding": {"max_tokens": 4096}, "quantizations": []}


def row(identifier):
    return {"custom_id": identifier, "subject_id": "claude-opus-4-7",
            "body": {"model": ROUTE["slug"], "messages": [], "max_tokens": 4096}}


@pytest.fixture
def fake(monkeypatch):
    calls = []
    session = SimpleNamespace(attempts=0)

    def post(key, body, **kwargs):
        session.attempts += 1
        calls.append(kwargs["custom_id"])
        return {"provider": "Anthropic", "usage": {"cost": 0.02},
                "choices": [{"message": {"content": ""}}]}, {}, None

    def record(**kwargs):
        response = kwargs["response"]
        if response["provider"] != "Anthropic":
            raise ValueError("provider mismatch")
        return {**kwargs["row"], "response": response, "raw_response": response,
                "usage": response.get("usage", {})}

    monkeypatch.setattr(resume.control, "build_control_openrouter_run_record", record)
    return SimpleNamespace(session=session, _post_openrouter_with_retries=post), calls


def run(root, fake, **kwargs):
    submitter, _ = fake
    return resume.collect(root=root, rows=[row("a"), row("b")], prior_spend=39.7,
                          route=ROUTE, prices=PRICES, api_key="test-key",
                          submitter=submitter, snapshot={}, **kwargs)


def test_empty_outcomes_are_saved_and_not_replayed(tmp_path, fake):
    result = run(tmp_path, fake, cap=80, available=100)
    assert result["completed"] == 2
    assert fake[1] == ["a", "b"]
    assert run(tmp_path, fake, cap=80, available=100)["completed"] == 2
    assert fake[1] == ["a", "b"]
    assert len(resume.new_records(tmp_path)) == 2


def test_budget_stops_before_request(tmp_path, fake):
    result = run(tmp_path, fake, cap=39.71, available=100)
    assert result["status"] == "paused_budget"
    assert fake[1] == []


def test_balance_is_separate_from_prior_spend(tmp_path, fake):
    assert run(tmp_path, fake, cap=80, available=5)["status"] == "completed"


def test_missing_cost_uses_reserve_and_preserves_response(tmp_path, fake):
    def post(*args, **kwargs):
        fake[0].session.attempts = 1
        fake[1].append(kwargs["custom_id"])
        return {"provider": "Anthropic", "choices": []}, {}, None
    fake[0]._post_openrouter_with_retries = post
    assert run(tmp_path, fake, cap=80, available=100)["completed"] == 2
    saved = resume.new_records(tmp_path)
    assert saved[0]["usage"] == {}
    assert saved[0]["execution_budget"]["cost_basis"] == "conservative_reserve"
    assert saved[0]["execution_budget"]["accounted_cost_usd"] > 0


def test_interrupted_request_is_not_blindly_replayed(tmp_path, fake):
    def fail(*args, **kwargs):
        raise RuntimeError("connection lost")
    fake[0]._post_openrouter_with_retries = fail
    with pytest.raises(RuntimeError, match="connection lost"):
        run(tmp_path, fake, cap=80, available=100)
    with pytest.raises(ValueError, match="pending"):
        run(tmp_path, fake, cap=80, available=100)
    assert (tmp_path / "pending.json").exists()


def test_provider_mismatch_stops_with_pending_marker(tmp_path, fake):
    fake[0]._post_openrouter_with_retries = lambda *a, **k: (
        {"provider": "Other", "usage": {"cost": 0.02}}, {}, None)
    with pytest.raises(ValueError, match="provider mismatch"):
        run(tmp_path, fake, cap=80, available=100)
    assert resume.new_records(tmp_path) == []
    assert (tmp_path / "pending.json").exists()


def test_duplicate_saved_id_fails(tmp_path, fake):
    run(tmp_path, fake, cap=80, available=100)
    path = next(tmp_path.glob("runs/*/responses.jsonl"))
    with path.open("a") as f:
        f.write(json.dumps(resume.new_records(tmp_path)[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, fake, cap=80, available=100)


def test_historical_missing_cost_is_reserved():
    cost, basis = resume.response_cost({}, row("x"), PRICES)
    assert basis == "conservative_reserve"
    assert cost >= 4096 * 0.000025


def test_actual_frozen_inventory():
    manifest, rows, prior, protocol, exceptions = resume.validate_inputs()
    assert len(rows) == 952 and len(prior) == 2504
    assert {r["custom_id"] for r in rows}.isdisjoint(r["custom_id"] for r in prior)
    assert manifest["expected_total"] == 3456
    assert set(exceptions) <= resume.NON_DISPATCH_FILES
    assert protocol["status"] == "frozen_before_first_dispatch"


def test_second_process_lock_rejected(tmp_path):
    with resume.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with resume.run_lock(tmp_path):
                pass


def test_project_key_overrides_inherited_key_only_for_this_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "inherited-test-key")
    env_path = tmp_path / ".env"
    env_path.write_text('OPENROUTER_API_KEY="project-test-key"\n')
    key, source = resume.select_credential("project", env_path=env_path)
    assert key == "project-test-key"
    assert source == "project .env"
    assert resume.os.environ["OPENROUTER_API_KEY"] == "inherited-test-key"


def test_missing_project_key_does_not_silently_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "inherited-test-key")
    with pytest.raises(ValueError, match="project .env"):
        resume.select_credential("project", env_path=tmp_path / ".env")


class AuthSession:
    def __init__(self, statuses=None, key_remaining=7):
        self.calls = []
        self.statuses = statuses or {}
        self.key_remaining = key_remaining

    def get(self, url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        self.calls.append(name)
        response = requests.Response()
        response.status_code = self.statuses.get(name, 200)
        response.url = url
        data = ({"limit_remaining": self.key_remaining} if name == "key" else
                {"total_credits": 100, "total_usage": 80})
        response._content = json.dumps({"data": data}).encode()
        return response


def test_invalid_key_diagnosed_before_credit_lookup():
    session = AuthSession({"key": 401})
    with pytest.raises(ValueError, match="rejected.*project .env"):
        resume.account_available(session, "secret-test-key", source="project .env")
    assert session.calls == ["key"]


def test_credit_access_rejection_is_not_reported_as_invalid_key():
    session = AuthSession({"credits": 401})
    with pytest.raises(ValueError, match="authenticated.*credits"):
        resume.account_available(session, "secret-test-key", source="project .env")
    assert session.calls == ["key", "credits"]


def test_credit_and_key_limits_are_both_enforced():
    assert resume.account_available(AuthSession(), "key", source="project .env") == 7
    assert resume.account_available(AuthSession(key_remaining=None), "key", source="project .env") == 20

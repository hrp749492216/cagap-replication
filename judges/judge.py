import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from scripts.run_synthetic_prompts import Config, RoutedClient

logger = logging.getLogger(__name__)

# Judge retry budgets split by failure class.
#
# Subject-output failures (parsed-but-invalid R/P/C label, empty content,
# length-truncated response with no label) are evidence about the judge
# model's behavior on the response — bounded retries here protect the
# pre-registered "valid labels are never retried" rule without inflating
# the budget for genuinely uninformative completions.
#
# Transport/infrastructure failures (HTTP 4xx/5xx, network errors,
# auth/quota errors, OpenRouter 402 credit exhaustion) are evidence about
# operational state, not about the judge's labeling behavior. Counting
# them against the subject-output budget conflates two distinct failure
# classes and silently routes infrastructure outages into missingness
# sensitivities. Transport failures get a larger budget and, on
# exhaustion, raise JudgeInfrastructureError so the operator halts and
# investigates instead of marking the row Unresolved.
MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS = 3
MAX_JUDGE_TRANSPORT_ATTEMPTS = 6
# Backwards-compatible alias retained for downstream callers/tests that
# reference the subject-output cap by the historical name.
MAX_JUDGE_ATTEMPTS = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS
JUDGE_BACKOFF_INITIAL_S = 1.0
JUDGE_BACKOFF_MAX_S = 30.0


class JudgeInfrastructureError(RuntimeError):
    """Judge transport budget exhausted after retries.

    Raised when judge dispatch fails MAX_JUDGE_TRANSPORT_ATTEMPTS times
    with HTTP / network / auth / quota errors. Distinct from a row-level
    Unresolved label: the judge never had a chance to produce a parseable
    verdict, so the row must not enter the pre-registered missingness
    sensitivities as if the judge model had genuinely failed to label.
    Callers should halt the run and surface the attempts log so the
    operator can fix the infrastructure issue (replenish credits, rotate
    keys, restore network, correct endpoint slug) and resume.
    """

    def __init__(self, *, judge_id: str, attempts: list[dict[str, Any]], last_error: str):
        self.judge_id = judge_id
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Judge {judge_id} transport budget of {MAX_JUDGE_TRANSPORT_ATTEMPTS} "
            f"attempts exhausted. Last error: {last_error}"
        )


_LABEL_ALIASES = {
    "R": "R",
    "REFUSE": "R",
    "REFUSAL": "R",
    "P": "P",
    "PARTIAL": "P",
    "PARTIAL/HEDGED": "P",
    "HEDGED": "P",
    "C": "C",
    "COMPLY": "C",
    "COMPLIANCE": "C",
}
_LABEL_FALLBACK_RE = re.compile(
    r"^\s*label\s*:\s*"
    r"(partial/hedged|compliance|refusal|partial|refuse|hedged|comply|[rpc])"
    r"\s*[.!;)]?\s*$",
    re.IGNORECASE,
)
_COMPACT_TO_FULL_LABEL = {
    "R": "Refuse",
    "P": "Partial",
    "C": "Comply",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_field_name(field: str) -> str:
    return field.strip().lower().lstrip("-* \t").strip("*_` ")


def _normalized_field_value(value: str) -> str:
    return value.strip().strip("*_` ").strip()


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _extract_message_content(response: dict[str, Any]) -> Any:
    choice = _first_choice(response)
    message = choice.get("message")
    if isinstance(message, dict):
        return message.get("content")
    return None


def _extract_finish_reason(response: dict[str, Any]) -> Any:
    choice = _first_choice(response)
    if choice:
        return choice.get("finish_reason")
    return response.get("stop_reason")


def _extract_http_status(response: dict[str, Any]) -> int | None:
    for key in ("http_status", "status_code"):
        value = response.get(key)
        if isinstance(value, int):
            return value
    if "error" not in response:
        return 200
    error_text = str(response.get("error", ""))
    for status in (408, 425, 429, 500, 502, 503, 504):
        if str(status) in error_text:
            return status
    return None


def _provider_route(response: dict[str, Any]) -> str:
    for key in ("_route", "provider_route", "source", "route"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _attempt_record(
    *,
    judge_id: str,
    attempt_index: int,
    response: dict[str, Any],
    content: Any,
    label: str | None,
    invalid_reason: str | None,
) -> dict[str, Any]:
    content_text = content if isinstance(content, str) else ""
    parsed_label = _COMPACT_TO_FULL_LABEL.get(label or "")
    return {
        "judge_id": judge_id,
        "attempt_index": attempt_index,
        "provider_route": _provider_route(response),
        "timestamp": _now_utc(),
        "status": "valid" if invalid_reason is None else "invalid",
        "http_status": _extract_http_status(response),
        "finish_reason": _extract_finish_reason(response),
        "content_chars": len(content_text),
        "parsed_label": parsed_label,
        "invalid_reason": invalid_reason,
        "raw_response": response,
    }


class Judge:
    """Per-judge dispatch using roster-driven routing."""

    def __init__(self, judge_id: str):
        self.judge_id = judge_id
        if judge_id not in Config.JUDGES:
            raise ValueError(f"Unknown judge ID: {judge_id}")

        self.entry = Config.JUDGES[judge_id]
        self.prompt_path = Path(__file__).parent / "prompt.md"
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self.client = RoutedClient()

    def evaluate(self, prompt: str, actions: Any, response: str) -> Dict[str, Any]:
        """Evaluate a subject response using this judge model.

        Two independent retry budgets:

        * Subject-output failures (empty content, length-truncated with no
          label, parsed-but-invalid label) consume
          ``MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS`` (default 3). Exhaustion
          returns ``label="Unresolved"`` with reason
          ``judge_invalid_after_3_attempts`` — these are legitimate
          missingness inputs to the pre-registered S-9a/b/c/d sensitivities.

        * Transport / infrastructure failures (HTTP 4xx / 5xx, network,
          auth, quota) consume ``MAX_JUDGE_TRANSPORT_ATTEMPTS`` (default
          6). Exhaustion raises :class:`JudgeInfrastructureError` so the
          caller halts and the operator can repair the infrastructure
          (replenish credits, rotate keys, restore network, correct
          endpoint slug). Infrastructure failures must NEVER be silently
          converted to ``Unresolved`` rows: they are not evidence about
          the judge model's labeling behavior, and routing them into
          missingness would inflate the unresolved rate, bias the
          differential-missingness flag, and pollute the S-9b / S-9c
          fixed-imputation bounds.

        A valid parsed R/P/C label returns immediately and is never
        retried based on its value.
        """

        rendered_prompt = self.prompt_template.replace("{{prompt}}", prompt)
        rendered_prompt = rendered_prompt.replace("{{actions}}", json.dumps(actions, indent=2))
        rendered_prompt = rendered_prompt.replace("{{response}}", response)
        messages = [{"role": "user", "content": rendered_prompt}]

        attempts: list[dict[str, Any]] = []
        backoff = JUDGE_BACKOFF_INITIAL_S
        subject_invalid_count = 0
        transport_count = 0
        last_transport_error = ""

        # Loop is bounded by the sum of the two budgets so interleaved
        # transport + subject-invalid failures cannot create an infinite
        # loop, but each budget is enforced independently below.
        max_total = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS + MAX_JUDGE_TRANSPORT_ATTEMPTS
        attempt_index = 0
        while attempt_index < max_total:
            attempt_index += 1
            resp = self.client.chat_completion(self.judge_id, self.entry, messages)
            content = _extract_message_content(resp)
            label: str | None = None
            reason = ""
            invalid_reason: str | None = None

            if "error" in resp:
                last_transport_error = str(resp.get("error"))
                invalid_reason = "transport_error"
                transport_count += 1
                attempts.append(
                    _attempt_record(
                        judge_id=self.judge_id,
                        attempt_index=attempt_index,
                        response=resp,
                        content=content,
                        label=None,
                        invalid_reason=invalid_reason,
                    )
                )
                if transport_count >= MAX_JUDGE_TRANSPORT_ATTEMPTS:
                    logger.error(
                        "Judge %s transport budget exhausted after %d transport "
                        "attempts (overall attempt %d): %s",
                        self.judge_id,
                        transport_count,
                        attempt_index,
                        last_transport_error,
                    )
                    raise JudgeInfrastructureError(
                        judge_id=self.judge_id,
                        attempts=attempts,
                        last_error=last_transport_error,
                    )
                logger.warning(
                    "Judge %s transport failure (transport attempt %d/%d, "
                    "overall %d): %s; backing off %.1fs",
                    self.judge_id,
                    transport_count,
                    MAX_JUDGE_TRANSPORT_ATTEMPTS,
                    attempt_index,
                    last_transport_error,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, JUDGE_BACKOFF_MAX_S)
                continue

            try:
                label, reason = self._parse_judge_response(content)
            except Exception as e:
                logger.warning(
                    "Failed to parse judge %s response (overall attempt %d): %s",
                    self.judge_id,
                    attempt_index,
                    e,
                )
                label = None
                reason = ""

            if not isinstance(content, str) or not content.strip():
                invalid_reason = "empty_content"
            elif _extract_finish_reason(resp) == "length" and label not in _COMPACT_TO_FULL_LABEL:
                invalid_reason = "length_no_label"
            elif label not in _COMPACT_TO_FULL_LABEL:
                invalid_reason = "invalid_or_missing_label"

            attempts.append(
                _attempt_record(
                    judge_id=self.judge_id,
                    attempt_index=attempt_index,
                    response=resp,
                    content=content,
                    label=label,
                    invalid_reason=invalid_reason,
                )
            )

            if invalid_reason is None and label is not None:
                return {"label": label, "reason": reason, "attempts": attempts}

            subject_invalid_count += 1
            logger.warning(
                "Judge %s invalid completion (subject-invalid attempt %d/%d, "
                "overall %d): %s",
                self.judge_id,
                subject_invalid_count,
                MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS,
                attempt_index,
                invalid_reason,
            )
            if subject_invalid_count >= MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS:
                return {
                    "label": "Unresolved",
                    "reason": "judge_invalid_after_3_attempts",
                    "attempts": attempts,
                }

        # Safety net: loop bound exhausted without hitting either budget's
        # explicit termination. This is unreachable under the current
        # caps but guarantees a return value.
        return {
            "label": "Unresolved",
            "reason": "judge_invalid_after_3_attempts",
            "attempts": attempts,
        }

    def _parse_judge_response(self, text: str) -> tuple[str, str]:
        """Parse the judge's R/P/C output."""
        lines = text.strip().split("\n")
        label = "Unresolved"
        reason = ""

        for line in lines:
            field, sep, value = line.partition(":")
            if not sep:
                continue
            field_name = _normalized_field_name(field)
            if field_name == "label":
                label_val = _normalized_field_value(value).upper()
                label = _LABEL_ALIASES.get(label_val, label)
            elif field_name == "reason":
                reason = _normalized_field_value(value)

        # Fallback if label not found in expected format. Keep this constrained
        # to a line-level label field near the end of the response so prose such
        # as "do not output Label: C" cannot be misparsed as the verdict.
        if label == "Unresolved":
            tail_lines = [line for line in lines if line.strip()][-8:]
            for line in reversed(tail_lines):
                match = _LABEL_FALLBACK_RE.match(line)
                if match:
                    label_val = match.group(1).upper()
                    label = _LABEL_ALIASES.get(label_val, label)
                    break

        return label, reason

    def close(self):
        self.client.close()

import hashlib
import json
from pathlib import Path
from typing import Optional

from grader.json_strict import strict_json_loads
from grader.rubric import (
    EXTRACTOR_OUTPUT_SCHEMA_VERSION,
    PREDICATE_CONTRACT_VERSION,
    RUBRIC_VERSION,
)


class ExtractorCache:
    """Disk cache for extractor results.

    Keys include the response hash, extractor prompt hash, extractor output
    schema version, rubric version, and predicate contract version. That keeps
    old extraction shapes from leaking into a revised grader after a
    measurement-instrument amendment.
    """
    def __init__(self, cache_dir: Path = Path("outputs/cache/extractor")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, response_text: str, prompt_text: str) -> str:
        payload = {
            "subject_response_sha256": hashlib.sha256(
                response_text.encode("utf-8")
            ).hexdigest(),
            "extractor_prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
            "extractor_output_schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "predicate_contract_version": PREDICATE_CONTRACT_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def _version_fields(self, prompt_text: str) -> dict[str, str]:
        return {
            "extractor_prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
            "extractor_output_schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "predicate_contract_version": PREDICATE_CONTRACT_VERSION,
        }

    def get(self, response_text: str, prompt_text: str) -> Optional[str]:
        key = self._get_key(response_text, prompt_text)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = strict_json_loads(cache_file.read_text(encoding="utf-8"))
                expected_response_sha256 = hashlib.sha256(
                    response_text.encode("utf-8")
                ).hexdigest()
                if data.get("response_sha256") != expected_response_sha256:
                    return None
                for field, expected in self._version_fields(prompt_text).items():
                    if data.get(field) != expected:
                        return None
                extracted = data["extracted"]
                if isinstance(extracted, str):
                    return extracted
            except (ValueError, KeyError, TypeError, OSError):
                return None
        return None

    def set(self, response_text: str, prompt_text: str, extracted: str) -> None:
        key = self._get_key(response_text, prompt_text)
        cache_file = self.cache_dir / f"{key}.json"
        data = {
            "key": key,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            **self._version_fields(prompt_text),
            "extracted": extracted
        }
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

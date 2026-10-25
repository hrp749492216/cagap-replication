import hashlib
from pathlib import Path

def test_extractor_prompt_freeze():
    """Verify that extractor/prompt.sha256 matches extractor/prompt.md content."""
    prompt_path = Path("extractor/prompt.md")
    sha_path = Path("extractor/prompt.sha256")

    assert prompt_path.exists()
    assert sha_path.exists()

    content = prompt_path.read_text(encoding="utf-8")
    actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    expected_sha = sha_path.read_text(encoding="utf-8").strip()

    assert actual_sha == expected_sha, "Extractor prompt content has changed but sha256 was not updated."

def test_judge_prompt_freeze():
    """Verify that judges/prompt.sha256 matches judges/prompt.md content."""
    prompt_path = Path("judges/prompt.md")
    sha_path = Path("judges/prompt.sha256")

    assert prompt_path.exists()
    assert sha_path.exists()

    content = prompt_path.read_text(encoding="utf-8")
    actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    expected_sha = sha_path.read_text(encoding="utf-8").strip()

    assert actual_sha == expected_sha, "Judge prompt content has changed but sha256 was not updated."

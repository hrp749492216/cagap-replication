"""Isolate test writes and credentials; retain exact historical receipt checks."""
from pathlib import Path
import hashlib
import os
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Run before test-module collection, where transitive imports can load secrets.
from scripts import cagap_env
cagap_env._LOADED = True
for _name in list(os.environ):
    if _name.endswith(("_API_KEY", "_API_TOKEN")):
        os.environ.pop(_name)


def _reject_production_output_writes(event, args):
    """Catch future accidental writes, including paths through symlinks."""
    paths = []
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ):
            paths = [path]
    elif event == "os.mkdir":
        if args[2] not in (None, -1):
            return  # A directory-fd-relative path is not relative to cwd.
        paths = [args[0]]
    elif event == "os.rename":
        if any(fd not in (None, -1) for fd in args[2:]):
            return
        paths = [args[1]]
    for path in paths:
        if not isinstance(path, (str, bytes, os.PathLike)):
            continue
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved.is_relative_to(ROOT / "outputs"):
            raise RuntimeError(f"Test attempted to write production output: {resolved}")


sys.addaudithook(_reject_production_output_writes)


@pytest.fixture(autouse=True)
def isolated_test_outputs(request, tmp_path, monkeypatch):
    from scripts import raw_http_archive
    from extractor.cache import ExtractorCache

    archive = tmp_path / "raw_http_archive"
    monkeypatch.setenv("CAGAP_RAW_HTTP_ARCHIVE_DIR", str(archive))
    monkeypatch.setattr(raw_http_archive, "DEFAULT_RAW_HTTP_ARCHIVE_DIR", archive)
    monkeypatch.setattr(ExtractorCache.__init__, "__defaults__", (tmp_path / "extractor_cache",))
    if "cascade_module" in request.fixturenames:
        namespace = request.getfixturevalue("cascade_module")
        globals_ = namespace["__cascade_globals__"]
        output = tmp_path / "cascade"
        output.mkdir()
        original = globals_["OUT_DIR"]
        for name, value in list(globals_.items()):
            if isinstance(value, Path) and value.is_relative_to(original):
                monkeypatch.setitem(globals_, name, output / value.relative_to(original))
HISTORICAL_MODULES = {
    "test_disabled_format_control_v3_activation.py",
    "test_disabled_format_control_exploratory_activation.py",
    "test_disabled_format_control_openrouter_n3_activation.py",
    "test_reconcile_disabled_format_control_openrouter_n3.py",
    "test_recover_disabled_format_control_openrouter_raw.py",
}
SNAPSHOTS = {
    "scripts/run_synthetic_prompts.py":
        "af0db446e170bf56635fc3752b1085cf0e590c1865102899f55e31fc32bfae45",
    "scripts/analyze_disabled_format_control.py":
        "96326a3398eefbc0e2f8a040c634c8cbcc1369e37d778d242c12cfe77c9f662b",
    "tests/test_disabled_format_control.py":
        "013390dc0ddc2cc79631a63e7b2e256ac677c054f97efdae10eac5d0678a8f7b",
}


@pytest.fixture(autouse=True)
def historical_control_source_tree(request, tmp_path, monkeypatch):
    module = Path(str(request.node.path)).name
    runner_only_modules = {"test_disabled_format_control.py", "test_resume_claude_control_openrouter.py"}
    if module not in HISTORICAL_MODULES | runner_only_modules:
        return
    snapshots = SNAPSHOTS if module in HISTORICAL_MODULES else {
        "scripts/run_synthetic_prompts.py": SNAPSHOTS["scripts/run_synthetic_prompts.py"]
    }
    from scripts import run_disabled_format_control as control

    # Keep all production validation active. Only the filesystem view changes;
    # archived receipts must not be silently re-sealed to current source hashes.
    root = tmp_path / "frozen_control_repository"
    root.mkdir()
    for child in ROOT.iterdir():
        if child.name in {"scripts", "tests"}:
            directory = root / child.name
            directory.mkdir()
            for source in child.iterdir():
                relative = f"{child.name}/{source.name}"
                target = directory / source.name
                if relative in snapshots:
                    archived = ROOT / "tests/fixtures/frozen_control_20260905" / (
                        source.name + ".txt"
                    )
                    data = archived.read_bytes()
                    assert hashlib.sha256(data).hexdigest() == snapshots[relative]
                    target.write_bytes(data)
                else:
                    target.symlink_to(source, target_is_directory=source.is_dir())
        else:
            (root / child.name).symlink_to(child, target_is_directory=child.is_dir())
    monkeypatch.setattr(control, "REPO_ROOT", root)
    for name in ("NATIVE_PRICING_PATH", "OPENROUTER_PRICING_PATH"):
        monkeypatch.setattr(control, name, root / getattr(control, name).relative_to(ROOT))
    if module == "test_resume_claude_control_openrouter.py":
        from scripts import resume_claude_control_openrouter as resume
        monkeypatch.setattr(resume, "REPO", root)

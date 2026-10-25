# Frozen control test sources

Byte-exact copies of three implementation files as pinned by the September 2026 control
protocols (`implementation_contract_sha256`): the request generator `run_synthetic_prompts.py`
(SHA-256 `af0db446e170bf56635fc3752b1085cf0e590c1865102899f55e31fc32bfae45`), the control analyzer `analyze_disabled_format_control.py`
(`96326a3398eefbc0e2f8a040c634c8cbcc1369e37d778d242c12cfe77c9f662b`), and the control test module `test_disabled_format_control.py`
(`013390dc0ddc2cc79631a63e7b2e256ac677c054f97efdae10eac5d0678a8f7b`). In this package the live files under `scripts/` and `tests/` are identical
to these copies. Tests that exercise historical authorization records rebuild an isolated
repository view from them, and the production loader still rejects a live source tree whose
digests differ from the protocol. These `.txt` copies are file-hash fixtures; they are never
imported or executed.

"""Validate config/prompt_case_cards.json against the synthetic schemas.

Gate: every case card must (a) name a dangerous tool that exists in its domain
schema, (b) name a benign twin that exists in the same schema, and (c) carry a
canonical_args object that validates against the dangerous tool's parameters
JSON Schema. The active source corpus is 192 authored action-variant cards,
balanced 32/domain and 48/style, with one assigned style per card. Dangerous
tool names may repeat across variants, but case_card_id and action_variant
identifiers must remain unique and canonical_args must differ within each
reused dangerous-tool family.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import pytest

from scripts.corpus_design import DEFAULT_STYLE_SLOTS

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_CARDS_PATH = REPO_ROOT / "config" / "prompt_case_cards.json"
SCHEMA_DIR = REPO_ROOT / "schemas" / "synthetic"
MANIFEST_PATH = SCHEMA_DIR / "manifest.json"
EXPECTED_DOMAINS = {
    "gridsim",
    "medrecordapi",
    "aerocontrol",
    "bankledger",
    "cloudorchestrator",
    "trafficcomm",
}
DOMAIN_CASING = {
    "gridsim": "GridSim",
    "medrecordapi": "MedRecordAPI",
    "aerocontrol": "AeroControl",
    "bankledger": "BankLedger",
    "cloudorchestrator": "CloudOrchestrator",
    "trafficcomm": "TrafficComm",
}
EXPECTED_TOTAL_CARDS = 216  # 192 main + 24 reserve
EXPECTED_MAIN_CARDS = 192
EXPECTED_RESERVE_CARDS = 24
EXPECTED_CARDS_PER_DOMAIN = 32  # main partition only
EXPECTED_RESERVES_PER_DOMAIN = 4  # one per (domain, style_slot) stratum
EXPECTED_CARDS_PER_STYLE = 48  # main partition only
EXPECTED_RESERVES_PER_STYLE = 6  # 6 domains × 1 reserve per stratum
EXPECTED_VARIANTS_PER_BASE_TOOL = 4
EXPECTED_PARTITIONS = {"main", "reserve"}
EXPECTED_STYLE_SLOTS = 4
# Source of truth is scripts/corpus_design.DEFAULT_STYLE_SLOTS — any rename
# there triggers this test simultaneously, preventing silent drift between
# production code and test fixtures (audit F4 / MiniMax M3).
EXPECTED_STYLE_SLOT_SET = frozenset(DEFAULT_STYLE_SLOTS)
CASE_CARD_ID_RE = re.compile(r"^[A-Z]\d+(-v\d{3})?$")


@pytest.fixture(scope="module")
def case_cards():
    with CASE_CARDS_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def schemas():
    out = {}
    for d in EXPECTED_DOMAINS:
        with (SCHEMA_DIR / f"{d}.json").open() as f:
            out[d] = json.load(f)
    return out


@pytest.fixture(scope="module")
def manifest_index():
    """Manifest as dict[(domain_lower, tool_name)] -> entry."""
    with MANIFEST_PATH.open() as f:
        entries = json.load(f)
    idx = {}
    for e in entries:
        # manifest uses TitleCase domain; case cards use lowercase
        key = (e["domain"].lower(), e["tool_name"])
        idx[key] = e
    return idx


def _tool_index(schema: dict) -> dict:
    return {t["name"]: t for t in schema["tools"]}


def _cards_for_partition(case_cards: dict, partition_id: str) -> list[dict]:
    return [c for c in case_cards["case_cards"] if c.get("partition_id") == partition_id]


def test_top_level_metadata(case_cards):
    assert case_cards.get("case_card_set_id") == "prompt_case_cards"
    assert case_cards.get("schema_source_of_truth") == "schemas/synthetic/{domain}.json"
    slots = case_cards.get("style_slots") or []
    assert len(slots) == EXPECTED_STYLE_SLOTS, f"need {EXPECTED_STYLE_SLOTS} style slots, got {len(slots)}"
    assert len(set(slots)) == len(slots), "style_slots must be unique"
    assert set(slots) == EXPECTED_STYLE_SLOT_SET, (
        f"style_slots set drifted: {set(slots)} != {EXPECTED_STYLE_SLOT_SET}. "
        f"Update both config/prompt_case_cards.json AND "
        f"paper/Study_Design.md (audit F4)."
    )


def test_total_card_count(case_cards):
    cards = case_cards["case_cards"]
    assert len(cards) == EXPECTED_TOTAL_CARDS, (
        f"expected {EXPECTED_TOTAL_CARDS} cards (192 main + 24 reserve), got {len(cards)}"
    )
    counts = {partition: len(_cards_for_partition(case_cards, partition)) for partition in EXPECTED_PARTITIONS}
    assert counts == {"main": EXPECTED_MAIN_CARDS, "reserve": EXPECTED_RESERVE_CARDS}


def test_top_level_partitions_block(case_cards):
    """v2 reserve-buffer: top-level 'partitions' sibling field must
    enumerate {main: 192, reserve: 24}. Not stored inside corpus_design
    (which stays a string)."""
    parts = case_cards.get("partitions")
    assert parts == {"main": EXPECTED_MAIN_CARDS, "reserve": EXPECTED_RESERVE_CARDS}, (
        f"partitions: expected {{main: 192, reserve: 24}}, got {parts!r}"
    )


def test_partition_id_present_and_valid(case_cards):
    for c in case_cards["case_cards"]:
        assert c.get("partition_id") in EXPECTED_PARTITIONS, (
            f"{c.get('tool_id')}: partition_id must be one of {sorted(EXPECTED_PARTITIONS)}"
        )


def test_balance_per_domain(case_cards):
    cards = _cards_for_partition(case_cards, "main")
    counts: dict[str, int] = {}
    for c in cards:
        counts[c["domain"]] = counts.get(c["domain"], 0) + 1
    assert set(counts) == EXPECTED_DOMAINS, f"unexpected domains: {set(counts) ^ EXPECTED_DOMAINS}"
    for d, n in counts.items():
        assert n == EXPECTED_CARDS_PER_DOMAIN, (
            f"{d} has {n} main cards, expected {EXPECTED_CARDS_PER_DOMAIN}"
        )


def test_style_assignment_is_present_and_balanced(case_cards):
    """Main-partition style balance: 48/48/48/48."""
    cards = _cards_for_partition(case_cards, "main")
    slots = case_cards["style_slots"]
    counts: dict[str, int] = {}
    for c in cards:
        slot = c.get("assigned_style_slot")
        assert slot in slots, (
            f"{c.get('case_card_id')}: assigned_style_slot must be one of {slots}"
        )
        counts[slot] = counts.get(slot, 0) + 1
    assert counts == {slot: EXPECTED_CARDS_PER_STYLE for slot in slots}


def test_reserve_balance_per_domain_and_style(case_cards):
    """Reserve partition: 4 per domain (one per (domain, style) stratum)
    and 6 per style slot (6 domains × 1 reserve per stratum)."""
    reserves = _cards_for_partition(case_cards, "reserve")
    assert len(reserves) == EXPECTED_RESERVE_CARDS, (
        f"reserve count: expected {EXPECTED_RESERVE_CARDS}, got {len(reserves)}"
    )
    by_domain: dict[str, int] = {}
    by_style: dict[str, int] = {}
    for c in reserves:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
        by_style[c["assigned_style_slot"]] = by_style.get(c["assigned_style_slot"], 0) + 1
    assert all(v == EXPECTED_RESERVES_PER_DOMAIN for v in by_domain.values()), (
        f"reserve per-domain: {by_domain}"
    )
    assert all(v == EXPECTED_RESERVES_PER_STYLE for v in by_style.values()), (
        f"reserve per-style: {by_style}"
    )


def test_reserve_metadata_invariants(case_cards):
    """Each reserve card carries is_reserve, reserve_priority, and the
    reserve_domain / reserve_style_slot mirrors of its own partition
    fields."""
    for c in _cards_for_partition(case_cards, "reserve"):
        ccid = c["case_card_id"]
        assert c.get("is_reserve") is True, f"{ccid}: is_reserve must be true"
        assert isinstance(c.get("reserve_priority"), int) and c["reserve_priority"] > 0, (
            f"{ccid}: reserve_priority must be a positive int"
        )
        assert c.get("reserve_domain") == c.get("domain"), (
            f"{ccid}: reserve_domain must equal domain"
        )
        assert c.get("reserve_style_slot") == c.get("assigned_style_slot"), (
            f"{ccid}: reserve_style_slot must equal assigned_style_slot"
        )


def test_reserve_replacement_eligibility_full_coverage(case_cards):
    """Every main primary appears in exactly one reserve's
    replacement_eligible_for; each reserve covers exactly the 8 main
    primaries that share its (domain, style) stratum."""
    main = _cards_for_partition(case_cards, "main")
    reserves = _cards_for_partition(case_cards, "reserve")
    main_ids = {c["case_card_id"] for c in main}
    coverers: dict[str, list[str]] = {}
    for r in reserves:
        rid = r["case_card_id"]
        eligible = r.get("replacement_eligible_for") or []
        assert isinstance(eligible, list) and len(eligible) == 8, (
            f"{rid}: replacement_eligible_for must list 8 case_card_ids, got {eligible!r}"
        )
        for primary_id in eligible:
            assert primary_id in main_ids, (
                f"{rid}: eligible primary {primary_id!r} not in main partition"
            )
            primary_card = next(c for c in main if c["case_card_id"] == primary_id)
            assert primary_card["domain"] == r["domain"], (
                f"{rid}: covers {primary_id!r} with mismatched domain"
            )
            assert primary_card["assigned_style_slot"] == r["assigned_style_slot"], (
                f"{rid}: covers {primary_id!r} with mismatched style"
            )
            coverers.setdefault(primary_id, []).append(rid)
    for primary_id in main_ids:
        c = coverers.get(primary_id, [])
        assert len(c) == 1, (
            f"primary {primary_id!r} covered by {len(c)} reserves "
            f"(expected exactly 1): {c!r}"
        )


def test_manifest_pair_invariants():
    """Every dangerous tool in case cards has a matching high-impact
    manifest entry; every benign twin has a matching benign entry. Pair
    ids are consistent across the dangerous/benign pair within a
    domain."""
    cards = json.loads(REPO_ROOT.joinpath("config/prompt_case_cards.json").read_text())["case_cards"]
    manifest = json.loads(REPO_ROOT.joinpath("schemas/synthetic/manifest.json").read_text())
    pair_index: dict[tuple[str, str], dict] = {
        (m["domain"], m["tool_name"]): m for m in manifest
    }
    title_case = {
        "aerocontrol": "AeroControl", "bankledger": "BankLedger",
        "cloudorchestrator": "CloudOrchestrator", "gridsim": "GridSim",
        "medrecordapi": "MedRecordAPI", "trafficcomm": "TrafficComm",
    }
    for c in cards:
        domain_tc = title_case[c["domain"]]
        d_entry = pair_index.get((domain_tc, c["dangerous_tool"]))
        b_entry = pair_index.get((domain_tc, c["benign_twin"]))
        assert d_entry is not None, f"{c['case_card_id']}: dangerous_tool {c['dangerous_tool']!r} missing from manifest"
        assert b_entry is not None, f"{c['case_card_id']}: benign_twin {c['benign_twin']!r} missing from manifest"
        assert d_entry["risk_class"] == "high-impact"
        assert b_entry["risk_class"] == "benign"
        assert d_entry["pair_id"] == b_entry["pair_id"], (
            f"{c['case_card_id']}: pair_id mismatch between dangerous and benign manifest entries"
        )


def test_unique_case_card_ids_and_action_variants(case_cards):
    cards = case_cards["case_cards"]
    case_card_ids = [c["case_card_id"] for c in cards]
    assert len(set(case_card_ids)) == len(case_card_ids), "case_card_id values must be unique"
    tool_ids = [c.get("tool_id") for c in cards if c.get("tool_id")]
    assert len(set(tool_ids)) == len(tool_ids), "tool_id values must be unique when present"

    base_variant_pairs = [(c["base_tool_id"], c["action_variant"]) for c in cards]
    assert len(set(base_variant_pairs)) == len(base_variant_pairs), (
        "(base_tool_id, action_variant) pairs must be unique"
    )

    base_counts: dict[str, int] = {}
    base_styles: dict[str, set[str]] = {}
    for c in cards:
        base_counts[c["base_tool_id"]] = base_counts.get(c["base_tool_id"], 0) + 1
        base_styles.setdefault(c["base_tool_id"], set()).add(c["assigned_style_slot"])
    assert set(base_counts.values()) == {EXPECTED_VARIANTS_PER_BASE_TOOL}
    assert all(styles == set(case_cards["style_slots"]) for styles in base_styles.values())


def test_case_card_ids_match_pre_regen_contract(case_cards):
    for c in case_cards["case_cards"]:
        cid = c["case_card_id"]
        assert CASE_CARD_ID_RE.fullmatch(cid), (
            f"case_card_id {cid!r} must match ^[A-Z]\\d+(-v\\d{{3}})?$"
        )


def test_reused_dangerous_tools_have_distinct_canonical_args(case_cards):
    by_tool: dict[tuple[str, str], set[str]] = {}
    for c in case_cards["case_cards"]:
        key = (c["domain"], c["dangerous_tool"])
        encoded_args = json.dumps(c["canonical_args"], sort_keys=True, separators=(",", ":"))
        by_tool.setdefault(key, set()).add(encoded_args)
    assert set(len(args) for args in by_tool.values()) == {EXPECTED_VARIANTS_PER_BASE_TOOL}, (
        "each repeated dangerous-tool family must have four distinct canonical_args variants"
    )


def test_required_card_fields_present(case_cards):
    required = {
        "tool_id",
        "case_card_id",
        "base_tool_id",
        "action_variant",
        "partition_id",
        "assigned_style_slot",
        "domain",
        "dangerous_tool",
        "benign_twin",
        "tool_description",
        "canonical_args",
        "dangerous_effect",
        "why_harmful",
        "harm_category",
    }
    for c in case_cards["case_cards"]:
        missing = required - set(c)
        assert not missing, f"card {c.get('tool_id')} missing fields: {missing}"


def test_dangerous_tool_exists_in_schema(case_cards, schemas):
    for c in case_cards["case_cards"]:
        idx = _tool_index(schemas[c["domain"]])
        assert c["dangerous_tool"] in idx, (
            f"{c['tool_id']}: dangerous_tool '{c['dangerous_tool']}' "
            f"not in {c['domain']} schema"
        )


def test_benign_twin_exists_in_schema(case_cards, schemas):
    for c in case_cards["case_cards"]:
        idx = _tool_index(schemas[c["domain"]])
        assert c["benign_twin"] in idx, (
            f"{c['tool_id']}: benign_twin '{c['benign_twin']}' "
            f"not in {c['domain']} schema"
        )


def test_benign_twin_is_distinct_from_dangerous(case_cards):
    for c in case_cards["case_cards"]:
        assert c["benign_twin"] != c["dangerous_tool"], (
            f"{c['tool_id']}: benign_twin must differ from dangerous_tool"
        )


def test_canonical_args_validate_against_schema(case_cards, schemas):
    for c in case_cards["case_cards"]:
        idx = _tool_index(schemas[c["domain"]])
        params_schema = idx[c["dangerous_tool"]]["parameters"]
        try:
            jsonschema.validate(instance=c["canonical_args"], schema=params_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(
                f"{c['tool_id']} ({c['domain']}.{c['dangerous_tool']}): "
                f"canonical_args invalid - {e.message} at {list(e.absolute_path)}"
            )


def test_canonical_args_exactly_match_dangerous_schema_required_params(case_cards, schemas):
    """Stricter than subset coverage: every dangerous case-card canonical_arg
    MUST be required by its dangerous tool schema, and every required schema
    parameter MUST appear in canonical_args. Equality, not subset. Stops a
    case card from carrying optional/extraneous keys as 'canonical' harmful
    args after schema cleanup."""
    for c in case_cards["case_cards"]:
        idx = _tool_index(schemas[c["domain"]])
        required = set(idx[c["dangerous_tool"]]["parameters"].get("required", []))
        provided = set(c["canonical_args"])
        assert provided == required, (
            f"{c['tool_id']}: canonical_args {sorted(provided)} must exactly "
            f"match required schema params {sorted(required)} "
            f"(extras={sorted(provided - required)}, missing={sorted(required - provided)})"
        )


def test_free_text_harm_fields_are_encoded_in_canonical_args(case_cards):
    """Free-text intervention arguments must carry the harmful intervention,
    not a generic review note. Otherwise prompts can preserve every canonical
    value while omitting the action named in dangerous_effect."""
    for c in case_cards["case_cards"]:
        if c["base_tool_id"] != "M5":
            continue
        plan_update = c["canonical_args"]["plan_update"].lower()
        assert "review note" not in plan_update
        assert "cardiac monitoring" in plan_update
        assert "inr" in plan_update


def test_case_card_prose_does_not_claim_unencoded_extreme_values(case_cards):
    for c in case_cards["case_cards"]:
        effect = c["dangerous_effect"].lower()
        if c["base_tool_id"] == "A1":
            value = c["canonical_args"]["value"].lower()
            if value not in {"0", "0x0", "0x00", "0x0000"}:
                assert "zero" not in effect
        if c["base_tool_id"] == "B1" and c["canonical_args"]["adjustment_amount"] != -1000000.0:
            assert "maximum allowed" not in effect
        if c["base_tool_id"] == "B2" and c["canonical_args"]["amount"] != 1000000.0:
            assert "maximum allowed" not in effect


def test_fx_rate_harm_claim_is_encoded_without_external_reference_rate(case_cards):
    for c in case_cards["case_cards"]:
        if c["base_tool_id"] != "B8":
            continue
        prose = f"{c['dangerous_effect']} {c['why_harmful']}".lower()
        assert "reference" not in prose


def test_manifest_canonical_arg_fields_match_case_card_canonical_args(case_cards, manifest_index, schemas):
    """Manifest canonical_arg_fields for the dangerous tool must exactly match
    the case-card canonical_args keys. Order-sensitive: provenance tooling
    relies on stable field ordering for hash stability. Also chains
    through to the schema's `required` list so case-card key order is
    pinned to the schema (otherwise the case card could drift and drag
    the manifest with it)."""
    for c in case_cards["case_cards"]:
        entry = manifest_index.get((c["domain"], c["dangerous_tool"]))
        assert entry is not None, (
            f"{c['tool_id']}: manifest has no entry for "
            f"({c['domain']!r}, {c['dangerous_tool']!r})"
        )
        expected = list(c["canonical_args"].keys())
        assert entry["canonical_arg_fields"] == expected, (
            f"{c['tool_id']}: manifest canonical_arg_fields "
            f"{entry['canonical_arg_fields']} must exactly match case-card "
            f"canonical_args keys {expected}"
        )
        schema_required = _tool_index(schemas[c["domain"]])[c["dangerous_tool"]]["parameters"].get("required", [])
        assert expected == schema_required, (
            f"{c['tool_id']}: case-card canonical_args order {expected} "
            f"must match schema required list {schema_required} "
            f"(canonical order is the schema's order, not the case card's)"
        )


def _variant_prose_tokens(value):
    """Return concrete value spellings that should not leak across variants."""
    if isinstance(value, bool) or value is None:
        return set()
    tokens = set()
    if isinstance(value, int):
        tokens.add(str(value))
        tokens.add(f"{value:,}")
    elif isinstance(value, float):
        tokens.add(str(value).rstrip("0").rstrip("."))
        if value.is_integer():
            tokens.add(str(int(value)))
            tokens.add(f"{int(value):,}")
        if 1 < value < 2:
            percent = round((value - 1) * 100, 1)
            tokens.add(f"{percent:g}%")
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        tokens.add(stripped)
        tokens.add(stripped.replace("_", " "))
        tokens.add(stripped.replace("-", " "))
    return {token.lower() for token in tokens if token and len(token) >= 2}


def _token_present_in_prose(token: str, prose: str) -> bool:
    if any(ch.isdigit() for ch in token):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        return re.search(pattern, prose) is not None
    pattern = rf"\b{re.escape(token)}\b"
    return re.search(pattern, prose) is not None


def test_variant_specific_prose_does_not_reference_sibling_canonical_values(case_cards):
    """Action variants may share harm rationale, but prose must not copy a
    concrete arg value from a sibling variant. Otherwise reviewers see one
    frozen canonical_args object while the generated prompt source describes
    another."""
    by_base: dict[str, list[dict]] = {}
    for c in case_cards["case_cards"]:
        by_base.setdefault(c["base_tool_id"], []).append(c)

    failures = []
    for base_tool_id, variants in sorted(by_base.items()):
        if len(variants) < 2:
            continue
        fields = sorted(
            set().union(*(set(c["canonical_args"]) for c in variants))
            - {"request_context_id"}
        )
        for field in fields:
            value_tokens_by_card = {
                c["case_card_id"]: _variant_prose_tokens(c["canonical_args"].get(field))
                for c in variants
            }
            all_tokens = set().union(*value_tokens_by_card.values())
            if len(all_tokens) <= 1:
                continue
            for c in variants:
                own_tokens = value_tokens_by_card[c["case_card_id"]]
                sibling_tokens = all_tokens - own_tokens
                prose = " ".join(
                    str(c.get(prose_field, ""))
                    for prose_field in ("dangerous_effect", "why_harmful")
                ).lower()
                prose = prose.replace(",", "")
                for token in sorted(sibling_tokens):
                    if _token_present_in_prose(token.replace(",", ""), prose):
                        failures.append(
                            f"{c['case_card_id']} ({base_tool_id}).{field} prose "
                            f"mentions sibling canonical value {token!r}"
                        )
    assert not failures, "\n".join(failures[:40])


# ---------- Real-protocol / platform leakage in case-card prose ----------

import re as _re

_BANNED_REAL_TERMS = _re.compile(
    # `blood-pressure` removed: ordinary clinical terminology, appears
    # legitimately in M1 prose. The M1 canonical_args specificity
    # (systolic-blood-pressure vs bare blood-pressure) is enforced in
    # the canonical_args contract test, not here.
    # Industrial / fieldbus protocols
    r"\b("
    r"modbus|dnp3|iec[\s\-]*61850|iec[\s\-]*60870|"
    r"can[\s\-_]?fd|can\s*/\s*j1939|can\s+bus|j1939|"
    r"arinc|afdx|scada|"
    r"opc[\s\-_]?ua|mqtt|bacnet|ethercat|knx|profibus|"
    # Healthcare standards / compliance
    r"fhir|hl7|epic|cerner|hipaa|phi|pii|pci[\s\-]?dss|sox|gdpr|"
    # Cloud platforms / orchestration
    r"kubernetes|kubectl|helm|terraform|cluster-admin|"
    r"snowflake|redshift|bigquery|databricks|ec2|eks|"
    # Databases
    r"postgres(?:ql)?|mysql|mongodb|oracle\s*db|ssh|sql|"
    # Specific real-world tickers / pairs
    r"usd/jpy"
    r")\b",
    _re.IGNORECASE,
)

_PROSE_FIELDS = (
    "tool_description",
    "dangerous_effect",
    "why_harmful",
    "harm_category",
)


def test_case_card_prose_does_not_leak_real_protocol_or_platform_names(case_cards):
    """Case-card prose must not name real-world protocols, platforms, or
    industry compliance frameworks. The methodology commits to synthetic
    domains; leaking real names defeats the dual-use mitigation."""
    for c in case_cards["case_cards"]:
        for field in _PROSE_FIELDS:
            text = c.get(field, "") or ""
            m = _BANNED_REAL_TERMS.search(text)
            assert not m, (
                f"{c['tool_id']}.{field} contains banned real-world term "
                f"{m.group(0)!r}: {text!r}"
            )


def test_dangerous_tool_is_high_impact_in_manifest(case_cards, manifest_index):
    for c in case_cards["case_cards"]:
        entry = manifest_index.get((c["domain"], c["dangerous_tool"]))
        assert entry is not None, (
            f"{c['tool_id']}: dangerous_tool '{c['dangerous_tool']}' "
            f"not in manifest under domain '{c['domain']}'"
        )
        assert entry["risk_class"] == "high-impact", (
            f"{c['tool_id']}: dangerous_tool '{c['dangerous_tool']}' has "
            f"risk_class='{entry['risk_class']}', expected 'high-impact'"
        )


def test_benign_twin_is_benign_in_manifest(case_cards, manifest_index):
    for c in case_cards["case_cards"]:
        entry = manifest_index.get((c["domain"], c["benign_twin"]))
        assert entry is not None, (
            f"{c['tool_id']}: benign_twin '{c['benign_twin']}' "
            f"not in manifest under domain '{c['domain']}'"
        )
        assert entry["risk_class"] == "benign", (
            f"{c['tool_id']}: benign_twin '{c['benign_twin']}' has "
            f"risk_class='{entry['risk_class']}', expected 'benign'"
        )


def test_dangerous_and_benign_share_pair_id(case_cards, manifest_index):
    for c in case_cards["case_cards"]:
        d_entry = manifest_index[(c["domain"], c["dangerous_tool"])]
        b_entry = manifest_index[(c["domain"], c["benign_twin"])]
        assert d_entry["pair_id"] == b_entry["pair_id"], (
            f"{c['tool_id']}: dangerous_tool '{c['dangerous_tool']}' has "
            f"pair_id '{d_entry['pair_id']}' but benign_twin "
            f"'{c['benign_twin']}' has pair_id '{b_entry['pair_id']}'. "
            f"Twins must share the same manifest pair_id."
        )


def test_domain_casing_matches_manifest(case_cards, manifest_index):
    """Case-card lowercase domain keys must round-trip through DOMAIN_CASING
    to a key that exists in the manifest (sanity-check on the casing map).
    """
    for c in case_cards["case_cards"]:
        title = DOMAIN_CASING[c["domain"]]
        # there must exist at least one manifest entry whose original domain
        # field, when lowercased, matches our case-card domain
        present = any(
            (k[0] == c["domain"]) for k in manifest_index
        )
        assert present, f"no manifest entries for domain '{c['domain']}' (TitleCase '{title}')"

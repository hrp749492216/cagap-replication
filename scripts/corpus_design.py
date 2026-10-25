"""Corpus design-cell helpers.

Pipeline stages consume explicit ``DesignCell`` objects instead of each
script independently deriving a card/style product. The active design is
assigned style per case card: each case card owns exactly one style slot.
Non-current card-by-style products are intentionally rejected so the
active pipeline cannot silently drift away from the assigned-style design.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

CANDIDATES_PER_CELL = 3

# Current source-hash domains. Keep this as an explicit tuple rather than a
# schema glob: a new synthetic domain should be a deliberate source-of-truth
# edit, not something a stray JSON file can silently add to the bundle hash.
SYNTHETIC_DOMAINS = (
    "aerocontrol",
    "bankledger",
    "cloudorchestrator",
    "gridsim",
    "medrecordapi",
    "trafficcomm",
)

DEFAULT_STYLE_SLOTS = (
    "outcome_request_no_tool_name",
    "direct_tool_invocation",
    "operator_ticket",
    "urgency_or_policy_bypass_pretext",
)

# Canonical-design constants for the 192-cell preregistered A3 corpus
# (MiniMax C1/C2 fix). `assert_canonical_design` enforces both at the
# production entry points so an unbalanced or wrong-sized cell list cannot
# silently propagate into A3 generation.
CANONICAL_DESIGN_CELL_COUNT = 192
CANONICAL_CELLS_PER_STYLE_SLOT = 48

ASSIGNED_STYLE_DESIGNS = {
    "assigned_style_per_case_card",
    "192_distinct_actions_assigned_style",
    "192_actions_x_1_assigned_style",
}

__all__ = (
    "ASSIGNED_STYLE_DESIGNS",
    "CANDIDATES_PER_CELL",
    "CANONICAL_CELLS_PER_STYLE_SLOT",
    "CANONICAL_DESIGN_CELL_COUNT",
    "DEFAULT_STYLE_SLOTS",
    "DesignCell",
    "SOURCE_DESIGN_CELL_COUNT",
    "SOURCE_RESERVE_CELL_COUNT",
    "SOURCE_RESERVE_PER_DOMAIN",
    "SOURCE_RESERVE_PER_STYLE_SLOT",
    "SYNTHETIC_DOMAINS",
    "action_variant_for",
    "active_case_cards",
    "active_source_case_cards",
    "assert_canonical_design",
    "assert_final_design",
    "assert_source_design_with_reserves",
    "base_tool_id_for",
    "build_design_cells",
    "candidate_count_for_design",
    "candidate_index_for_record",
    "candidate_key_for_record",
    "case_card_id_for",
    "case_cards_by_id",
    "cell_key_for_record",
    "normalize_case_card_data",
    "style_slots_for_config",
)


@dataclass(frozen=True)
class DesignCell:
    case_card_id: str
    style_slot: str
    card: Mapping[str, Any]
    base_tool_id: str | None = None
    action_variant: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.case_card_id, self.style_slot)

    @property
    def case_id(self) -> str:
        return f"{self.case_card_id}_{self.style_slot}"

    @property
    def prompt_id(self) -> str:
        """Row-level prompt identifier; intentionally identical to ``case_id``."""
        return self.case_id

    @property
    def domain(self) -> str:
        return str(self.card.get("domain") or "")


def normalize_case_card_data(data: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(data, list):
        return {"case_cards": data}
    return dict(data)


def case_card_id_for(card: Mapping[str, Any]) -> str:
    for key in ("case_card_id", "tool_id", "id"):
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("case card must define case_card_id, tool_id, or id")


def base_tool_id_for(card: Mapping[str, Any]) -> str:
    value = card.get("base_tool_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = card.get("tool_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return case_card_id_for(card)


def action_variant_for(card: Mapping[str, Any]) -> str | None:
    value = card.get("action_variant")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def style_slots_for_config(data: Mapping[str, Any] | list[Mapping[str, Any]]) -> tuple[str, ...]:
    normalized = normalize_case_card_data(data)
    slots = normalized.get("style_slots")
    if slots is None:
        return DEFAULT_STYLE_SLOTS
    return _normalize_style_slots(slots, where="style_slots")


def _normalize_style_slots(slots: Iterable[Any], *, where: str) -> tuple[str, ...]:
    if isinstance(slots, str) or isinstance(slots, Mapping):
        raise ValueError(f"{where} must be a sequence of strings")
    out: list[str] = []
    for idx, slot in enumerate(slots):
        if not isinstance(slot, str):
            raise ValueError(f"{where}[{idx}] must be a string")
        normalized = slot.strip()
        if not normalized:
            raise ValueError(f"{where}[{idx}] must not be blank")
        out.append(normalized)
    if len(set(out)) != len(out):
        raise ValueError(f"{where} must be unique")
    return tuple(out)


_VALID_PARTITIONS = frozenset({"main", "pilot", "reserve"})


def active_case_cards(
    cards: Iterable[Mapping[str, Any]],
    *,
    include_partitions: Iterable[str] = ("main",),
) -> list[Mapping[str, Any]]:
    """Return cards that participate in the active prompt design.

    Source bundles include pilot/calibration cards (excluded by default)
    and reserve cards (excluded by default; opt in via include_partitions
    for the generator + A4 cascade source-side path). The default
    ('main',) preserves the legacy behavior of the assigned-style
    confirmatory design. Cards without a partition_id field are treated
    as active for fixture compatibility.
    """
    accepted = frozenset(include_partitions)
    if not accepted.issubset(_VALID_PARTITIONS):
        raise ValueError(
            f"include_partitions {sorted(accepted)} contains entries not in "
            f"{sorted(_VALID_PARTITIONS)}"
        )
    out: list[Mapping[str, Any]] = []
    for card in cards:
        partition = card.get("partition_id")
        if partition is None:
            out.append(card)
            continue
        if not isinstance(partition, str):
            raise ValueError(f"invalid partition_id: {partition!r}")
        normalized = partition.strip()
        if not normalized or normalized != partition:
            raise ValueError(f"invalid partition_id: {partition!r}")
        if normalized not in _VALID_PARTITIONS:
            raise ValueError(f"invalid partition_id: {partition!r}")
        if normalized not in accepted:
            continue
        out.append(card)
    return out


def active_source_case_cards(
    cards: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Source-side case cards: main + reserve partitions.

    Used by the generator and the A4 cascade so they iterate the full
    216-cell source set (192 main + 24 reserve). Final-corpus assertions
    use active_case_cards (main-only) after the selector relabels
    substituted reserves.
    """
    return active_case_cards(cards, include_partitions=("main", "reserve"))


def case_cards_by_id(cards: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for card in cards:
        cid = case_card_id_for(card)
        if cid in out:
            raise ValueError(f"duplicate case_card_id: {cid!r}")
        out[cid] = card
    return out


def _uses_assigned_style_design(data: Mapping[str, Any], cards: list[Mapping[str, Any]]) -> bool:
    design = data.get("corpus_design") or data.get("design")
    if isinstance(design, str) and design in ASSIGNED_STYLE_DESIGNS:
        return True
    return any(
        bool(str(card.get("assigned_style_slot") or "").strip())
        for card in cards
    )


def _validate_style_slot(slot: str, style_slots: tuple[str, ...], where: str) -> None:
    if slot not in style_slots:
        raise ValueError(
            f"{where} uses unknown style_slot {slot!r}; expected one of {style_slots!r}"
        )


def build_design_cells(
    data: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    style_slots: Iterable[str] | None = None,
    include_partitions: Iterable[str] = ("main",),
) -> list[DesignCell]:
    normalized = normalize_case_card_data(data)
    cards = active_case_cards(
        list(normalized.get("case_cards") or []),
        include_partitions=include_partitions,
    )
    slots = (
        _normalize_style_slots(style_slots, where="style_slots")
        if style_slots is not None
        else style_slots_for_config(normalized)
    )
    if not slots:
        raise ValueError("at least one style slot is required")
    cards_index = case_cards_by_id(cards)
    assigned_style_design = _uses_assigned_style_design(normalized, cards)

    cells: list[DesignCell] = []
    explicit_cells = normalized.get("design_cells")
    if explicit_cells is not None:
        for idx, raw_cell in enumerate(explicit_cells):
            if not isinstance(raw_cell, Mapping):
                raise ValueError(f"design_cells[{idx}] must be an object")
            raw_cid = raw_cell.get("case_card_id")
            if raw_cid is None or (isinstance(raw_cid, str) and not raw_cid.strip()):
                raise ValueError(f"design_cells[{idx}] is missing case_card_id")
            if not isinstance(raw_cid, str):
                raise ValueError(f"design_cells[{idx}].case_card_id must be a string")
            cid = raw_cid.strip()
            raw_slot = raw_cell.get("style_slot")
            if raw_slot is None or (isinstance(raw_slot, str) and not raw_slot.strip()):
                raise ValueError(f"design_cells[{idx}] is missing style_slot")
            if not isinstance(raw_slot, str):
                raise ValueError(f"design_cells[{idx}].style_slot must be a string")
            slot = raw_slot.strip()
            if cid not in cards_index:
                raise ValueError(f"design_cells[{idx}] references unknown case_card_id {cid!r}")
            _validate_style_slot(slot, slots, f"design_cells[{idx}]")
            card = cards_index[cid]
            assigned_slot = str(card.get("assigned_style_slot") or "").strip()
            if assigned_slot:
                _validate_style_slot(assigned_slot, slots, f"case card {cid!r}")
                if slot != assigned_slot:
                    raise ValueError(
                        f"design_cells[{idx}] style_slot {slot!r} does not match "
                        f"case card {cid!r} assigned_style_slot {assigned_slot!r}"
                    )
            cells.append(DesignCell(
                case_card_id=cid,
                style_slot=slot,
                card=card,
                base_tool_id=base_tool_id_for(card),
                action_variant=action_variant_for(card),
            ))
    elif assigned_style_design:
        for card in cards:
            cid = case_card_id_for(card)
            slot = str(card.get("assigned_style_slot") or "").strip()
            if not slot:
                raise ValueError(f"case card {cid!r} is missing assigned_style_slot")
            _validate_style_slot(slot, slots, f"case card {cid!r}")
            cells.append(DesignCell(
                case_card_id=cid,
                style_slot=slot,
                card=card,
                base_tool_id=base_tool_id_for(card),
                action_variant=action_variant_for(card),
            ))
    else:
        raise ValueError(
            "active corpus design requires assigned_style_slot on each case "
            "card or an explicit design_cells block; implicit card-by-style "
            "products are disabled in the 192 action-variant design"
        )

    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for cell in cells:
        if cell.key in seen:
            duplicates.append(cell.key)
        seen.add(cell.key)
    if duplicates:
        raise ValueError(f"duplicate design cells: {duplicates[:5]}")
    if explicit_cells is not None:
        covered_card_ids = {cell.case_card_id for cell in cells}
        missing_card_ids = sorted(set(cards_index) - covered_card_ids)
        if missing_card_ids:
            raise ValueError(
                f"design_cells missing {len(missing_card_ids)} active "
                f"case_card_id(s): {missing_card_ids[:5]}"
            )
    if assigned_style_design:
        seen_assigned_card_ids: set[str] = set()
        duplicate_card_ids: list[str] = []
        for cell in cells:
            if cell.case_card_id in seen_assigned_card_ids:
                duplicate_card_ids.append(cell.case_card_id)
            seen_assigned_card_ids.add(cell.case_card_id)
        if duplicate_card_ids:
            raise ValueError(
                "assigned-style design allows one design cell per case_card_id; "
                f"duplicates: {duplicate_card_ids[:5]}"
            )
    return cells


def candidate_count_for_design(
    data: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    candidates_per_cell: int = CANDIDATES_PER_CELL,
    include_partitions: Iterable[str] = ("main",),
) -> int:
    return (
        len(build_design_cells(data, include_partitions=include_partitions))
        * candidates_per_cell
    )


def assert_canonical_design(
    cells: list[DesignCell],
    *,
    expected_count: int = CANONICAL_DESIGN_CELL_COUNT,
    expected_per_slot: int = CANONICAL_CELLS_PER_STYLE_SLOT,
) -> None:
    """Enforce the 192-cell / 48-per-slot preregistered design at production
    entry points.

    The structural invariants the OSF preregistration commits to are:
    exactly 192 design cells AND a 48-per-style-slot balance. The test
    suite enforces this on the canonical fixture, but adding the gate
    inside ``build_design_cells`` would also reject smaller test fixtures.
    Splitting it into a separate gate keeps test fixtures flexible while
    still preventing an unbalanced or wrong-sized cell list from reaching
    live A3 generation or A4 selection.
    """
    if len(cells) != expected_count:
        raise ValueError(
            f"design cell count: expected {expected_count}, got {len(cells)}"
        )
    counts = Counter(cell.style_slot for cell in cells)
    if any(v != expected_per_slot for v in counts.values()):
        raise ValueError(
            f"unbalanced style slots: {dict(counts)} (expected each "
            f"slot to have exactly {expected_per_slot} cells)"
        )


# Source-side design constants for the v2 reserve-buffer corpus
# (192 main + 24 reserve = 216 source cells).
SOURCE_DESIGN_CELL_COUNT = 216
SOURCE_RESERVE_CELL_COUNT = 24
SOURCE_RESERVE_PER_DOMAIN = 4
SOURCE_RESERVE_PER_STYLE_SLOT = 6


def assert_source_design_with_reserves(
    cards: Iterable[Mapping[str, Any]],
) -> None:
    """Source-side invariants: 192 main + 24 reserve cards, 32+4 per
    domain, 48+6 per style, full reserve coverage.

    Reserve coverage rules:
    - Every reserve carries reserve_priority=1, reserve_domain matching
      its domain, reserve_style_slot matching its assigned_style_slot,
      and replacement_eligible_for listing exactly the 8 main primaries
      sharing the (domain, style_slot) stratum.
    - Every main primary appears in exactly one reserve's
      replacement_eligible_for list.
    """
    cards = list(cards)
    main = [c for c in cards if c.get("partition_id") == "main"]
    reserve = [c for c in cards if c.get("partition_id") == "reserve"]
    if len(main) != CANONICAL_DESIGN_CELL_COUNT:
        raise ValueError(
            f"main card count: expected {CANONICAL_DESIGN_CELL_COUNT}, "
            f"got {len(main)}"
        )
    if len(reserve) != SOURCE_RESERVE_CELL_COUNT:
        raise ValueError(
            f"reserve card count: expected {SOURCE_RESERVE_CELL_COUNT}, "
            f"got {len(reserve)}"
        )

    # Per-domain + per-style reserve balance
    by_domain = Counter(c.get("domain") for c in reserve)
    if set(by_domain) != set(SYNTHETIC_DOMAINS):
        raise ValueError(
            f"reserve domains: expected {sorted(SYNTHETIC_DOMAINS)}, "
            f"got {sorted(str(k) for k in by_domain)}"
        )
    if any(v != SOURCE_RESERVE_PER_DOMAIN for v in by_domain.values()):
        raise ValueError(
            f"unbalanced reserve per domain: {dict(by_domain)} (expected "
            f"each domain to have exactly {SOURCE_RESERVE_PER_DOMAIN})"
        )
    by_style = Counter(c.get("assigned_style_slot") for c in reserve)
    if set(by_style) != set(DEFAULT_STYLE_SLOTS):
        raise ValueError(
            f"reserve style slots: expected {sorted(DEFAULT_STYLE_SLOTS)}, "
            f"got {sorted(str(k) for k in by_style)}"
        )
    if any(v != SOURCE_RESERVE_PER_STYLE_SLOT for v in by_style.values()):
        raise ValueError(
            f"unbalanced reserve per style: {dict(by_style)} (expected "
            f"each style to have exactly {SOURCE_RESERVE_PER_STYLE_SLOT})"
        )

    # Reserve metadata invariants
    main_by_id = {case_card_id_for(c): c for c in main}
    main_ids = set(main_by_id)
    eligible_seen: dict[str, list[str]] = {}
    for r in reserve:
        rid = case_card_id_for(r)
        if not r.get("is_reserve"):
            raise ValueError(f"reserve {rid!r} missing is_reserve=true")
        if r.get("reserve_priority") != 1:
            raise ValueError(
                f"reserve {rid!r} reserve_priority must be 1, got "
                f"{r.get('reserve_priority')!r}"
            )
        if r.get("reserve_domain") != r.get("domain"):
            raise ValueError(
                f"reserve {rid!r} reserve_domain != domain "
                f"({r.get('reserve_domain')!r} vs {r.get('domain')!r})"
            )
        if r.get("reserve_style_slot") != r.get("assigned_style_slot"):
            raise ValueError(
                f"reserve {rid!r} reserve_style_slot != assigned_style_slot"
            )
        eligible = r.get("replacement_eligible_for") or []
        if not isinstance(eligible, list) or len(eligible) != 8:
            raise ValueError(
                f"reserve {rid!r} replacement_eligible_for must be a list "
                f"of 8 case_card_ids, got {eligible!r}"
            )
        for primary_id in eligible:
            if primary_id not in main_ids:
                raise ValueError(
                    f"reserve {rid!r} replacement_eligible_for entry "
                    f"{primary_id!r} not in main partition"
                )
            primary = main_by_id[primary_id]
            if (
                primary.get("domain") != r.get("domain")
                or primary.get("assigned_style_slot") != r.get("assigned_style_slot")
            ):
                raise ValueError(
                    f"reserve {rid!r} replacement_eligible_for entry "
                    f"{primary_id!r} has mismatched stratum "
                    f"({primary.get('domain')!r}, "
                    f"{primary.get('assigned_style_slot')!r})"
                )
            eligible_seen.setdefault(primary_id, []).append(rid)
    # Every main primary appears in exactly one reserve list
    for primary_id in main_ids:
        coverers = eligible_seen.get(primary_id, [])
        if len(coverers) != 1:
            raise ValueError(
                f"primary {primary_id!r} covered by {len(coverers)} reserves "
                f"(expected exactly 1): {coverers!r}"
            )


def assert_final_design(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int = CANONICAL_DESIGN_CELL_COUNT,
    expected_per_slot: int = CANONICAL_CELLS_PER_STYLE_SLOT,
) -> None:
    """Final-corpus invariants: 192 main rows, 48/48/48/48 style balance,
    every row has partition_id=='main' (substituted reserves relabeled).
    Substitution provenance fields must be consistent: source_partition_id
    is 'main' or 'reserve', is_substituted is bool, replacement_for is
    null iff is_substituted is False.
    """
    rows = list(rows)
    if len(rows) != expected_count:
        raise ValueError(
            f"final corpus row count: expected {expected_count}, "
            f"got {len(rows)}"
        )
    style_counts: Counter[str] = Counter()
    seen_cells: set[tuple[str, str]] = set()
    for row in rows:
        case_card_id = str(row.get("case_card_id") or "")
        if row.get("partition_id") != "main":
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} has "
                f"partition_id={row.get('partition_id')!r}, expected 'main'"
            )
        spi = row.get("source_partition_id")
        if spi not in {"main", "reserve"}:
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} has invalid "
                f"source_partition_id={spi!r}"
            )
        sub = row.get("is_substituted")
        if not isinstance(sub, bool):
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} is_substituted "
                f"must be bool, got {sub!r}"
            )
        rep = row.get("replacement_for")
        if sub and not isinstance(rep, str):
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} is_substituted "
                f"true requires replacement_for as string, got {rep!r}"
            )
        if (not sub) and rep is not None:
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} is_substituted "
                f"false requires replacement_for=null, got {rep!r}"
            )
        if sub and spi != "reserve":
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} is_substituted "
                f"true requires source_partition_id=='reserve', got {spi!r}"
            )
        style_slot = str(row.get("style_slot") or "")
        if style_slot not in DEFAULT_STYLE_SLOTS:
            raise ValueError(
                f"final corpus row {row.get('case_card_id')!r} has unknown "
                f"style_slot={style_slot!r}"
            )
        cell = (case_card_id, style_slot)
        if cell in seen_cells:
            raise ValueError(f"duplicate final corpus cell: {cell!r}")
        seen_cells.add(cell)
        style_counts[style_slot] += 1
    if set(style_counts) != set(DEFAULT_STYLE_SLOTS):
        raise ValueError(
            f"final style slots: expected {sorted(DEFAULT_STYLE_SLOTS)}, "
            f"got {sorted(style_counts)}"
        )
    if any(v != expected_per_slot for v in style_counts.values()):
        raise ValueError(
            f"unbalanced final style slots: {dict(style_counts)} "
            f"(expected {expected_per_slot} per slot)"
        )


def cell_key_for_record(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record.get("case_card_id") or ""), str(record.get("style_slot") or ""))


def candidate_index_for_record(record: Mapping[str, Any]) -> int:
    candidate_index = record.get("candidate_index")
    if candidate_index is None:
        raise ValueError("candidate record missing candidate_index")
    if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
        raise ValueError("candidate record candidate_index must be an integer")
    if not 0 <= candidate_index < CANDIDATES_PER_CELL:
        raise ValueError(
            f"candidate_index must be in range 0..{CANDIDATES_PER_CELL - 1}"
        )
    return candidate_index


def candidate_key_for_record(record: Mapping[str, Any]) -> tuple[str, str, int]:
    case_card_id = record.get("case_card_id")
    if case_card_id is None or (isinstance(case_card_id, str) and not case_card_id.strip()):
        raise ValueError("candidate record missing case_card_id")
    if not isinstance(case_card_id, str):
        raise ValueError("candidate record case_card_id must be a string")
    style_slot = record.get("style_slot")
    if style_slot is None or (isinstance(style_slot, str) and not style_slot.strip()):
        raise ValueError("candidate record missing style_slot")
    if not isinstance(style_slot, str):
        raise ValueError("candidate record style_slot must be a string")
    candidate_index = candidate_index_for_record(record)
    return (
        case_card_id.strip(),
        style_slot.strip(),
        candidate_index,
    )

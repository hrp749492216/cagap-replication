# Execution-governance amendment: exploratory DISABLED+FORMAT control

- **Dated:** 2026-09-05
- **Owner authorization time:** 2026-09-05T19:22:31Z
- **Classification:** Post-registration exploratory
- **OSF filing required:** No
- **Protocol:** `CAGAP-POSTREG-DISABLED-FORMAT-2026-09-03-v3-GROK-GPT-KIMI`
- **Protocol SHA-256:** `575c336713cad45f2866d254f4686c6dbca3beaf9009f7cff83ac8de6a0d06e7`
- **Locked-input SHA-256:** `5a9e12c02532b12d4866f8b1693196381fcfd3d42be9d481d72ab7ed6779512c`

## Decision and timing

On 2026-09-05, the project owner confirmed that the DISABLED+FORMAT control is
an exploratory study and instructed the project not to file a new OSF record
for it. At that time, the official successor campaign contained zero canary
responses, zero full responses, zero labels, and zero analysis results. The
only completed Kimi batch was the separately identified 24-row diagnostic
pilot, whose rows are excluded from every study endpoint.

This amendment prospectively replaces the still-unperformed OSF activation
step with a dated, hash-bound project-owner execution authorization. It does
not retroactively relabel an executed study and does not represent an OSF
filing.

## Scope of the amendment

Only execution governance changes. The following scientific and operational
commitments remain byte-for-byte or semantically unchanged:

- the three subjects and Kimi → Grok → GPT execution order;
- all three arms and the exact FORMAT instruction;
- the native Kimi and pinned OpenRouter route descriptors;
- all 10,368 full request rows and nine excluded canary rows;
- request identities, randomization, hashes, estimands, missingness rules,
  retry policy, stopping rule, and H1/H2 isolation;
- subject-specific canary review before full dispatch; and
- the existing OpenRouter and Kimi spending ceilings.

The 2026-09-03 scientific addendum, frozen protocol, locked-input manifest,
prepared OSF manifest, and non-executable v1 receipt template remain preserved
as historical artifacts. Their statements that OSF filing is the next
activation gate are superseded for this exploratory campaign only.

## Replacement activation boundary

Official submission requires all of the following before network access:

1. the unchanged frozen protocol and locked-input hashes;
2. a machine-readable governance amendment binding this document and an
   approved activation launcher by SHA-256;
3. an immutable v2 execution receipt binding that amendment, the exact
   three-subject population, currently authorized profiles, and the existing
   spending allocations;
4. `submission_permitted: true`, attributed to the project owner; and
5. the existing explicit command-line acknowledgement, exact request-count
   gate, and route-specific cost gate.

The approved activation launcher may replace only the OSF-specific receipt
validation step. It must delegate generation, locked-input validation, route
validation, request construction, provider transport, and cost enforcement to
the byte-pinned scientific runner. Any stale path, hash, classification,
subject list, spending allocation, or approval time fails closed.

The initial receipt authorizes the `canary` profile only. A full shard remains
blocked until its subject's canary is retrieved, reviewed, and bound by a later
prospective authorization update.

## Remaining execution sequence

No provider request is made by this amendment. The next external action is the
three-row Kimi official canary. It must be retrieved and reviewed for route and
model identity, completeness, visible schemas, valid tool behavior, usable
output, and spending before the Kimi full shard can begin. Grok and GPT follow
under the same canary-first rule, with GPT last.

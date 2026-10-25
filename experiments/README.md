<!-- generated-by: gsd-doc-writer -->
# Experiments

Versioned experimental inputs that are small enough and sufficiently reviewed to belong in the public reproducibility repository.

## Collections

| Directory | Contents |
|---|---|
| [`aligned/`](aligned/README.md) | Locked 192-row action-variant prompt corpus and its selection manifest |
| [`benign_capability_battery/`](benign_capability_battery/README.md) | Thirty benign control tasks plus a checked-in SHA-256 digest |

This directory contains inputs, not response data. Raw model responses, judge batches, intermediate grading files, and analysis outputs belong under ignored `outputs/` paths and become public only through an explicit redacted release process.

## Change discipline

Experimental inputs are tied to configuration, schemas, tests, and source hashes. After changing a corpus file:

1. Run the collection-specific tests.
2. Recompute any checked-in digest or selection manifest through its owning script.
3. Refresh the material-lock and OSF manifest chain if the artifact is part of a release.
4. Record whether the change affects the confirmatory design or only a descriptive extension.


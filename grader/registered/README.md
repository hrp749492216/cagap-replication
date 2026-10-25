# Main-study predicate grader

`semantic_predicates.py` in this directory is the byte-identical predicate grader pinned by
`materials_lock.json` (SHA-256 `2da31253687c7ab14ff3c0c436f97b2803a071d06fcac821ccce10e01745c488`). It produced the
main-study labels and is kept for exact comparison; the pipeline does not import it.

`grader/semantic_predicates.py`, the file the pipeline imports, is the later revision pinned by
every DISABLED+FORMAT control protocol in `experiments/disabled_format_control/` (SHA-256
`d9959f043157d96e36da79ce064dba6a228155ff5f8cb7686c8280c57b977367`). The revision adds the schema tool's risk class to
the predicate trace and labels a call to a non-target high-impact schema tool Partial instead of
Refuse; `diff` the two files to see the exact change. To grade with the main-study version, copy
this file over `grader/semantic_predicates.py`.

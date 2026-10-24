# Redactions and omissions in this submission

* `PREREG_AMENDMENT_4_POMDP_GRPO.md`: one sentence that cross-referenced an unrelated internal
  project was replaced by `[REDACTED ...]`. Nothing about the design, arms, metrics or tests was
  changed. Because of this edit, the shipped file does **not** match the sha256 in
  `PREREG_AMENDMENT_4_POMDP_GRPO.sha256`; that hash refers to the unredacted original, which was
  hashed at 2026-10-23 20:44 CDT before any POMDP/GRPO result existed.
* `PREREG_AMENDMENT_2_REGIME.md`, `PREREG_AMENDMENT_3_SECTOR.md` and `PREREG_AMENDMENT_4_...` list an
  "ILL rule classifier" (Information Lattice Learning, Yu, Evans & Varshney, JAIR 2023) among the
  arms. Those arms, and one further amendment that only concerned them, were run but are not part
  of this submission; every number reported here comes from the code in this folder. The shipped
  versions of `run_regime.py`, `run_sector.py` and `run_rl.py` do not include those arms, and were
  re-run in this folder to confirm that the reported numbers are reproduced.
* Reproducibility note: the Markov-switching baseline in `run_regime.py` now seeds statsmodels' random start search. The original run did not, so its RG3 AUC was 0.4530 instead of 0.4529 here; all accuracies and every other number are identical (checked value by value: 6,588 values).
* `turnaround/`: the pre-registration (`turnaround/PREREG.md`) also lists an "ILL rule
  classifier" arm. That arm was run but is not part of this submission. `turnaround/run_turnaround.py`
  is the executed script with that arm removed; every other arm is an independent deterministic job. The
  data files are shipped gzipped; `data/SHA256SUMS` holds the hashes of the uncompressed files, and the
  script checks them after decompressing.
  Spot check (2026-10-24): with the shipped script and the gzipped data, the data hashes verify and the
  logistic-regression and gradient-boosting arms reproduce all 2,232 stored test scores exactly
  (max |difference| = 0.0).
* `PREREG_AMENDMENT_2_REGIME.md`, `PREREG_AMENDMENT_3_SECTOR.md` and `PREREG_AMENDMENT_4_POMDP_GRPO.md`:
  line 3 was an informal working note quoting the question that prompted each amendment. It was
  replaced by a neutral one-line scope sentence (and, in amendment 3, the heading words
  "(the question asked)" by "(the motivating question)"). Nothing about the design, arms, metrics or
  tests was changed. Because of this edit the shipped files do **not** match their `.sha256` files;
  those hashes refer to the originals, which were hashed before the corresponding results existed.
* `turnaround/PREREG.md`: line 3 received the same edit (an informal note replaced by a neutral
  question sentence), so the shipped file does **not** match `turnaround/PREREG.sha256`; that hash
  refers to the original, which was hashed before any price was downloaded.

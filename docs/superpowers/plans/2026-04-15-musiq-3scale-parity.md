# Plan: Restore MUSIQ 3rd Scale (Original-Res) to Match pyiqa Default

Date: 2026-04-15
Branch: `bench-tuning` (continuing MUSIQ CoreML migration)

## Background

After shipping single-model MUSIQ-AVA via CoreML (previous plan
`2026-04-15-musiq-coreml-conversion.md`), T14 parity gate passes against a
*custom* 2-scale pyiqa config (`max_seq_len_from_original_res=None`) with
Spearman=0.9994 — the CoreML port is faithful.

But the bench regressed: v0.3.0 (pyiqa default 3-scale) scored Spearman 0.769
on AVA-500; v0.4.0 (CoreML 2-scale) scores 0.643. Ship criterion
`Bench Spearman ≥ 0.70` fails by a wide margin. The missing scale is
pyiqa's default *original-resolution* patch block
(`max_seq_len_from_original_res=-1`), which MUSIQ-AVA was trained with.

## Goal

Add the 3rd (original-resolution) scale to preprocessor + CoreML export while
keeping CoreML's static input shape. Target bench Spearman ≥ 0.70 on 500-AVA.

## Design

Google MUSIQ's published config caps the original-res scale at 512 patches
(`max_seq_len_from_original_res=512`). Use that as the fixed 3rd block:

```
seqLen = 49 (scale 224, 7×7) + 144 (scale 384, 12×12) + 512 (orig-res capped)
       = 705
rowDim = 3075                           (unchanged)
scale IDs: 0 = 224, 1 = 384, 2 = orig
```

For original-res:
- `count_h = ceil(h/32)`, `count_w = ceil(w/32)` (TF-SAME padding as-is).
- Emit patches in row-major order (matching `F.unfold`).
- Pad with zero patches (mask=0) if `count_h*count_w < 512`.
- Truncate to first 512 in row-major order if larger.
- Spatial-hash index computed on the true `count_h × count_w` grid, then
  truncated/padded alongside patches.
- Mask = 1 for active patch rows, 0 for padding rows.

The truncation is deliberately first-512-in-row-major to exactly match
pyiqa's `_pad_or_cut_to_max_seq_len`.

## Ship criteria

- Parity (CoreML vs pyiqa, `max_seq_len_from_original_res=512`): Spearman ≥ 0.97,
  max|Δ| ≤ 0.10, mean|Δ| ≤ 0.03 on AVA-100.
- Bench (`run.py eval`): Spearman ≥ 0.70 on 500-AVA with re-optimized thresholds.
- All Swift + Python tests pass.

## Tasks

### T16: 3-scale reference + regenerated fixtures

**Files:**
- `testing/bench/bench/parity.py`
- `testing/bench/tests/fixtures/make_musiq_fixtures.py`
- `ImageRaterTests/Fixtures/musiq_reference/patch_tensor_500x400.*`

**Changes:**
1. In `parity.py::pyiqa_scores`, set
   `metric.net.data_preprocess_opts["max_seq_len_from_original_res"] = 512`.
2. In `make_musiq_fixtures.py`:
   - Add constant `ORIG_RES_MAX_SEQ = 512`.
   - After the 2-scale block, append an original-res block built by calling
     `_extract_patches_and_positions_from_image(img_multi, ..., scale_id=2, max_seq_len=512)`.
   - Resulting full tensor: shape `[1, 49+144+512, 3075] = [1, 705, 3075]`.
3. Regenerate fixtures: `python3 testing/bench/tests/fixtures/make_musiq_fixtures.py`.
   Commit the updated `.f32` / `.shape` files under
   `ImageRaterTests/Fixtures/musiq_reference/`.

**Verification:** `.shape` of `patch_tensor_500x400.f32` reads `1,705,3075`.

---

### T17: Swift MUSIQPreprocessor 3rd scale

**Files:**
- `ImageRater/Pipeline/MUSIQPreprocessor.swift`
- `ImageRaterTests/MUSIQPreprocessorTests.swift`

**Changes:**
1. In `MUSIQPreprocessor`:
   - `seqLen = 705`.
   - Add `static let origResMaxSeqLen = 512`.
   - In `patchTensor(pixels:h:w:channels:)`, after the `for (scaleId, scale) in scales.enumerated()` loop:
     - Build patches from the *normalized* original pixels directly via
       `unfoldPatches(pixels:h:w:channels:patch:patchSize)`.
     - Compute spatial positions via `hashSpatialPositions(countH:countW:gridSize:)` on true grid.
     - `activeN = min(countH*countW, origResMaxSeqLen)`.
     - For `i in 0..<activeN`: copy patch + spatial + scale_id=2 + mask=1.
     - Rows `activeN..<origResMaxSeqLen` remain zero (mask=0), `tPtr` was zero-initialized.
     - `rowOffset += origResMaxSeqLen`.
2. Update `precondition(rowOffset == seqLen)` — already guards correctness.
3. `MUSIQPreprocessorTests`:
   - `test_patchTensor_500x400_matches_pyiqa_reference`: update shape expectation to `[1, 705, 3075]`.
   - Tolerance unchanged (`1e-2`).

**Verification:** `xcodebuild test -scheme ImageRater -only-testing:ImageRaterTests/MUSIQPreprocessorTests` green.

---

### T18: CoreML re-export with seqLen=705

**Files:**
- `scripts/convert_musiq.py`
- `ImageRater/MLModels/musiq-ava.mlpackage` (regenerated)

**Changes:**
1. `SEQ_LEN = 49 + 144 + 512`.
2. Update the dummy scale_id assignment: rows [0..49)=0, [49..193)=1, [193..705)=2.
3. Mask dummy: leave all ones (static graph trace only — actual mask values come from runtime input).
4. Docstring update.
5. Re-run `python3 scripts/convert_musiq.py` → new `.mlpackage`.
6. Verify Focal Xcode scheme build produces `.mlmodelc` with new input shape.

**Verification:**
- `FocalScorerSmokeTests` passes (model loads, produces in-range score for solid 256×256 image).
- `mlpackage` metadata via `python3 -c "import coremltools as ct; m=ct.models.MLModel('ImageRater/MLModels/musiq-ava.mlpackage'); print(m.input_description)"` shows shape (1, 705, 3075).

---

### T19: Parity re-validation + bench re-optimize

**Steps:**
1. Clear bench cache: `rm -rf testing/bench/.cache/scores_*.json testing/bench/.cache/parity_*`.
2. Rebuild FocalScorer Release in Xcode (new mlmodelc).
3. Run parity test: `cd testing/bench && python3 -m pytest tests/test_musiq_parity.py -v`.
   Expect Spearman ≥ 0.97, max|Δ| ≤ 0.10, mean|Δ| ≤ 0.03.
4. `python3 run.py score` (re-scores 500 AVA via new CoreML).
5. `python3 run.py optimize --trials 500` → new `params.candidate.json`.
6. Copy candidate → current: `cp testing/bench/params.candidate.json testing/bench/params.current.json`.
7. `python3 scripts/gen_defaults.py` → regenerates Swift defaults.
8. `python3 run.py eval` → confirm Spearman ≥ 0.70.

**Verification:** The final eval `metrics.json` under `testing/bench/results/v0.4.0@<sha>/` shows `spearman ≥ 0.70`.

---

### T20: Leaderboard + final commit

1. `python3 run.py leaderboard` → regenerates `testing/bench/LEADERBOARD.md`.
2. Inspect: v0.4.0 row present with Spearman ≥ 0.70.
3. Full test run: `xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test` + `cd testing/bench && python3 -m pytest`.
4. Commit: `feat: ship v0.4.0 3-scale MUSIQ CoreML pipeline` (bundles T16–T20 if not already split).

---

## Risk / Rollback

- If parity still diverges: verify truncation order matches pyiqa exactly (row-major, first N).
  Swift must use `for i in 0..<min(countH*countW, 512)` iterating `py*countW + px` order.
- If bench Spearman still < 0.70 after 3-scale: candidate cause is threshold-tuning noise; try
  `--trials 2000` and stratified re-sample. Do NOT further grow seqLen (CoreML weight count
  unchanged but attention cost scales O(seqLen²) — ANE OOM risk above ~1000).
- Rollback: revert all commits for T16–T20 (leaves 2-scale v0.4.0 intact).

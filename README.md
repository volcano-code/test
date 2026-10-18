# P-CRAFT PINO source archive

This repository contains the current local P-CRAFT model, training, evaluation,
physics and pipeline source. Architecture branches select PINO3d, UFNO3d,
UFNODualBranch3d, HRPINO3d, CalibratedPINO and CalibratedPINOD23. The two
calibrated variants are wrappers, not independent backbone architectures.

Each `versions/` branch selects one local checkpoint directory, including named
external comparisons. Its selection record and checkpoint metadata preserve the
local distinction. Shared code is the current source snapshot: it is NOT claimed
to be the historical training source for every checkpoint.

Weights (about 12 GB), datasets, caches, generated meshes and vendor clones are
not in Git. See ARCHIVE_MANIFEST.json for checkpoint locations and sizes and
checkpoint_metadata/ for available configurations and metrics. A source archive
does not establish training reproducibility or model accuracy. Native, Python
and third-party dependencies used by local scripts must be provisioned separately.

All local files were preserved. No training jobs were started for this archive.

## Selected variant

```json
{
  "name": "current_candidate_targeted_safe_pino_20260708_rerun",
  "files": [
    {
      "path": "checkpoints/current_candidate_targeted_safe_pino_20260708_rerun/decision_aware_targeted_safe_pino.pt",
      "bytes": 16814610
    }
  ],
  "weights_uploaded": false
}
```

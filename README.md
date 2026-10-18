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
  "name": "v29c_v27_pinball_s2606",
  "files": [
    {
      "path": "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt",
      "bytes": 26311030
    },
    {
      "path": "checkpoints/v29c_v27_pinball_s2606/history.json",
      "bytes": 68921
    },
    {
      "path": "checkpoints/v29c_v27_pinball_s2606/training_summary.json",
      "bytes": 1066
    }
  ],
  "weights_uploaded": false
}
```

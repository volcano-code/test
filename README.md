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
  "name": "taskdisjoint_five_model_comparator_20260718",
  "files": [
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/cnn3d/best_model.pt",
      "bytes": 1022517
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/cnn3d/history.json",
      "bytes": 20685
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/cnn3d/protocol.json",
      "bytes": 191
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/full_pino/best_model.pt",
      "bytes": 154573255
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/full_pino/epoch_50.pt",
      "bytes": 54383427
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/full_pino/last_model.pt",
      "bytes": 54384811
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/full_pino/training_history.json",
      "bytes": 134068
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/geometry_mlp/best_model.pt",
      "bytes": 43513
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/geometry_mlp/history.json",
      "bytes": 20480
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/geometry_mlp/protocol.json",
      "bytes": 191
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno/best_model.pt",
      "bytes": 134897881
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno/epoch_50.pt",
      "bytes": 44960229
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno/last_model.pt",
      "bytes": 44961453
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno/training_history.json",
      "bytes": 155565
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno_dual_no_physics/best_model.pt",
      "bytes": 154573255
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno_dual_no_physics/epoch_50.pt",
      "bytes": 54383427
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno_dual_no_physics/last_model.pt",
      "bytes": 54384811
    },
    {
      "path": "checkpoints/taskdisjoint_five_model_comparator_20260718/ufno_dual_no_physics/training_history.json",
      "bytes": 187930
    }
  ],
  "weights_uploaded": false
}
```

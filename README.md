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
  "name": "pino_component_ablations_20260720",
  "files": [
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_asymmetric_loss/best_model.pt",
      "bytes": 154573383
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_asymmetric_loss/epoch_50.pt",
      "bytes": 54383427
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_asymmetric_loss/last_model.pt",
      "bytes": 54384811
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_asymmetric_loss/training_history.json",
      "bytes": 119937
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_physics/best_model.pt",
      "bytes": 154573383
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_physics/epoch_50.pt",
      "bytes": 54383427
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_physics/last_model.pt",
      "bytes": 54384811
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_physics/training_history.json",
      "bytes": 131854
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_scale_fno/best_model.pt",
      "bytes": 134898009
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_scale_fno/epoch_50.pt",
      "bytes": 44960229
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_scale_fno/last_model.pt",
      "bytes": 44961453
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_scale_fno/training_history.json",
      "bytes": 135043
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_structural_context/best_model.pt",
      "bytes": 154571079
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_structural_context/epoch_50.pt",
      "bytes": 54382659
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_structural_context/last_model.pt",
      "bytes": 54384043
    },
    {
      "path": "checkpoints/pino_component_ablations_20260720/w_o_structural_context/training_history.json",
      "bytes": 141105
    }
  ],
  "weights_uploaded": false
}
```

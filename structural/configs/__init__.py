"""配置模块"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """加载YAML配置，默认加载default.yaml"""
    if path is None:
        path = Path(__file__).parent / "default.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class MaterialConfig:
    youngs_modulus: float = 3.5e9
    poisson_ratio: float = 0.36
    yield_stress: float = 50.0e6
    density: float = 1.24e3
    cell_size: float = 1.0


@dataclass
class PINOConfig:
    modes: int = 12
    width: int = 32
    n_layers: int = 4
    in_channels: int = 4
    out_channels: int = 9
    lifting_channels: int = 64
    projection_channels: int = 64
    padding: int = 8


@dataclass
class RepairConfig:
    channels: list = field(default_factory=lambda: [16, 32, 32])
    kernel_size: int = 3
    delta_max: float = 0.3
    steps: int = 200
    lr: float = 1e-3
    lambda_structural: float = 10.0
    lambda_volume: float = 5.0
    lambda_shape: float = 20.0
    lambda_smooth: float = 1.0
    lambda_printable: float = 2.0
    volume_budget: float = 1.3
    lambda_redistribution: float = 8.0
    lambda_locality: float = 4.0
    shell_width: int = 2
    high_stress_threshold: float = 0.65
    low_stress_threshold: float = 0.20
    max_net_material_ratio: float = 0.02


@dataclass
class SurrogateConfig:
    calibration_mode: str = "inflation"
    stress_inflation_factor: float = 1.15
    stress_quantile_margin: float = 0.95
    fem_probe_interval: int = 25
    rollback_on_fem_fail: bool = True
    acceptance_metric: str = "best_fem_feasible"
    unsafe_region_weight: float = 2.5
    hard_case_replay_ratio: float = 0.35


@dataclass
class EvaluationConfig:
    use_fem_for_final_metrics: bool = True
    multi_load_aggregation: str = "worst_case"
    displacement_limit: float = 5e-3
    volume_budget_ratio: float = 1.3
    overhang_limit: float = 0.15
    silhouette_metric: str = "mv_iou"


@dataclass
class StructShadowConfig:
    resolution: int = 64
    material: MaterialConfig = field(default_factory=MaterialConfig)
    pino: PINOConfig = field(default_factory=PINOConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None):
        raw = load_config(path)
        # YAML safe_load may parse scientific notation (e.g. 3.5e9) as str
        mat_raw = raw.get('material', {})
        for k in mat_raw:
            if isinstance(mat_raw[k], str):
                try:
                    mat_raw[k] = float(mat_raw[k])
                except ValueError:
                    pass
        mat = MaterialConfig(**mat_raw)
        pino = PINOConfig(**raw.get('pino', {}))

        repair_raw = {**raw.get('repair_cnn', {}), **raw.get('repair', {})}
        repair = RepairConfig(**{k: v for k, v in repair_raw.items()
                                 if k in RepairConfig.__dataclass_fields__})
        surrogate = SurrogateConfig(**{
            k: v for k, v in raw.get('surrogate', {}).items()
            if k in SurrogateConfig.__dataclass_fields__
        })
        evaluation = EvaluationConfig(**{
            k: v for k, v in raw.get('evaluation', {}).items()
            if k in EvaluationConfig.__dataclass_fields__
        })
        return cls(
            resolution=raw.get('resolution', 64),
            material=mat,
            pino=pino,
            repair=repair,
            surrogate=surrogate,
            evaluation=evaluation,
        )

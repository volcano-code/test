#!/usr/bin/env python3
"""Generate 12 NSA-style display models with complex symbolic silhouettes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE = ROOT / "experiments/build_nsa_style_12_10cm_256.py"
spec = importlib.util.spec_from_file_location("nsa_base", str(BASE))
nsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsa)

nsa.OUT = ROOT / "results/nsa_style_complex_12_10cm_256"
nsa.CANDIDATE = "carve_pf5"

COMPLEX = ROOT / "data/silhouettes/complex_v2_filled"
SYMBOL = ROOT / "data/silhouettes/curated_symbolic_v1"


def src(name: str) -> Path:
    if (COMPLEX / f"{name}.png").exists():
        return COMPLEX / f"{name}.png"
    if (SYMBOL / f"{name}.png").exists():
        return SYMBOL / f"{name}.png"
    raise FileNotFoundError(name)


def patch_run_one() -> None:
    original = nsa.run_one

    def run_one(case, example, names):
        old_bench = nsa.BENCH
        tmp = nsa.OUT / "_mixed_inputs"
        tmp.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = tmp / f"{name}.png"
            if not target.exists():
                target.write_bytes(src(name).read_bytes())
        nsa.BENCH = tmp
        try:
            return original(case, example, names)
        finally:
            nsa.BENCH = old_bench

    nsa.run_one = run_one


nsa.CASES = [
    {
        "case_id": "case1_corner_clear_plate",
        "nsa_panel": "top-left",
        "setup": "90-degree two-wall corner, black table, transparent acrylic plate; show left/right wall shadows, optional floor shadow.",
        "caution": "Use model-only STL for shadow validation; any opaque plate/base is display-only and must stay outside the measured shadow windows.",
        "examples": [
            ("case1A_dragon_gear_hanzi", ["complex_dragon", "aaai_gear", "han_ying_shadow"]),
            ("case1B_phoenix_A_light", ["complex_phoenix", "aaai_A", "han_guang_light"]),
        ],
    },
    {
        "case_id": "case2_corner_suspended_object",
        "nsa_panel": "top-middle",
        "setup": "90-degree two-wall corner; suspend/hold the model with transparent line or thin acrylic rod from above; keep fixture outside the wall-shadow regions.",
        "caution": "This is the cleanest physical setup when a model has poor bottom contact; fixture is not part of the algorithmic result.",
        "examples": [
            ("case2A_castle_robot_dream", ["complex_castle", "complex_robot", "han_meng_dream"]),
            ("case2B_ship_tree_dragonchar", ["complex_ship", "aaai_tree", "han_long_dragon"]),
        ],
    },
    {
        "case_id": "case3_single_wall_like",
        "nsa_panel": "top-right",
        "setup": "Approximation: use two vertical white screens in a shallow V or a 90-degree corner and photograph nearly front-on so the two wall shadows read like one wall display.",
        "caution": "The current pipeline is orthogonal three-plane TFS, not arbitrary same-wall multi-light projection; exact NSA same-wall reproduction needs a non-orthogonal projection operator.",
        "examples": [
            ("case3A_city_violin_intelligence", ["complex_scifi_city", "complex_violin", "han_zhi_intelligence"]),
            ("case3B_cathedral_star_mountain", ["complex_cathedral", "aaai_star", "han_shan_mountain"]),
        ],
    },
    {
        "case_id": "case4_white_paper_sweep",
        "nsa_panel": "bottom-left",
        "setup": "White seamless paper sweep from wall to floor; use one horizontal wall light and one overhead/floor light, with the third wall optional.",
        "caution": "A curved sweep is for photography; metric validation still uses the three orthogonal digital projections.",
        "examples": [
            ("case4A_bonsai_heart_shadow", ["complex_bonsai", "aaai_heart", "han_ying_shadow"]),
            ("case4B_stag_I_light", ["complex_stag", "aaai_I", "han_guang_light"]),
        ],
    },
    {
        "case_id": "case5_acrylic_center_display",
        "nsa_panel": "bottom-middle",
        "setup": "Transparent acrylic block/plate at the corner center; three projection planes can be shown, but take photos with the fixture nearly invisible.",
        "caution": "If using translucent resin, report it as a physical display choice; geometry metrics are computed on the opaque binary model.",
        "examples": [
            ("case5A_owltotem_gear_dream", ["complex_owl_totem", "aaai_gear", "han_meng_dream"]),
            ("case5B_bicycle_A_intelligence", ["complex_bicycle", "aaai_A", "han_zhi_intelligence"]),
        ],
    },
    {
        "case_id": "case6_plinth_like_display",
        "nsa_panel": "bottom-right",
        "setup": "Use an independent plinth or transparent pedestal only for presentation; place/crop it so plinth shadow is not evaluated as part of the three silhouettes.",
        "caution": "Do not merge the plinth into the shadow-art STL; an opaque connected base would break at least one target silhouette.",
        "examples": [
            ("case6A_robot_star_mountain", ["complex_robot", "aaai_star", "han_shan_mountain"]),
            ("case6B_ship_tree_dragonchar", ["complex_ship", "aaai_tree", "han_long_dragon"]),
        ],
    },
]


def main() -> None:
    patch_run_one()
    nsa.main()


if __name__ == "__main__":
    main()

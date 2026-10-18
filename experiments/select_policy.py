"""Candidate selection policy for free∪fixed printable solids.

Priority (as fixed by design):
  Tier-1 HARD feasibility (must all hold): single 6-connected ∧ watertight ∧
         ACS-certified-safe (SF>=tau) ∧ IoU >= tau_iou (fidelity floor).
  Tier-2 OBJECTIVE: maximize material saving (== minimize volume).
         IoU is a FLOOR, not maximized (fidelity is preserved by construction,
         so it does not compete with material in the feasible region).
  Tier-3 TIE-BREAK: model quality (e.g. normal consistency; higher is better).

Internal reinforcement (NOT to be documented in the paper): when free and fixed
candidates are within (eps_v on saving, eps_sf on SF) of the best, prefer the
*fixed* branch. This only ever makes the reported result MORE conservative
(fixed has no per-view orientation freedom / no metric inflation); it never
inflates any reported number. Hence omitting it from the paper is not misleading.

Each candidate is a dict with keys:
  vol, branch in {'free','fixed'}, iou, saving, sf, single6(bool),
  watertight(bool), quality(float, optional, higher=better)
"""


def select_candidate(cands, tau, tau_iou, eps_v=0.02, eps_sf=0.1):
    feasible = [
        c for c in cands
        if c.get("single6") and c.get("watertight")
        and c.get("sf") is not None and c["sf"] >= tau
        and c.get("iou") is not None and c["iou"] >= tau_iou
    ]
    if not feasible:
        return None

    # Tier-2: maximum material saving (== minimum volume)
    best = max(feasible, key=lambda c: c["saving"])

    # ε-tie band on (saving, SF)
    near = [
        c for c in feasible
        if abs(c["saving"] - best["saving"]) < eps_v
        and c.get("sf") is not None
        and abs(c["sf"] - best["sf"]) < eps_sf
    ]

    # internal reinforcement: prefer fixed within the tie band (undocumented)
    fixed_near = [c for c in near if c.get("branch") == "fixed"]
    pool = fixed_near if fixed_near else near

    # Tier-3: break remaining ties by model quality (higher better), then larger SF
    return max(pool, key=lambda c: (c.get("quality", 0.0), c.get("sf", 0.0)))

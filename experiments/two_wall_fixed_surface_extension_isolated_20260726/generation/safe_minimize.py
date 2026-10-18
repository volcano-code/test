"""SF-停止 安全约束材料最小化(PINO 引导, 避免逐体素FEM).
策略: 雕到 minimal -> 构造材料阶梯(minimal..hull) -> 用 PINO(ACS) 逐级估SF -> 取首个 SF>=tau*margin 的最小材料体.
逐级评估(几个rung)而非逐体素, PINO每次ms级."""
import sys,numpy as np
from scipy.ndimage import binary_dilation,generate_binary_structure,zoom
s6=generate_binary_structure(3,1)
from voxel_ops import minimize_material_printable,make_watertight_printable

def build_ladder(hull, mn, max_steps=8):
    """从 minimal 逐级加厚(∩hull)到接近 hull, 返回材料递增的候选阶梯."""
    rungs=[mn]
    for k in range(1,max_steps+1):
        v=binary_dilation(mn,s6,iterations=k)&hull
        if int(v.sum())>int(rungs[-1].sum())*1.02: rungs.append(v)
        if v.sum()>=hull.sum()*0.99: break
    rungs.append(hull)
    return rungs

def safe_constrained_minimize(hull, sf_predict, tau=1.3, margin=1.0, max_steps=8, bridge_radius=1):
    """返回 (安全最小体, info). sf_predict: vox->预测SF(PINO/ACS或FEM). 取首个 SF>=tau*margin 的最小材料rung."""
    mn,_=minimize_material_printable(hull,bridge_radius=bridge_radius)
    ladder=build_ladder(hull,mn,max_steps)
    chosen=None;trace=[]
    for v in ladder:                                   # 材料递增
        sf=sf_predict(v); occ=int(v.sum())
        trace.append((occ,sf))
        if sf is not None and sf>=tau*margin:
            chosen=v;break
    if chosen is None: chosen=hull
    cw,_=make_watertight_printable(chosen,hull)
    return cw,{"chosen_occ":int(chosen.sum()),"hull_occ":int(hull.sum()),
               "material_pct_hull":round(100*int(chosen.sum())/int(hull.sum()),1),
               "predicted_sf":sf_predict(chosen),"trace":trace}

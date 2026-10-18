"""优化模块 — StructShadow的三个核心方法论贡献

1. ALMOptimizer: 增广拉格朗日法，将结构安全从软惩罚升级为硬约束
2. SafetyTransferBound: PINO近似误差→结构安全保证的误差传递定理
3. AdaptiveGradientRouter: 约束感知的自适应梯度路由（内外解耦策略）
"""

from .alm_optimizer import ALMOptimizer, ALMConfig, ConstraintState
from .safety_bound import SafetyTransferBound, SafetyVerificationResult
from .gradient_router import AdaptiveGradientRouter, RoutingDecision

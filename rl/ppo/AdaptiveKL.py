import numpy as np

class AdaptiveKL:
    """
    Adaptive KL penalty controller (proportional style).

    This controller adjusts the KL coefficient (kl_coef)
    so that the policy KL stays around a desired target.

    - target_kl: desired KL magnitude.
    - init_kl_coef: initial KL penalty weight.
    - gain: proportional gain for update (smaller = smoother).
    - min_kl_coef / max_kl_coef: safety bounds.
    """

    def __init__(self,
                 target_kl=2,
                 init_kl_coef=0.02,
                 alpha=0.1,
                 min_kl_coef=1e-3,
                 max_kl_coef=1.0):
        self.target_kl = target_kl
        self.kl_coef = init_kl_coef
        self.alpha = alpha
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef

    def update(self, current_kl):
        """
        Update kl_coef based on the current KL.

        - If KL is too high → increase kl_coef (stronger penalty)
        - If KL is too low  → decrease kl_coef (weaker penalty)

        Returns the updated kl_coef.
        """

        # Proportional error: how far current_kl is from target_kl (relative)
        # error = current_kl / target_kl - 1
        # clip the error to avoid too aggressive updates
        proportional_error = np.clip(
            current_kl / (self.target_kl + 1e-8) - 1.0,
            -0.2,  # at most -20%
            0.2    # at most +20%
        )
        # multiplicative factor for kl_coef
        mult = 1.0 + self.alpha * proportional_error
        # update kl_coef
        self.kl_coef *= mult
        # clamp into safe range
        self.kl_coef = float(
            max(self.min_kl_coef, min(self.max_kl_coef, self.kl_coef))
        )

        return self.kl_coef


# class AdaptiveKL:
#     """
#     Adaptive KL penalty controller.
#     This controller adjusts the KL coefficient (kl_coef)
#     so that the policy KL stays around a desired target.

#     - target_kl: desired KL magnitude.
#     - init_kl_coef: initial KL penalty weight.
#     - alpha: multiplicative adjustment factor (>1).
#     - min_kl_coef / max_kl_coef: safety bounds.
#     """

#     def __init__(
#         self,
#         target_kl=0.2,
#         init_kl_coef=0.02,
#         alpha=1.5,
#         min_kl_coef=1e-3,
#         max_kl_coef=1e-1,
#     ):
#         self.target_kl = target_kl
#         self.kl_coef = init_kl_coef
#         self.alpha = alpha
#         self.min_kl_coef = min_kl_coef
#         self.max_kl_coef = max_kl_coef

#     def update(self, current_kl):
#         """
#         Update kl_coef based on the current KL.

#         - If KL is too high → increase kl_coef (stronger penalty)
#         - If KL is too low  → decrease kl_coef (weaker penalty)

#         Returns the updated kl_coef.
#         """

#         # KL too large → increase penalty
#         if current_kl > 1.5 * self.target_kl:
#             self.kl_coef *= self.alpha

#         # KL too small → decrease penalty
#         elif current_kl < 0.5 * self.target_kl:
#             self.kl_coef /= self.alpha

#         # lamp into safe range
#         self.kl_coef = float(
#             max(self.min_kl_coef, min(self.max_kl_coef, self.kl_coef))
#         )

#         return self.kl_coef


class AdaptiveKLController:
    """自适应KL系数控制器"""
    
    def __init__(self, init_kl_coef=0.02, target_kl=6.0, horizon=10000):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
    
    def update(self, current_kl):
        """根据当前KL值动态调整系数"""
        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        mult = 1 + proportional_error * 0.1
        self.kl_coef *= mult
        
        # 确保在合理范围内
        self.kl_coef = np.clip(self.kl_coef, 0.001, 1.0)
        
        return self.kl_coef

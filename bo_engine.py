"""
bo_engine.py — Bayesian Optimization Engine for LIG Sensor Fabrication
=======================================================================
Inputs : laser power  (5 – 100 W)
         scanning speed (1 – 50 000 mm/min)
Target : resistance (Ω) — MINIMIZE

Internal design
---------------
• Speed is converted to log₁₀ scale before anything else.
  This ensures equal coverage across the huge 1–50 000 range.
• Both inputs normalised to [0, 1] before GP fitting.
• Resistance is log-transformed then standardised (mean 0, std 1).
  This stabilises the GP and handles the multiplicative noise
  that a basic multimeter introduces.
• Kernel : Matern 5/2 with ARD — separate length scales for power and
  log-speed let the model learn which axis drives resistance more strongly.
• Transfer learning : implemented via sklearn GP's per-sample α (alpha).
  More dissimilar material → higher α → that data has less influence on
  the posterior mean and variance.
• Batch BO : Kriging Believer.  After picking x*, the GP is temporarily
  updated with (x*, predicted_mean) before picking x*₂, and so on.
  self.gp is NEVER modified — only ephemeral copies are used.
"""

import warnings
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

warnings.filterwarnings("ignore")


class LIGOptimizer:
    # ── Fixed parameter bounds ────────────────────────────────────────────────
    POWER_MIN     = 5.0
    POWER_MAX     = 100.0
    SPEED_MIN     = 1.0
    SPEED_MAX     = 50_000.0
    LOG_SPEED_MIN = np.log10(1.0)         # 0.0
    LOG_SPEED_MAX = np.log10(50_000.0)    # ≈ 4.699

    def __init__(self, base_noise: float = 0.05):
        """
        Parameters
        ----------
        base_noise : float
            Noise std for current-material data in *normalised y* space.
            0.05 ≈ 5 % measurement error — suitable for a basic multimeter.
            Raise to 0.10–0.15 if readings are visibly unstable.
        """
        self.base_noise  = base_noise
        self.gp          = None
        self.X_train     = None   # normalised X  — current material only
        self.y_train     = None   # normalised y  — current material only
        self.y_best_norm = np.inf
        self._y_log_mean = 0.0
        self._y_log_std  = 1.0

    # =========================================================================
    # Normalisation helpers
    # =========================================================================

    def _norm_X(self, power, speed) -> np.ndarray:
        """(power, speed) arrays → normalised [0, 1]² matrix."""
        p = np.atleast_1d(np.asarray(power,  dtype=float))
        s = np.atleast_1d(np.asarray(speed,  dtype=float))
        p_n = (p - self.POWER_MIN) / (self.POWER_MAX - self.POWER_MIN)
        s_n = (np.log10(s) - self.LOG_SPEED_MIN) / \
              (self.LOG_SPEED_MAX - self.LOG_SPEED_MIN)
        return np.column_stack([p_n, s_n])

    def _denorm_X(self, X_n: np.ndarray):
        """Normalised [0, 1]² → (power array, speed array)."""
        X_n   = np.atleast_2d(X_n)
        power = X_n[:, 0] * (self.POWER_MAX - self.POWER_MIN) + self.POWER_MIN
        log_s = X_n[:, 1] * (self.LOG_SPEED_MAX - self.LOG_SPEED_MIN) \
                + self.LOG_SPEED_MIN
        speed = 10.0 ** log_s
        return power, speed

    def _norm_y(self, resistance) -> np.ndarray:
        """log(resistance) → standardised using current material's statistics."""
        r     = np.atleast_1d(np.asarray(resistance, dtype=float))
        log_r = np.log(r + 1e-8)
        return (log_r - self._y_log_mean) / (self._y_log_std + 1e-8)

    def _denorm_y(self, y_n) -> np.ndarray:
        """Inverse of _norm_y → resistance values."""
        y_n   = np.atleast_1d(np.asarray(y_n, dtype=float))
        log_r = y_n * self._y_log_std + self._y_log_mean
        return np.exp(log_r)

    # =========================================================================
    # Latin Hypercube Sampling
    # =========================================================================

    def lhs_sample(self, n: int = 15, seed: int = None) -> list:
        """
        Generate *n* initial parameter combinations with LHS.

        Speed is sampled uniformly on the log₁₀ axis then converted back,
        so combinations are spread across decades (e.g. 10, 100, 1 000,
        10 000 mm/min) rather than clustering near the upper end.

        Returns
        -------
        list of (power: float, speed: int) tuples
        """
        rng     = np.random.default_rng(seed)
        sampler = LatinHypercube(d=2, seed=rng)
        raw     = sampler.random(n=n)

        powers = self.POWER_MIN + raw[:, 0] * (self.POWER_MAX - self.POWER_MIN)
        log_s  = self.LOG_SPEED_MIN + raw[:, 1] * \
                 (self.LOG_SPEED_MAX - self.LOG_SPEED_MIN)
        speeds = 10.0 ** log_s

        powers = np.round(powers, 1)
        speeds = np.round(speeds, 0).astype(int)
        return list(zip(powers.tolist(), speeds.tolist()))

    # =========================================================================
    # Surrogate model fitting (with transfer learning)
    # =========================================================================

    def fit(self, experiments: list, transfer_data: list = None) -> bool:
        """
        Fit the Gaussian Process model.

        Parameters
        ----------
        experiments : list of (power, speed, resistance)
            Data from the *current* material (trusted fully).

        transfer_data : list of (power, speed, resistance, similarity_weight)
            Data from *previous* materials.
            similarity_weight ∈ [0, 1] — computed from carbon content:
              w = exp(−|Δcarbon| / 20)
            Transfer learning implementation
            ─────────────────────────────────
            sklearn GP accepts a vector `alpha` (one value per training point).
            alpha[i] is the *variance* of Gaussian noise on observation i.
            We set:
              current material   → alpha = base_noise²      (tiny, fully trusted)
              similar material   → alpha = (base_noise + (1−w)·2)²
              dissimilar (w<0.05)→ skipped entirely
            This way dissimilar data barely shifts the posterior, while
            near-identical material data contributes almost as much as
            current measurements.

        Returns
        -------
        bool — True on success, False if < 2 experiments.
        """
        if len(experiments) < 2:
            return False

        # ── Current material ──────────────────────────────────────────────────
        curr_X = self._norm_X(
            [e[0] for e in experiments],
            [e[1] for e in experiments]
        )
        curr_r = np.array([e[2] for e in experiments], dtype=float)

        # Normalization constants come from current data only
        log_r            = np.log(curr_r + 1e-8)
        self._y_log_mean = float(np.mean(log_r))
        self._y_log_std  = float(max(np.std(log_r), 0.05))

        curr_y     = self._norm_y(curr_r)
        curr_alpha = np.full(len(curr_y), self.base_noise ** 2)

        # ── Transfer data ─────────────────────────────────────────────────────
        t_X, t_y, t_alpha = [], [], []
        if transfer_data:
            for p, s, r, w in transfer_data:
                w = float(w)
                if w < 0.05:
                    continue  # Too dissimilar — skip
                noise = self.base_noise + (1.0 - w) * 2.0
                t_X.append(self._norm_X([p], [s])[0])
                t_y.append(float(self._norm_y([float(r)])[0]))
                t_alpha.append(noise ** 2)

        if t_X:
            all_X     = np.vstack([curr_X, np.array(t_X)])
            all_y     = np.concatenate([curr_y, np.array(t_y)])
            all_alpha = np.concatenate([curr_alpha, np.array(t_alpha)])
        else:
            all_X, all_y, all_alpha = curr_X, curr_y, curr_alpha

        # ── Build and fit GP ──────────────────────────────────────────────────
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=[1.0, 1.0],
                length_scale_bounds=[(0.05, 10.0), (0.05, 10.0)],
                nu=2.5      # Matern 5/2: smooth but not infinitely so
            )
        )
        self.gp = GaussianProcessRegressor(
            kernel               = kernel,
            alpha                = all_alpha,
            n_restarts_optimizer = 10,
            normalize_y          = False,   # We normalise manually
            random_state         = 42
        )
        self.gp.fit(all_X, all_y)

        self.X_train     = curr_X.copy()
        self.y_train     = curr_y.copy()
        self.y_best_norm = float(np.min(curr_y))
        return True

    # =========================================================================
    # Expected Improvement acquisition function
    # =========================================================================

    @staticmethod
    def _EI(X: np.ndarray, gp, y_best: float, xi: float = 0.05) -> np.ndarray:
        """
        Expected Improvement.
        xi = 0.05 → moderate exploration pressure.
        Higher xi = explore more (useful early on).
        Lower  xi = exploit more (useful near convergence).
        """
        X       = np.atleast_2d(X)
        mu, sig = gp.predict(X, return_std=True)
        imp     = y_best - mu - xi
        Z       = imp / (sig + 1e-9)
        ei      = imp * norm.cdf(Z) + sig * norm.pdf(Z)
        ei[sig < 1e-10] = 0.0
        return ei

    def _maximize_EI(self, gp, y_best: float, n_restarts: int = 15) -> np.ndarray:
        """
        x* = argmax EI(x) over [0, 1]².
        Multi-start L-BFGS-B + dense random grid fallback.
        """
        bounds  = [(0.0, 1.0), (0.0, 1.0)]
        best_ei = -np.inf
        best_x  = None

        for _ in range(n_restarts):
            x0 = np.random.uniform(0.0, 1.0, 2)
            try:
                res = minimize(
                    fun     = lambda x: -float(
                        self._EI(x.reshape(1, -1), gp, y_best).item()),
                    x0      = x0,
                    method  = "L-BFGS-B",
                    bounds  = bounds,
                    options = {"maxiter": 200, "ftol": 1e-10}
                )
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    best_x  = res.x.copy()
            except Exception:
                continue

        # Dense grid ensures we don't miss a good region
        grid    = np.random.uniform(0, 1, (1000, 2))
        ei_grid = self._EI(grid, gp, y_best)
        idx     = int(np.argmax(ei_grid))
        if ei_grid[idx] > best_ei:
            best_x = grid[idx]

        return best_x

    # =========================================================================
    # Batch BO — Kriging Believer
    # =========================================================================

    def suggest_batch(self, n: int = 15) -> list:
        """
        Generate *n* suggestions in one shot using the Kriging Believer.

        For each of the n suggestions:
          1.  x* = argmax EI under the current working GP
          2.  ŷ  = GP_mean(x*)    ← "believe" the prediction
          3.  Add (x*, ŷ) as a virtual observation (noise ≈ 0)
          4.  Refit GP with the *same* kernel hyperparameters (no re-optimisation)
          5.  Repeat

        self.gp is NEVER modified.

        Returns
        -------
        list of (power: float, speed: int) tuples
        """
        if self.gp is None:
            raise RuntimeError("Call fit() before suggest_batch().")

        # Working copies — original GP left intact
        working_gp = self.gp
        virt_X     = self.X_train.copy()
        virt_y     = self.y_train.copy()
        y_best     = self.y_best_norm
        sugg_norm  = []

        for i in range(n):
            next_x = self._maximize_EI(working_gp, y_best)
            if next_x is None:
                next_x = np.random.uniform(0.0, 1.0, 2)

            sugg_norm.append(next_x.copy())

            if i < n - 1:
                y_pred = float(working_gp.predict(next_x.reshape(1, -1))[0])
                y_best = min(y_best, y_pred)

                new_X     = np.vstack([virt_X, next_x.reshape(1, -1)])
                new_y     = np.append(virt_y, y_pred)
                new_alpha = np.full(len(new_y), self.base_noise ** 2)
                new_alpha[-1] = 1e-8  # Near-zero noise → GP passes through virtual pt

                try:
                    tmp = GaussianProcessRegressor(
                        kernel               = working_gp.kernel_,   # fitted kernel
                        alpha                = new_alpha,
                        n_restarts_optimizer = 0,   # keep hyperparams fixed
                        normalize_y          = False
                    )
                    tmp.fit(new_X, new_y)
                    working_gp = tmp
                    virt_X, virt_y = new_X, new_y
                except Exception:
                    pass  # Keep previous GP on numerical failure

        # ── Convert to physical parameters ────────────────────────────────────
        arr    = np.array(sugg_norm)
        pw, sp = self._denorm_X(arr)
        pw = np.clip(np.round(pw, 1), self.POWER_MIN, self.POWER_MAX)
        sp = np.clip(np.round(sp, 0).astype(int), int(self.SPEED_MIN), int(self.SPEED_MAX))
        return list(zip(pw.tolist(), sp.tolist()))

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_predicted_optimal(self):
        """
        Minimise GP mean over [0, 1]² to find the predicted global optimum.
        This may lie outside the observed data.

        Returns (power, speed, predicted_resistance)  or  None.
        """
        if self.gp is None:
            return None

        best_val = np.inf
        best_x   = None
        bounds   = [(0.0, 1.0), (0.0, 1.0)]

        for _ in range(60):
            x0 = np.random.uniform(0.0, 1.0, 2)
            try:
                res = minimize(
                    fun    = lambda x: float(self.gp.predict(x.reshape(1, -1))[0]),
                    x0     = x0,
                    method = "L-BFGS-B",
                    bounds = bounds
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x   = res.x.copy()
            except Exception:
                continue

        if best_x is None:
            return None

        pw, sp = self._denorm_X(best_x.reshape(1, -1))
        r      = self._denorm_y(np.array([best_val]))[0]
        return (round(float(pw[0]), 1), int(round(float(sp[0]))), round(float(r), 4))

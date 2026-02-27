import numpy as np
from scipy.optimize import minimize_scalar

# --------------------------------------------------------------------
# Weibull helpers (S(t)=exp(-(t/lam)^k), f(t) = (k/lam) (t/lam)^(k-1) exp(-(t/lam)^k))
# --------------------------------------------------------------------

# numerical stability for log
EPS = 1e-12

# Weibull rv
def weibull_rv(k, lam, size, rng):
    return lam * rng.weibull(a=k, size=size)

# log S(t) for Weilbull(k, lam)
def weibull_logS(t, k, lam):
    t = np.clip(t, EPS, None)
    return - (t / lam) ** k

# log f(t) for Weibull(k, lam)
def weibull_logf(t, k, lam):
    t = np.clip(t, EPS, None)
    return np.log(k) - np.log(lam) + (k - 1.0) * (np.log(t) - np.log(lam)) - (t / lam) ** k

# Right censoring (Z, delta)
def simulate_right_censoring_weibull(n, k_T, lam_T, k_C, lam_C, rng):
    T = weibull_rv(k_T, lam_T, n, rng)
    C = weibull_rv(k_C, lam_C, n, rng)
    Z = np.minimum(T, C)
    delta = (T <= C).astype(int)
    return Z, delta


# --------------------------------------------------------------------
# Scores (log-score based)
# --------------------------------------------------------------------

def hazard_score_censored_loglikelihood_weibull(k, lam, Z, delta):
    return delta * weibull_logf(Z, k, lam) + (1 - delta) * weibull_logS(Z, k, lam)

def IPCW_log_score_oracle_weibull(k, lam, Z, delta, k_C, lam_C, min_surv=1e-6, weight_cap=None):
    G = np.exp(weibull_logS(Z, k_C, lam_C))

    # Error when weights are undefined
    if min_surv == 0:
        bad = (delta != 0) & (G <= 0)
        if np.any(bad):
            idx = np.where(bad)[0][:10]
            raise ValueError(f"km: IPCW weights undefined: G(Z) == 0 for {bad.sum()} observation(s) with delta==1 at {idx.tolist()}")

    G = np.maximum(G, min_surv)
    w = delta / G
    if weight_cap is not None:
        w = np.minimum(w, weight_cap)

    return w * weibull_logf(Z, k, lam), w

# Kaplan–Meier for censoring survival G(t)=P(C>t): G_hat(t) = \Pi_{i : t_i <= t} (1 - d_i/r_i), d_i = # events at t_i, r_i = # survivals up to t_i (i.e. at risk)
def km_fit_survival(times, events, clip_zeros_to_next_lowest=True):
    order = np.argsort(times)
    t = times[order]
    e = events[order].astype(int)

    uniq, idx_start, counts = np.unique(t, return_index=True, return_counts=True)
    n = len(t)
    d = np.zeros_like(uniq, dtype=float)
    r = np.zeros_like(uniq, dtype=float)

    for j, start in enumerate(idx_start):
        end = start + counts[j]
        d[j] = e[start:end].sum()
        r[j] = n - start

    step = 1.0 - d / np.maximum(r, 1.0)
    surv = np.cumprod(step)

    if clip_zeros_to_next_lowest:
        pos = surv[surv > 0]
        min_pos = np.min(pos) if pos.size else None

    # evaluate estimated survival
    def G_eval(x):
        x = np.asarray(x)
        pos = np.searchsorted(uniq, x, side="right") - 1    # find pos s.t. uniq[pos-1] <= x < uniq[pos] (for every x[i]), subtract 1 to get uniq[pos] <= x
        out = np.ones_like(x, dtype=float)
        mask = pos >= 0
        out[mask] = surv[pos[mask]]

        if clip_zeros_to_next_lowest & np.any(out == 0):
            if min_pos is None:
                raise ValueError("KM survival is zero everywhere")
            out = np.where(out == 0, min_pos, out)
        return out
    
    return G_eval

def IPCW_weights_plugin_km(Z, delta, G_eval, min_surv=1e-6, weight_cap=None):
    # Error when weights are undefined
    if min_surv == 0:
        bad = (delta != 0) & (G_eval(Z) <= 0)
        if np.any(bad):
            idx = np.where(bad)[0]
            raise ValueError(f"km: IPCW weights undefined: G(Z) == 0 for {bad.sum()} observation(s) with delta==1 at {idx.tolist()}")

    G = np.maximum(G_eval(Z), min_surv)
    w = delta / G
    if weight_cap is not None:
        w = np.minimum(w, weight_cap)
    return w


# --------------------------------------------------------------------
# Maximization/fitting helpers
# --------------------------------------------------------------------

# one dimensional maximization
def maximize_1d_scalar(f, bounds, method, options):
    res = minimize_scalar(lambda x: -f(x), method=method, bounds=bounds, options=options)
    x_star = float(res.x)
    return x_star, float(f(x_star))

# lam_hat(k) = ( (sum_i Z_i^k) / (sum_i delta_i) )^(1/k)
def fit_weibull_hazard_profile(Z, delta, k_bounds=(0.5, 15.0), method="bounded", options=None):
    D = float(delta.sum())
    if D <= 0:
        raise ValueError("D<=0")
    
    if options is None:
        options = {
            "maxiter" : 5000,
            "disp" : int(1),
            "xatol" : 1e-12,
        }

    def lam_hat(k):
        Zk = np.sum(Z ** k)
        return (Zk / D) ** (1.0 / k)

    def profile_mean(k):
        lam = lam_hat(k)
        return hazard_score_censored_loglikelihood_weibull(k, lam, Z, delta).mean()

    k_star, max_mean = maximize_1d_scalar(profile_mean, k_bounds, method, options)
    lam_star = lam_hat(k_star)
    return k_star, lam_star, max_mean

# lam_hat(k) = ( (sum_i [ w_i*Z_i^k ]) / (sum_i w_i) )^(1/k)
def fit_weibull_IPCW_profile(Z, delta, weights, k_bounds=(0.5, 15.0), method="bounded", options=None):
    w = np.asarray(weights, dtype=float)
    W = float(w.sum())
    if W <= 0:
        raise ValueError("W<=0")
    
    if options is None:
        options = {
            "maxiter" : 5000,
            "disp" : int(1),
            "xatol" : 1e-12,
        }

    def lam_hat(k):
        Zwk = np.sum(w * (Z ** k))
        return (Zwk / W) ** (1.0 / k)

    def profile_mean(k):
        lam = lam_hat(k)
        return (w * weibull_logf(Z, k, lam)).mean()

    k_star, max_mean = maximize_1d_scalar(profile_mean, k_bounds, method, options)
    lam_star = lam_hat(k_star)
    return k_star, lam_star, max_mean


# --------------------------------------------------------------------
# Compute parameters
# --------------------------------------------------------------------

def lamC_same_shape_for_cen_prob(k, lam_T, p_cen):
    if not (0.0 < p_cen < 1.0):
        raise ValueError("p_cen must be in (0,1)")
    return lam_T * ((1.0 - p_cen) / (p_cen)) ** (1.0 / k)

def make_distr_params(lam_T=2.0, k_T_values=(0.75, 1.0, 2.5), p_cen_values=(0.2, 0.4, 0.6)):
    params = []
    for k_T in k_T_values:
        for p_cen in p_cen_values:
            k_C = k_T
            lam_C = lamC_same_shape_for_cen_prob(k=k_T, lam_T=lam_T, p_cen=p_cen)
            params.append(
                {
                    "k_T": k_T,
                    "lam_T": lam_T,
                    "k_C": k_C,
                    "lam_C": lam_C,
                    "p_cen_target": p_cen,
                }
            )
    return params


# --------------------------------------------------------------------
# Compute bias and RMSE
# --------------------------------------------------------------------

def bias_sqrtmse(x, target):
    x = np.asarray(x, dtype=float)

    bias = float(np.mean(x - target))
    sqrtmse = float(np.sqrt(np.mean((x - target) ** 2)))
    return bias, sqrtmse
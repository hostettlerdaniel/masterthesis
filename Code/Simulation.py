import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from tqdm.auto import tqdm
import helperfunctions as helpers

# --------------------------------------------------------------------
# Compute divergence and parameters
# --------------------------------------------------------------------

def testset_divergence(Z, delta, k_T, lam_T, k_C, lam_C, IPCW_mode="oracle", min_surv=1e-6, weight_cap=None, k_bounds=(0.5, 15.0), rng=None, split=0.5, return_oracle_gap = True):
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(Z)
    idx = rng.permutation(n)

    # split data
    ntr = int(np.floor(split * n))
    tr, te = idx[:ntr], idx[ntr:]
    Ztr, dtr = Z[tr], delta[tr]
    Zte, dte = Z[te], delta[te]

    # Hazard: fit on training data, divergence on test data
    kh, lamh, _ = helpers.fit_weibull_hazard_profile(Ztr, dtr, k_bounds=k_bounds)
    truth_te = helpers.hazard_score_censored_loglikelihood_weibull(k_T, lam_T, Zte, dte).mean()
    fit_te = helpers.hazard_score_censored_loglikelihood_weibull(kh, lamh, Zte, dte).mean()
    div_h = truth_te - fit_te

    # IPCW: weights computed from training data, divergence on test data
    if IPCW_mode == "oracle":
        _, wtr = helpers.IPCW_log_score_oracle_weibull(k_T, lam_T, Ztr, dtr, k_C, lam_C, min_surv=min_surv, weight_cap=weight_cap)
        ki, lami, _ = helpers.fit_weibull_IPCW_profile(Ztr, dtr, wtr, k_bounds=k_bounds)
        _, wte = helpers.IPCW_log_score_oracle_weibull(k_T, lam_T, Zte, dte, k_C, lam_C, min_surv=min_surv, weight_cap=weight_cap)
        truth_te = (wte * helpers.weibull_logf(Zte, k_T, lam_T)).mean()
        fit_te = (wte * helpers.weibull_logf(Zte, ki, lami)).mean()
        div_i = truth_te - fit_te
        score_gap_true = np.nan
    elif IPCW_mode == "plugin_km":
        Ghat = helpers.km_fit_survival(Ztr, events=(1 - dtr))
        wtr = helpers.IPCW_weights_plugin_km(Ztr, dtr, Ghat, min_surv=min_surv, weight_cap=weight_cap)
        ki, lami, _ = helpers.fit_weibull_IPCW_profile(Ztr, dtr, wtr, k_bounds=k_bounds)
        wte = helpers.IPCW_weights_plugin_km(Zte, dte, Ghat, min_surv=min_surv, weight_cap=weight_cap)
        truth_te = (wte * helpers.weibull_logf(Zte, k_T, lam_T)).mean()
        fit_te = (wte * helpers.weibull_logf(Zte, ki, lami)).mean()
        div_i = truth_te - fit_te
        # plug-in vs oracle at true model on test set
        if return_oracle_gap:
            _, wte_oracle = helpers.IPCW_log_score_oracle_weibull(k_T, lam_T, Zte, dte, k_C, lam_C, min_surv=min_surv, weight_cap=weight_cap)
            score_true_oracle = (wte_oracle * helpers.weibull_logf(Zte, k_T, lam_T)).mean()
            score_true_plugin = truth_te
            score_gap_true = score_true_plugin - score_true_oracle
    else:
        raise ValueError("IPCW_mode must be 'oracle' or 'plugin_km'")

    return div_h, div_i, (kh, lamh), (ki, lami), score_gap_true


# --------------------------------------------------------------------
# CI
# --------------------------------------------------------------------

def bootstrap_mean_ci_upperci(x, B=5000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x)
    R = len(x)
    m = x.mean()

    means = np.empty(B)
    for b in range(B):
        xb = rng.choice(x, size=R, replace=True)
        means[b] = xb.mean()

    # quantile at 1-alpha
    U = np.quantile(means, 1 - alpha)

    # quantiles at alpha/2 and 1-alpha/2
    ci = (np.quantile(means, alpha/2), np.quantile(means, 1 - alpha/2))

    return m, ci, U


# --------------------------------------------------------------------
# Run simulation function
# --------------------------------------------------------------------

def run_simulation(params, seed=42, ns=(50, 100, 200, 500, 1000, 5000, 20000), R=500, true_bounds=(0.5, 15.0), IPCW_mode="oracle", min_surv=1e-6, weight_cap=None, split=0.5, k_bounds=(0.5, 15.0)):
    rng = np.random.default_rng(seed)
    low, high = true_bounds

    # Uniformly distributed params
    if params is None:
        params=[]
        k_T, lam_T, k_C, lam_C = rng.uniform(low, high, size=4)
        params.append(
            {
                "k_T": k_T,
                "lam_T": lam_T,
                "k_C": k_C,
                "lam_C": lam_C,
                "p_cen_target": None,
            }
        )

    results_all = []

    tot_iter = len(params) * len(ns) * R
    progressbar_tot = tqdm(total=tot_iter, desc=f"Total run ({IPCW_mode})", leave=True)
    
    for param in params:
        results_param = []

        k_T, lam_T, k_C, lam_C = param["k_T"], param["lam_T"], param["k_C"], param["lam_C"]
        #print("[True parameters]")
        #print(f"  Event T ~ Weibull(k={k_T:.4f}, lam={lam_T:.4f})")
        #print(f"  Censoring C ~ Weibull(k={k_C:.4f}, lam={lam_C:.4f})")
        #if param["p_cen_target"] is not None:
        #    tmp_p_cen = param["p_cen_target"]
        #    print(f"  Censoring probability P(C<T): {tmp_p_cen:.2f}")
        #print(f"  IPCW mode: {IPCW_mode}\n")

        for n in ns:
            div_h = np.empty(R)
            div_i = np.empty(R)
            kh_hat   = np.empty(R, dtype=float)
            lamh_hat = np.empty(R, dtype=float)
            ki_hat   = np.empty(R, dtype=float)
            lami_hat = np.empty(R, dtype=float)
            gap_true = np.empty(R, dtype=float)

            # Simulate data and compute divergence
            desc = f""
            if param["p_cen_target"] is not None:
                desc += f"pcen{param['p_cen_target']:.2f}_"
            desc += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
            descname = f"n{int(n)}_{desc}"
            progressbar = tqdm(total=R, desc=descname, leave=False)
            for r in range(R):
                Z, delta = helpers.simulate_right_censoring_weibull(n, k_T, lam_T, k_C, lam_C, rng)
                dh, di, (kh, lamh), (ki, lami), gap = testset_divergence(Z, delta, k_T, lam_T, k_C, lam_C, IPCW_mode=IPCW_mode, min_surv=min_surv, weight_cap=weight_cap, k_bounds=k_bounds, rng=rng, split=split)
                div_h[r] = dh
                div_i[r] = di
                kh_hat[r] = kh
                lamh_hat[r] = lamh
                ki_hat[r] = ki
                lami_hat[r] = lami
                gap_true[r] = gap
                progressbar.update(1)
                progressbar_tot.update(1)
            progressbar.close()

            mhb, cihb, dech = bootstrap_mean_ci_upperci(x=div_h, rng=rng)
            mib, ciib, deci = bootstrap_mean_ci_upperci(x=div_i, rng=rng)

            # Reject propriety if upper one sided CI < 0
            reject_prop_h = (dech < 0)
            reject_prop_i = (deci < 0)

            bias_k_h, sqrtmse_k_h = helpers.bias_sqrtmse(kh_hat, k_T)
            bias_k_i, sqrtmse_k_i = helpers.bias_sqrtmse(ki_hat, k_T)

            bias_lam_h, sqrtmse_lam_h = helpers.bias_sqrtmse(lamh_hat, lam_T)
            bias_lam_i, sqrtmse_lam_i = helpers.bias_sqrtmse(lami_hat, lam_T)

            gap_m, gap_ci, _ = bootstrap_mean_ci_upperci(gap_true, rng=rng)

            row = {
                "params": param,
                "n": int(n),

                "div_hazard_mean": mhb,
                "div_hazard_ci": cihb,
                "div_hazard_reject": bool(reject_prop_h),


                "div_IPCW_mean": mib,
                "div_IPCW_ci": ciib,
                "div_IPCW_reject": bool(reject_prop_i),

                "bias_k_hazard": bias_k_h,
                "sqrtmse_k_hazard": sqrtmse_k_h,
                "bias_lam_hazard": bias_lam_h,
                "sqrtmse_lam_hazard": sqrtmse_lam_h,

                "bias_k_IPCW": bias_k_i,
                "sqrtmse_k_IPCW": sqrtmse_k_i,
                "bias_lam_IPCW": bias_lam_i,
                "sqrtmse_lam_IPCW": sqrtmse_lam_i,

                "score_gap_true_mean": gap_m,
                "score_gap_true_ci": gap_ci,
            }
            
            results_param.append(row)
            results_all.append(row)

            # Print results
            #print(f"n={n:>7d} | Hazard div mean {mhb: .3e} CI [{cihb[0]:.3e},{cihb[1]:.3e}] | reject propriety: {reject_prop_h} (1-alpha quantile: {dech:.3e})")
            #print(f"          | IPCW   div mean {mib: .3e} CI [{ciib[0]:.3e},{ciib[1]:.3e}] | reject propriety: {reject_prop_i} (1-alpha quantile: {deci:.3e})")
            #print(f"          | SQRTMSE(k): hazard={sqrtmse_k_h:.3e}, IPCW={sqrtmse_k_i:.3e} | SQRTMSE(lam): hazard={sqrtmse_lam_h:.3e}, IPCW={sqrtmse_lam_i:.3e}")

            #if IPCW_mode == "plugin_km":
            #    print(f"          | gap_true (plugin - oracle) mean {gap_m: .3e} CI [{gap_ci[0]:.3e},{gap_ci[1]:.3e}]\n")
            #else:
            #    print()

            # --------------------------------------------------------------------
            # Plot hist
            if n == np.max(ns):
                qlo_h, qhi_h = np.quantile(div_h, [0.025, 0.975])
                qlo_i, qhi_i = np.quantile(div_i, [0.025, 0.975])
                bins = 40
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

                # Hazard
                axes[0].hist(div_h, bins=bins, edgecolor="white")
                axes[0].axvline(0.0, linestyle="--", linewidth=1)
                axes[0].axvline(np.mean(div_h), linestyle="-", linewidth=1, label=f"mean = {np.mean(div_h):.3e}")
                axes[0].axvline(cihb[0], color="gray", linestyle="--", linewidth=1, label=f"95% CI(mean) [{cihb[0]:.2e},{cihb[1]:.2e}]")
                axes[0].axvline(cihb[1], color="gray", linestyle="--", linewidth=1)
                axes[0].axvline(qlo_h, color="gray", linestyle=":", linewidth=1, label="2.5%, 97.5% quantiles")
                axes[0].axvline(qhi_h, color="gray", linestyle=":", linewidth=1)
                axes[0].set_title(f"Hazard (n = {int(n)})")
                axes[0].set_xlabel("divergence")
                axes[0].set_ylabel("count")
                axes[0].legend()

                # IPCW
                axes[1].hist(div_i, bins=bins, edgecolor="white")
                axes[1].axvline(0.0, linestyle="--", linewidth=1)
                axes[1].axvline(np.mean(div_i), linestyle="-", linewidth=1, label=f"mean = {np.mean(div_i):.3e}")
                axes[1].axvline(ciib[0], color="gray", linestyle="--", linewidth=1, label=f"95% CI(mean) [{ciib[0]:.2e},{ciib[1]:.2e}]")
                axes[1].axvline(ciib[1], color="gray", linestyle="--", linewidth=1)
                axes[1].axvline(qlo_i, color="gray", linestyle=":", linewidth=1, label="2.5%, 97.5% quantiles")
                axes[1].axvline(qhi_i, color="gray", linestyle=":", linewidth=1)
                axes[1].set_title(f"IPCW-{IPCW_mode} (n = {int(n)})")
                axes[1].set_xlabel("divergence")
                axes[1].legend()

                fig.suptitle("Divergence distributions at largest sample size")
                fig.tight_layout()
                
                # Save file
                out_dir = Path("plots_pdf")
                out_dir.mkdir(parents=True, exist_ok=True)
                fcen = f""
                if param["p_cen_target"] is not None:
                    fcen += f"pcen{param['p_cen_target']:.2f}_"
                fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
                fname = f"hist_div_{IPCW_mode}_n{int(n)}_{fcen}.pdf"
                fig.savefig(out_dir / fname)
                out_dir = Path("plots_png")
                out_dir.mkdir(parents=True, exist_ok=True)
                fcen = f""
                if param["p_cen_target"] is not None:
                    fcen += f"pcen{param['p_cen_target']:.2f}_"
                fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
                fname = f"hist_div_{IPCW_mode}_n{int(n)}_{fcen}.png"
                fig.savefig(out_dir / fname)
                
                plt.close(fig)
                # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # Plot mean div with CI
        results_sorted = sorted(results_param, key=lambda d: d["n"])

        ns_arr = np.array([d["n"] for d in results_sorted], dtype=float)

        mhb = np.array([d["div_hazard_mean"] for d in results_sorted], dtype=float)
        lh  = np.array([d["div_hazard_ci"][0] for d in results_sorted], dtype=float)
        uh  = np.array([d["div_hazard_ci"][1] for d in results_sorted], dtype=float)

        mib = np.array([d["div_IPCW_mean"] for d in results_sorted], dtype=float)
        li  = np.array([d["div_IPCW_ci"][0] for d in results_sorted], dtype=float)
        ui  = np.array([d["div_IPCW_ci"][1] for d in results_sorted], dtype=float)


        fig = plt.figure(figsize=(9, 5))
        plt.xscale("log")
        #plt.yscale("log")
        plt.plot(ns_arr, mhb, marker="o", label="Hazard")
        plt.fill_between(ns_arr, lh, uh, alpha=0.2)
        plt.plot(ns_arr, mib, marker="o", label=f"IPCW-{IPCW_mode}")
        plt.fill_between(ns_arr, li, ui, alpha=0.2)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("n (log scale)")
        plt.ylabel("divergence mean( S_true(test) - S_fit(test))")
        plt.title("divergence vs sample size (mean and 95% bootstrap CI)")
        plt.legend()
        plt.tight_layout()

        # Save file
        out_dir = Path("plots_pdf")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"div_{IPCW_mode}_n{int(n)}_{fcen}.pdf"
        fig.savefig(out_dir / fname)
        out_dir = Path("plots_png")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"div_{IPCW_mode}_n{int(n)}_{fcen}.png"
        fig.savefig(out_dir / fname)
        plt.close(fig)
        # --------------------------------------------------------------------

        # Convergence params
        sqrtmse_k_h_arr  = np.array([d["sqrtmse_k_hazard"] for d in results_sorted], dtype=float)
        sqrtmse_k_i_arr  = np.array([d["sqrtmse_k_IPCW"] for d in results_sorted], dtype=float)
        sqrtmse_lam_h_arr = np.array([d["sqrtmse_lam_hazard"] for d in results_sorted], dtype=float)
        sqrtmse_lam_i_arr = np.array([d["sqrtmse_lam_IPCW"] for d in results_sorted], dtype=float)

        # --------------------------------------------------------------------
        # Plot convergences 
        
        # Convergence k
        fig = plt.figure(figsize=(9, 5))
        plt.xscale("log")
        plt.plot(ns_arr, sqrtmse_k_h_arr, marker="o", label="SQRTMSE(k) hazard")
        plt.plot(ns_arr, sqrtmse_k_i_arr, marker="o", label=f"SQRTMSE(k) IPCW-{IPCW_mode}")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("n (log scale)")
        plt.ylabel("SQRTMSE")
        plt.title("Consistency: SQRTMSE of k")
        plt.legend()
        plt.tight_layout()

        # Save file
        out_dir = Path("plots_pdf")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"sqrtmse_k_{IPCW_mode}_n{int(n)}_{fcen}.pdf"
        fig.savefig(out_dir / fname)
        out_dir = Path("plots_png")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"sqrtmse_k_{IPCW_mode}_n{int(n)}_{fcen}.png"
        fig.savefig(out_dir / fname)
        plt.close(fig)

        # Convergence lam
        fig = plt.figure(figsize=(9, 5))
        plt.xscale("log")
        plt.plot(ns_arr, sqrtmse_lam_h_arr, marker="o", label="SQRTMSE(lambda) hazard")
        plt.plot(ns_arr, sqrtmse_lam_i_arr, marker="o", label=f"SQRTMSE(lambda) IPCW-{IPCW_mode}")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("n (log scale)")
        plt.ylabel("SQRTMSE")
        plt.title("Consistency: SQRTMSE of lambda")
        plt.legend()
        plt.tight_layout()

        # Save file
        out_dir = Path("plots_pdf")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"sqrtmse_lam_{IPCW_mode}_n{int(n)}_{fcen}.pdf"
        fig.savefig(out_dir / fname)
        out_dir = Path("plots_png")
        out_dir.mkdir(parents=True, exist_ok=True)
        fcen = f""
        if param["p_cen_target"] is not None:
            fcen += f"pcen{param['p_cen_target']:.2f}_"
        fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
        fname = f"sqrtmse_lam_{IPCW_mode}_n{int(n)}_{fcen}.png"
        fig.savefig(out_dir / fname)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # Plot plug-in vs oracle score gap curve
        if IPCW_mode == "plugin_km":
            gapm = np.array([d["score_gap_true_mean"] for d in results_sorted], dtype=float)
            gl   = np.array([d["score_gap_true_ci"][0] for d in results_sorted], dtype=float)
            gu   = np.array([d["score_gap_true_ci"][1] for d in results_sorted], dtype=float)

            fig = plt.figure(figsize=(9, 5))
            plt.xscale("log")
            plt.plot(ns_arr, gapm, marker="o", label="gap: S_plug-in - S_oracle (true params)")
            plt.fill_between(ns_arr, gl, gu, alpha=0.2)
            plt.axhline(0.0, linestyle="--", linewidth=1)
            plt.xlabel("n (log scale)")
            plt.ylabel("score gap")
            plt.title("Nuisance estimation effect: plug-in vs oracle IPCW score (true model)")
            plt.legend()
            plt.tight_layout()

            # Save file
            out_dir = Path("plots_pdf")
            out_dir.mkdir(parents=True, exist_ok=True)
            fcen = f""
            if param["p_cen_target"] is not None:
                fcen += f"pcen{param['p_cen_target']:.2f}_"
            fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
            fname = f"gap_true_plugin_oracle_{IPCW_mode}_n{int(n)}_{fcen}.pdf"
            fig.savefig(out_dir / fname)
            out_dir = Path("plots_png")
            out_dir.mkdir(parents=True, exist_ok=True)
            fcen = f""
            if param["p_cen_target"] is not None:
                fcen += f"pcen{param['p_cen_target']:.2f}_"
            fcen += f"kT{k_T:.3f}_lamT{lam_T:.3f}_kC{k_C:.3f}_lamC{lam_C:.3f}"
            fname = f"gap_true_plugin_oracle_{IPCW_mode}_n{int(n)}_{fcen}.png"
            fig.savefig(out_dir / fname)
            plt.close(fig)
        # --------------------------------------------------------------------
    
    progressbar_tot.close()
    
    out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_results_csv(results_all, out_dir, IPCW_mode)

    return results_all

def flatten_result_row(row):
    p = row["params"]
    out = {
        "k_T": p.get("k_T"),
        "lam_T": p.get("lam_T"),
        "k_C": p.get("k_C"),
        "lam_C": p.get("lam_C"),
        "p_cen_target": p.get("p_cen_target"),
        "n": row.get("n"),
        "div_hazard_mean": row.get("div_hazard_mean"),
        "div_hazard_ci_low": (row.get("div_hazard_ci") or (np.nan, np.nan))[0],
        "div_hazard_ci_high": (row.get("div_hazard_ci") or (np.nan, np.nan))[1],
        "div_hazard_reject": int(bool(row.get("div_hazard_reject"))),
        "div_IPCW_mean": row.get("div_IPCW_mean"),
        "div_IPCW_ci_low": (row.get("div_IPCW_ci") or (np.nan, np.nan))[0],
        "div_IPCW_ci_high": (row.get("div_IPCW_ci") or (np.nan, np.nan))[1],
        "div_IPCW_reject": int(bool(row.get("div_IPCW_reject"))),
        "sqrtmse_k_hazard": row.get("sqrtmse_k_hazard"),
        "sqrtmse_k_IPCW": row.get("sqrtmse_k_IPCW"),
        "sqrtmse_lam_hazard": row.get("sqrtmse_lam_hazard"),
        "sqrtmse_lam_IPCW": row.get("sqrtmse_lam_IPCW"),
        "score_gap_true_mean": row.get("score_gap_true_mean"),
        "score_gap_true_ci_low": (row.get("score_gap_true_ci") or (np.nan, np.nan))[0],
        "score_gap_true_ci_high": (row.get("score_gap_true_ci") or (np.nan, np.nan))[1],
    }
    return out

def export_results_csv(results_all, out_dir, IPCW_mode):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = [flatten_result_row(r) for r in results_all]

    csv_path = out_dir / f"results_{IPCW_mode}.csv"
    fieldnames = list(flat[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat:
            r_fmt = {
                k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()
            }
            w.writerow(r_fmt)
    return


# --------------------------------------------------------------------
# Run simulation
# --------------------------------------------------------------------

# Oracle IPCW: should look proper, div >= 0
#run_simulation(params=None, seed=42, ns=(50, 100, 200, 500, 1000, 5000, 20000), R=500, IPCW_mode="oracle", min_surv=0.0)

# Plug-in KM IPCW: tests asymptotic propriety of plug-in IPCW
#run_simulation(params=None, seed=42, ns=(50, 100, 200, 500, 1000, 5000, 20000), R=500, IPCW_mode="plugin_km", min_surv=0.0)

params = helpers.make_distr_params(lam_T=2.0, k_T_values=(0.75, 1.0, 2.5), p_cen_values=(0.2, 0.4, 0.6))

# Oracle IPCW: should look proper, div >= 0
run_simulation(params=params, seed=42, ns=(50, 100, 200, 500, 1000, 5000, 20000), R=500, IPCW_mode="oracle", min_surv=0.0)

# Plug-in KM IPCW: tests asymptotic propriety of plug-in IPCW
run_simulation(params=params, seed=42, ns=(50, 100, 200, 500, 1000, 5000, 20000), R=500, IPCW_mode="plugin_km", min_surv=0.0)

print("\n ---------------------------------------Simulation complete--------------------------------------- \n")
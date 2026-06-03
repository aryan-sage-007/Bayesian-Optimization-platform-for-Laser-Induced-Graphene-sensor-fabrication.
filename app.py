"""
app.py — LIG Sensor Optimizer  (Streamlit frontend)
====================================================
Run with:
    streamlit run app.py

Requirements:
    pip install streamlit scikit-learn scipy numpy pandas plotly
"""

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bo_engine import LIGOptimizer
import data_manager as dm

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title = "LIG Optimizer",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS — tighten up the default Streamlit padding slightly
st.markdown("""
<style>
div[data-testid="metric-container"] { padding: 4px 0; }
div[data-testid="stForm"] { border: 1px solid rgba(128,128,128,0.15);
                             border-radius: 8px; padding: 12px 16px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session state initialisation
# =============================================================================
for key, default in [("material_id", None), ("optimizer", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# Helper functions
# =============================================================================

def _fit_optimizer(material_id: str, mat_data: dict) -> LIGOptimizer:
    """Fit a fresh LIGOptimizer on all saved experiments + transfer data."""
    opt  = LIGOptimizer(base_noise=0.05)
    exps = dm.get_all_experiments(mat_data)
    if len(exps) >= 2:
        transfer = dm.get_transfer_data(material_id, mat_data["carbon_content"])
        opt.fit(
            [(e["power"], e["speed"], e["resistance"]) for e in exps],
            transfer or None
        )
    return opt


def _set_material(mid: str):
    """Switch active material and refit the optimizer from saved data."""
    st.session_state.material_id = mid
    d = dm.load_material(mid)
    if d:
        st.session_state.optimizer = _fit_optimizer(mid, d)


def _stable_seed(material_id: str) -> int:
    """Deterministic seed from material ID — same LHS every time app restarts."""
    return int(hashlib.md5(material_id.encode()).hexdigest(), 16) % 100_000


def _convergence_pct(mat_data: dict) -> float | None:
    """Improvement (%) from the second-to-last batch to the last batch."""
    mins, run_min = [], np.inf
    for b in mat_data.get("batches", []):
        for pt in b["points"]:
            if pt["resistance"] is not None:
                run_min = min(run_min, pt["resistance"])
        if run_min < np.inf:
            mins.append(run_min)
    if len(mins) >= 2 and mins[-2] > 0:
        return (mins[-2] - mins[-1]) / mins[-2] * 100.0
    return None


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## ⚡ LIG Optimizer")

    materials = dm.list_materials()

    # ── Switch material ───────────────────────────────────────────────────────
    if materials:
        st.markdown("---")
        label_map = {
            m["material_id"]: f"{m['name']}  (C: {m['carbon_content']}%)"
            for m in materials
        }
        all_opts = [None] + [m["material_id"] for m in materials]
        cur_idx  = all_opts.index(st.session_state.material_id) \
                   if st.session_state.material_id in all_opts else 0

        selected = st.selectbox(
            "Switch material",
            options      = all_opts,
            format_func  = lambda x: "— select —" if x is None else label_map.get(x, x),
            index        = cur_idx
        )
        if selected and selected != st.session_state.material_id:
            _set_material(selected)
            st.rerun()

    # ── New material ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ New material", expanded=(len(materials) == 0)):
        nm_name   = st.text_input("Material name", placeholder="e.g. Polyimide-A",
                                  key="nm_name")
        nm_carbon = st.number_input("Carbon content (%)", 0.0, 100.0, 70.0, 0.5,
                                    key="nm_carbon")
        if st.button("Create & Start", type="primary", use_container_width=True,
                     key="nm_create"):
            if nm_name.strip():
                mid = dm.create_material(nm_name.strip(), float(nm_carbon))
                _set_material(mid)
                st.rerun()
            else:
                st.error("Enter a material name.")

    # ── Active material quick stats ───────────────────────────────────────────
    if st.session_state.material_id:
        d = dm.load_material(st.session_state.material_id)
        if d:
            exps = dm.get_all_experiments(d)
            st.markdown("---")
            st.markdown(f"**{d['name']}**")
            st.caption(f"Carbon: {d['carbon_content']}%")
            ca, cb = st.columns(2)
            ca.metric("Done", len(exps))
            if exps:
                best = min(exps, key=lambda e: e["resistance"])
                cb.metric("Best R", f"{best['resistance']:.3f} Ω")

# =============================================================================
# Landing page (no material selected)
# =============================================================================
if st.session_state.material_id is None:
    st.title("⚡ LIG Sensor Optimizer")
    st.markdown("""
    **Bayesian Optimization for Laser Induced Graphene sensor fabrication.**
    Finds the optimal *laser power* and *scanning speed* to **minimize resistance**,
    using as few physical experiments as possible.

    ### Workflow
    | Step | What happens |
    |------|-------------|
    | 1 | Create a material in the sidebar (name + carbon %) |
    | 2 | **Batch 1 — LHS** : 15 combinations covering the full space intelligently |
    | 3 | Test each combo on your machine, enter resistance values |
    | 4 | **BO Batch** : GP model suggests the next 15 most promising points |
    | 5 | Repeat until convergence, then read off optimal parameters |

    **Transfer learning** automatically re-uses data from similar materials
    (matched by carbon content), so each new material converges faster.
    """)

    if materials:
        st.markdown("---")
        st.subheader("Saved materials")
        df_m = pd.DataFrame(materials)[
            ["name","carbon_content","n_complete","best_R","n_batches","created_at"]
        ]
        df_m.columns = ["Material","Carbon %","Experiments","Best R (Ω)",
                        "Batches","Created"]
        st.dataframe(df_m, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Create your first material in the sidebar to get started.")

# =============================================================================
# Active material view
# =============================================================================
else:
    mat_data = dm.load_material(st.session_state.material_id)
    if mat_data is None:
        st.error("Material data not found — it may have been deleted.")
        st.stop()

    all_exps            = dm.get_all_experiments(mat_data)
    cur_batch, miss_idx = dm.get_current_batch(mat_data)
    n_batches           = len(mat_data.get("batches", []))

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(f"## {mat_data['name']}")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Carbon content",  f"{mat_data['carbon_content']}%")
    h2.metric("Experiments done", len(all_exps))
    h3.metric("Batches completed", n_batches)
    if all_exps:
        best_obs = min(all_exps, key=lambda e: e["resistance"])
        h4.metric("Best R (Ω)", f"{best_obs['resistance']:.4f}")
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_opt, tab_plots, tab_results = st.tabs(["🔬 Optimize", "📊 Plots", "🏆 Results"])

    # =========================================================================
    # TAB 1 — OPTIMIZE
    # =========================================================================
    with tab_opt:

        # ── A: No batches yet ─────────────────────────────────────────────────
        if not mat_data.get("batches"):
            st.subheader("Step 1 — Generate initial exploration batch")
            st.markdown("""
            No experiments yet.  The first batch uses **Latin Hypercube Sampling (LHS)**
            to place 15 parameter combinations that give maximum coverage of the
            Power × Speed space.  Speed is sampled on a **log scale** so you get
            combinations at low speeds (e.g. 50 mm/min) as well as high ones
            (e.g. 40 000 mm/min) — not just clusters near the top of the range.
            """)
            if st.button("🎲  Generate 15 LHS Points", type="primary"):
                opt    = LIGOptimizer()
                pts    = opt.lhs_sample(n=15,
                                        seed=_stable_seed(st.session_state.material_id))
                dm.add_batch(st.session_state.material_id, "lhs", pts)
                st.rerun()

        # ── B: Current batch has un-entered values ────────────────────────────
        elif cur_batch is not None:
            bnum   = cur_batch["batch_number"]
            btype  = cur_batch["type"].upper()
            n_tot  = len(cur_batch["points"])
            n_done = n_tot - len(miss_idx)
            left   = len(miss_idx)

            st.subheader(f"Batch {bnum}  ({btype})  —  {n_done} / {n_tot} values entered")

            if left > 0:
                st.info(
                    f"Run the **{left} remaining** combination"
                    f"{'s' if left > 1 else ''} on your machine, "
                    f"then fill in the resistance values below and click **Save**.\n\n"
                    f"You can save partial results and come back later — "
                    f"already-saved rows will show ✅."
                )

            with st.form(key=f"form_{bnum}"):
                # Header
                hdr = st.columns([0.4, 1.2, 1.8, 2.1, 0.5])
                for col, lbl in zip(hdr, ["**#**", "**Power (W)**",
                                          "**Speed (mm/min)**",
                                          "**Resistance (Ω)**", "**✓**"]):
                    col.markdown(lbl)
                st.divider()

                inputs: dict = {}
                for i, pt in enumerate(cur_batch["points"]):
                    row = st.columns([0.4, 1.2, 1.8, 2.1, 0.5])
                    row[0].write(str(i + 1))
                    row[1].write(f"{pt['power']:.1f}")
                    row[2].write(f"{int(pt['speed'])}")
                    if pt["resistance"] is not None:
                        row[3].write(f"{pt['resistance']:.4f}")
                        row[4].write("✅")
                    else:
                        inputs[i] = row[3].number_input(
                            label            = f"r{bnum}_{i}",
                            min_value        = 0.001,
                            max_value        = 1_000_000.0,
                            value            = None,
                            step             = 0.1,
                            format           = "%.4f",
                            label_visibility = "collapsed",
                            key              = f"inp_{bnum}_{i}"
                        )
                        row[4].write("⏳")

                saved_btn = st.form_submit_button(
                    "💾  Save resistance values",
                    type             = "primary",
                    use_container_width = True
                )

            if saved_btn:
                saved = 0
                for idx, val in inputs.items():
                    if val is not None and val > 0:
                        dm.update_resistance(
                            st.session_state.material_id, bnum, idx, val
                        )
                        saved += 1
                if saved:
                    st.success(f"✅ Saved {saved} value{'s' if saved > 1 else ''}.")
                    st.rerun()
                else:
                    st.warning("Enter at least one resistance value before saving.")

        # ── C: All batches complete → offer next BO batch ─────────────────────
        else:
            st.subheader(
                f"✅ Batch {n_batches} complete — {len(all_exps)} experiments total"
            )

            # Convergence signal
            pct = _convergence_pct(mat_data)
            if pct is not None:
                if pct < 1.0 and n_batches >= 3:
                    st.warning(
                        f"⚠️ Last batch improved resistance by only **{pct:.2f}%**. "
                        f"The optimizer is likely converging — consider stopping here."
                    )
                else:
                    st.success(
                        f"📈 Last batch improved resistance by **{pct:.1f}%**. "
                        f"Keep running batches for a better result."
                    )

            # Current best
            if all_exps:
                best = min(all_exps, key=lambda e: e["resistance"])
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Best power observed",  f"{best['power']:.1f} W")
                bc2.metric("Best speed observed",  f"{int(best['speed'])} mm/min")
                bc3.metric("Min R observed",       f"{best['resistance']:.4f} Ω")

            st.markdown("---")
            go_col, stop_col = st.columns(2)

            with go_col:
                if st.button(
                    "🤖  Generate Next BO Batch  (15 points)",
                    type="primary", use_container_width=True
                ):
                    ok          = False
                    err_msg     = ""
                    opt         = None
                    suggestions = []

                    with st.spinner(
                        "Fitting GP surrogate model "
                        "(Matern 5/2 · ARD · transfer learning)…"
                    ):
                        try:
                            opt      = LIGOptimizer(base_noise=0.05)
                            exp_list = [
                                (e["power"], e["speed"], e["resistance"])
                                for e in all_exps
                            ]
                            transfer = dm.get_transfer_data(
                                st.session_state.material_id,
                                mat_data["carbon_content"]
                            )
                            if transfer:
                                n_transfer = len(transfer)
                                unique_mats = len({round(w, 2) for _, _, _, w in transfer})
                                st.toast(
                                    f"Using {n_transfer} transfer points "
                                    f"from {unique_mats} similar material(s).",
                                    icon="🔁"
                                )
                            ok = opt.fit(exp_list, transfer or None)
                            if not ok:
                                err_msg = "Need at least 2 experiments to fit the model."
                        except Exception as exc:
                            err_msg = f"Model fitting error: {exc}"

                    if ok:
                        with st.spinner(
                            "Running Kriging Believer — generating 15 suggestions…"
                        ):
                            try:
                                suggestions = opt.suggest_batch(n=15)
                            except Exception as exc:
                                ok      = False
                                err_msg = f"Batch generation error: {exc}"

                    if ok and suggestions:
                        dm.add_batch(
                            st.session_state.material_id, "bo", suggestions
                        )
                        st.session_state.optimizer = opt
                        st.rerun()
                    elif err_msg:
                        st.error(err_msg)

            with stop_col:
                if st.button("🏁  Done — view results", use_container_width=True):
                    st.info(
                        "Switch to the **Results** tab above to see optimal "
                        "parameters and the convergence curve."
                    )

    # =========================================================================
    # TAB 2 — PLOTS
    # =========================================================================
    with tab_plots:
        if not all_exps:
            st.info("No experiments yet — complete at least one batch first.")
        else:
            df = pd.DataFrame(all_exps)

            p_tab, s_tab = st.tabs(["R vs Power", "R vs Scanning Speed"])

            # ── R vs Power ────────────────────────────────────────────────────
            with p_tab:
                df_p   = df.sort_values("power").reset_index(drop=True)
                min_ip = int(df_p["resistance"].idxmin())

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x          = df_p["power"],
                    y          = df_p["resistance"],
                    mode       = "lines+markers",
                    marker     = dict(size=8, color="#378ADD",
                                      line=dict(width=1, color="#1a4e8a")),
                    line       = dict(color="#378ADD", width=1.5),
                    customdata = np.stack(
                        [df_p["speed"], df_p["batch"], df_p["batch_type"]], axis=1
                    ),
                    hovertemplate=(
                        "<b>Power %{x:.1f} W</b><br>"
                        "Speed: %{customdata[0]:.0f} mm/min<br>"
                        "R: %{y:.4f} Ω<br>"
                        "Batch %{customdata[1]} (%{customdata[2]})"
                        "<extra></extra>"
                    ),
                    name="Measurements"
                ))
                fig.add_trace(go.Scatter(
                    x    = [df_p.loc[min_ip, "power"]],
                    y    = [df_p.loc[min_ip, "resistance"]],
                    mode = "markers",
                    marker = dict(size=16, color="#E24B4A", symbol="star",
                                  line=dict(width=1.5, color="#8a1c1c")),
                    name = "Minimum R",
                    hovertemplate=(
                        "⭐ <b>Minimum</b><br>"
                        f"Power: {df_p.loc[min_ip, 'power']:.1f} W<br>"
                        f"R: {df_p.loc[min_ip, 'resistance']:.4f} Ω"
                        "<extra></extra>"
                    )
                ))
                fig.update_layout(
                    xaxis_title   = "Power (W)",
                    yaxis_title   = "Resistance (Ω)",
                    height        = 450,
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    xaxis  = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                                  zeroline=False),
                    yaxis  = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                                  zeroline=False),
                    legend = dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── R vs Scanning Speed ───────────────────────────────────────────
            with s_tab:
                df_s   = df.sort_values("speed").reset_index(drop=True)
                min_is = int(df_s["resistance"].idxmin())

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x          = df_s["speed"],
                    y          = df_s["resistance"],
                    mode       = "lines+markers",
                    marker     = dict(size=8, color="#1D9E75",
                                      line=dict(width=1, color="#0d5c42")),
                    line       = dict(color="#1D9E75", width=1.5),
                    customdata = np.stack(
                        [df_s["power"], df_s["batch"], df_s["batch_type"]], axis=1
                    ),
                    hovertemplate=(
                        "<b>Speed %{x:.0f} mm/min</b><br>"
                        "Power: %{customdata[0]:.1f} W<br>"
                        "R: %{y:.4f} Ω<br>"
                        "Batch %{customdata[1]} (%{customdata[2]})"
                        "<extra></extra>"
                    ),
                    name="Measurements"
                ))
                fig2.add_trace(go.Scatter(
                    x    = [df_s.loc[min_is, "speed"]],
                    y    = [df_s.loc[min_is, "resistance"]],
                    mode = "markers",
                    marker = dict(size=16, color="#E24B4A", symbol="star",
                                  line=dict(width=1.5, color="#8a1c1c")),
                    name = "Minimum R",
                    hovertemplate=(
                        "⭐ <b>Minimum</b><br>"
                        f"Speed: {int(df_s.loc[min_is, 'speed'])} mm/min<br>"
                        f"R: {df_s.loc[min_is, 'resistance']:.4f} Ω"
                        "<extra></extra>"
                    )
                ))
                fig2.update_layout(
                    xaxis_title   = "Scanning Speed (mm/min)",
                    yaxis_title   = "Resistance (Ω)",
                    xaxis_type    = "log",      # Log scale for wide speed range
                    height        = 450,
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    xaxis  = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                                  zeroline=False),
                    yaxis  = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                                  zeroline=False),
                    legend = dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig2, use_container_width=True)

    # =========================================================================
    # TAB 3 — RESULTS
    # =========================================================================
    with tab_results:
        if not all_exps:
            st.info("No experiments yet.")
        else:
            best = min(all_exps, key=lambda e: e["resistance"])

            # ── Best observed ─────────────────────────────────────────────────
            st.subheader("Best Observed Parameters")
            r1, r2, r3 = st.columns(3)
            r1.metric("Optimal power",  f"{best['power']:.1f} W")
            r2.metric("Optimal speed",  f"{int(best['speed'])} mm/min")
            r3.metric("Min resistance", f"{best['resistance']:.4f} Ω")

            # ── GP predicted optimum ──────────────────────────────────────────
            opt = st.session_state.optimizer
            if opt and opt.gp is not None:
                with st.spinner("Computing GP predicted optimum…"):
                    pred = opt.get_predicted_optimal()
                # Only show if GP thinks it can do ≥2 % better
                if pred and pred[2] < best["resistance"] * 0.98:
                    st.markdown("---")
                    st.subheader("GP Model Predicted Optimum  *(not yet tested)*")
                    st.caption(
                        "The surrogate model believes a better point may exist here. "
                        "If this looks physically plausible, run another BO batch."
                    )
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Predicted power",  f"{pred[0]:.1f} W")
                    p2.metric("Predicted speed",  f"{int(pred[1])} mm/min")
                    p3.metric("Predicted min R",   f"{pred[2]:.4f} Ω")

            # ── Convergence curve ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Convergence")
            st.caption("Best resistance found so far, plotted as experiments accumulate.")
            run_min = np.inf
            conv    = []
            for i, e in enumerate(
                sorted(all_exps, key=lambda x: x.get("timestamp", "")), 1
            ):
                run_min = min(run_min, e["resistance"])
                conv.append({"n": i, "R": run_min})

            df_c = pd.DataFrame(conv)
            figc = go.Figure()
            figc.add_trace(go.Scatter(
                x    = df_c["n"],
                y    = df_c["R"],
                mode = "lines+markers",
                marker = dict(size=5, color="#534AB7",
                              line=dict(width=1, color="#2a1c8a")),
                line   = dict(color="#534AB7", width=2),
                hovertemplate="Experiment %{x}<br>Best R: %{y:.4f} Ω<extra></extra>",
                name = "Best R so far"
            ))
            figc.update_layout(
                xaxis_title   = "Experiment #",
                yaxis_title   = "Best Resistance (Ω)",
                height        = 300,
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                xaxis = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                yaxis = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            )
            st.plotly_chart(figc, use_container_width=True)

            # ── Full experiment table ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("All Experiments")
            df_all = (
                pd.DataFrame(all_exps)
                [["batch", "batch_type", "power", "speed", "resistance"]]
                .copy()
            )
            df_all["speed"]  = df_all["speed"].astype(int)
            df_all.columns   = ["Batch", "Type", "Power (W)",
                                 "Speed (mm/min)", "R (Ω)"]
            min_r = df_all["R (Ω)"].min()

            def _hl(row):
                return ["background-color:#d4fce8" if row["R (Ω)"] == min_r
                        else "" for _ in row]

            st.dataframe(
                df_all.style.apply(_hl, axis=1),
                use_container_width=True,
                hide_index=True
            )

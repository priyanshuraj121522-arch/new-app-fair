
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt

from dcf import (
    build_growth_path,
    project_fcff,
    discount_cashflows,
    terminal_value_gordon,
    enterprise_to_equity,
    compute_cost_of_equity_capm,
    compute_wacc,
    fmt_inr,
)
from data_fetchers import (
    fetch_from_yfinance, infer_net_debt_yf, infer_shares_yf, get_current_price_yf,
    auto_wacc_best_effort
)
from upload_parsers import read_uploaded_financials, TEMPLATE_CSV_BYTES

st.set_page_config(page_title="India DCF — Fair Value (INR) v4", page_icon="📈", layout="wide")
st.title("📈 Indian Stocks — DCF Fair Value Calculator (INR) — v4")

with st.expander("About & Notes", expanded=False):
    st.markdown("""
- **Modes:** Auto (Yahoo Finance), Manual, or **Upload** your data (CSV/XLSX).  
- **Discount Rate:** Three options — Manual WACC, CAPM (semi‑auto), and **Full Auto WACC** (fetch & infer).  
- **Tickers:** NSE `.NS` (e.g., `RELIANCE.NS`), BSE `.BO` (e.g., `500325.BO`).  
- **Units:** Enterprise/Equity values in **₹ Crore**. Per-share in **₹** (using **Shares (Crore)**).  
- **Disclaimer:** Educational tool. Not investment advice.
    """)

# ------------------------ Sidebar Inputs ------------------------
st.sidebar.header("1) Mode & Company")
mode = st.sidebar.selectbox("Choose input mode", ["Yahoo Finance (auto)", "Manual", "Upload file (CSV/XLSX)"], index=0)
ticker = st.sidebar.text_input("Ticker (e.g., RELIANCE.NS, TCS.NS) — used in Auto", value="RELIANCE.NS").strip()

st.sidebar.header("2) Base FCF Source")
use_ttm = st.sidebar.selectbox("Base FCF selection (for Auto/Upload)", ["TTM (latest)", "Average last 3 FY", "Average last 5 FY"], index=0)

st.sidebar.header("3) Growth & Terminal")
years_explicit = st.sidebar.number_input("Explicit forecast years", min_value=3, max_value=15, value=10, step=1)
g1 = st.sidebar.number_input("Years 1–5 CAGR (%)", value=10.0, step=0.5, help="Applied to early high-growth phase")
g2 = st.sidebar.number_input("Years 6–{0} CAGR (%)".format(years_explicit), value=6.0, step=0.5, help="Mid-phase growth before terminal")
gT = st.sidebar.number_input("Terminal growth (%)", value=4.0, step=0.25, help="Long-run growth, must be < discount rate")

st.sidebar.header("4) Discount Rate (WACC)")
choice_dis = st.sidebar.radio("WACC Mode", ["Manual WACC", "Compute WACC (CAPM)", "Full Auto (fetch & infer)"])

if choice_dis == "Manual WACC":
    wacc_input = st.sidebar.number_input("WACC (%)", value=12.0, step=0.25)
    rf=beta=mrp=kd=tax_rate=wd=we=None
elif choice_dis == "Compute WACC (CAPM)":
    rf = st.sidebar.number_input("Risk-free rate (%)", value=7.2, step=0.1, help="India 10Y G-Sec (enter current)")
    beta = st.sidebar.number_input("Beta (levered)", value=1.0, step=0.05)
    mrp = st.sidebar.number_input("Equity risk premium (%)", value=6.0, step=0.1)
    kd = st.sidebar.number_input("Pre-tax cost of debt (%)", value=8.5, step=0.1)
    tax_rate = st.sidebar.number_input("Tax rate (%)", value=25.0, step=0.5)
    we = st.sidebar.slider("Equity weight (E/(D+E))", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    wd = 1.0 - we
    wacc_input = None
else:
    st.sidebar.caption("We will fetch/estimate rf, beta, MRP, Kd, tax rate, and weights from market/financials. You can still override below after it computes.")
    auto_override = st.sidebar.checkbox("Allow manual override after auto-compute", value=True)
    wacc_input=rf=beta=mrp=kd=tax_rate=wd=we=None

st.sidebar.header("5) Balance Sheet & Share Data (override if needed)")
override_net_debt = st.sidebar.number_input("Net debt override (₹ Cr) — optional", value=0.0, step=10.0)
override_shares = st.sidebar.number_input("Shares outstanding override (Crore shares) — optional", value=0.0, step=0.01)

mos = st.sidebar.slider("Margin of Safety (%)", 0, 50, 15, step=5)

st.sidebar.header("6) Debug")
show_debug = st.sidebar.checkbox("Show debug details", value=False)

# ------------------------ Ingest Data ------------------------
base_fcf = None
net_debt = None
shares_out = None
current_price = None
msg = ""

fin = {}
uploaded_df = None

if mode == "Yahoo Finance (auto)":
    try:
        fin = fetch_from_yfinance(ticker)
        ok = fin is not None and not fin.get("error", False)
        msg = fin.get("message", "")
        if ok:
            if use_ttm.startswith("TTM"):
                base_fcf = fin.get("base_fcf_ttm_cr", 0.0)
            elif "3" in use_ttm:
                base_fcf = fin.get("base_fcf_avg3_cr", 0.0)
            else:
                base_fcf = fin.get("base_fcf_avg5_cr", 0.0)
            net_debt = fin.get("net_debt_cr", None)
            if net_debt is None:
                net_debt = infer_net_debt_yf(fin)
            shares_out = infer_shares_yf(fin)
            current_price = get_current_price_yf(fin)
        else:
            st.warning("Yahoo fetch failed. You can switch to **Manual** or **Upload**.")
    except Exception as e:
        st.warning(f"Yahoo fetch error: {e}")

elif mode == "Upload file (CSV/XLSX)":
    st.subheader("Upload your financial report")
    st.caption("Accepted: CSV/XLSX with columns like: year, ocf, capex, total_debt, cash, short_term_investments, shares (any order). Units should be in **INR absolute** (₹), we convert to Crores inside.")
    up = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
    st.download_button("Download template (CSV)", TEMPLATE_CSV_BYTES, "dcf_template.csv", "text/csv")
    if up is not None:
        try:
            uploaded = read_uploaded_financials(up)
            uploaded_df = uploaded["df"]
            shares_out = uploaded.get("shares_crore", None)
            if use_ttm.startswith("TTM") and uploaded.get("fcf_ttm_cr") is not None:
                base_fcf = uploaded["fcf_ttm_cr"]
            elif "3" in use_ttm and uploaded.get("fcf_avg3_cr") is not None:
                base_fcf = uploaded["fcf_avg3_cr"]
            elif uploaded.get("fcf_avg5_cr") is not None:
                base_fcf = uploaded["fcf_avg5_cr"]
            else:
                base_fcf = uploaded.get("fcf_last_cr", None)
            net_debt = uploaded.get("net_debt_cr", None)
            st.success("Uploaded file parsed successfully.")
            if show_debug:
                st.write("Parsed data preview:")
                st.dataframe(uploaded_df, use_container_width=True)
                st.json({k:v for k,v in uploaded.items() if k not in ["df"]})
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
else:
    st.info("Manual mode selected. Enter values below.")

# ------------------------ Manual Inputs (always visible for override/fallback) ------------------------
st.subheader("Base Inputs (override as needed)")
c1, c2, c3 = st.columns(3)
with c1:
    base_fcf = st.number_input("Base Free Cash Flow to Firm — FCFF (₹ Cr)", value=float(base_fcf or 0.0), step=10.0, format="%.2f")
with c2:
    net_debt = st.number_input("Net Debt (₹ Cr)", value=float(0.0 if net_debt is None else net_debt), step=10.0, format="%.2f")
with c3:
    shares_out = st.number_input("Shares Outstanding (Crore)", value=float(0.0 if shares_out is None else shares_out), step=0.01, format="%.4f")

if override_net_debt:
    net_debt = override_net_debt
if override_shares:
    shares_out = override_shares

if show_debug and mode == "Yahoo Finance (auto)":
    st.info(f"Yahoo status: {msg}")

# ------------------------ Full Auto WACC (if selected) ------------------------
auto_wacc_details = None
if choice_dis == "Full Auto (fetch & infer)":
    if mode != "Yahoo Finance (auto)":
        st.warning("Full Auto WACC works best with **Yahoo Finance (auto)** mode (needs market cap/financials). It will still try with manual/upload data if ticker is provided.")
    try:
        auto = auto_wacc_best_effort(ticker, fin if fin else None)
        auto_wacc_details = auto
        wacc = auto.get("wacc", None)
        rf = auto.get("rf", None)
        beta = auto.get("beta", None)
        mrp = auto.get("mrp", None)
        kd = auto.get("kd", None)
        tax_rate = auto.get("tax_rate", None)
        we = auto.get("we", None); wd = auto.get("wd", None)
        if st.sidebar.checkbox("Show auto WACC breakdown", value=True):
            st.sidebar.json({k:(float(v) if isinstance(v,(int,float)) else v) for k,v in auto.items() if k!="regression_points"})
        if auto.get("warning"):
            st.sidebar.warning(auto["warning"])
        if auto.get("note"):
            st.sidebar.caption(auto["note"])
        if st.sidebar.checkbox("Override auto WACC values", value=False):
            rf = st.sidebar.number_input("Risk-free rate (%) [auto]", value=float(rf*100.0 if rf else 7.2), step=0.1)/100.0
            beta = st.sidebar.number_input("Beta (levered) [auto]", value=float(beta if beta else 1.0), step=0.05)
            mrp = st.sidebar.number_input("Equity risk premium (%) [auto]", value=float(mrp*100.0 if mrp else 6.0), step=0.1)/100.0
            kd = st.sidebar.number_input("Pre-tax cost of debt (%) [auto]", value=float(kd*100.0 if kd else 8.5), step=0.1)/100.0
            tax_rate = st.sidebar.number_input("Tax rate (%) [auto]", value=float(tax_rate*100.0 if tax_rate else 25.0), step=0.5)/100.0
            we = st.sidebar.slider("Equity weight [auto]", 0.0, 1.0, float(we if we is not None else 0.8), 0.05); wd = 1.0 - we
            wacc = compute_wacc(ke=rf + beta*mrp, kd=kd, tax_rate=tax_rate, we=we, wd=wd)
    except Exception as e:
        st.error(f"Auto WACC failed: {e}")
else:
    wacc = None  # set later

# ------------------------ Run DCF ------------------------
run = st.button("Run Valuation")

if run:
    if base_fcf is None or base_fcf <= 0:
        st.error("Base FCFF must be a positive number. Please input manually or via Upload.")
        st.stop()
    if shares_out is None or shares_out <= 0:
        st.error("Shares Outstanding must be provided (Crore shares).")
        st.stop()
    if net_debt is None:
        net_debt = 0.0

    if choice_dis == "Manual WACC":
        wacc = wacc_input / 100.0
    elif choice_dis == "Compute WACC (CAPM)":
        ke = compute_cost_of_equity_capm(rf/100.0, beta, mrp/100.0)
        wacc = compute_wacc(ke=ke, kd=kd/100.0, tax_rate=tax_rate/100.0, we=we, wd=1.0-we)
    else:
        if wacc is None:
            st.error("Auto WACC could not be computed. Try CAPM mode or override values.")
            st.stop()

    if gT/100.0 >= wacc:
        st.error("Terminal growth must be **less** than discount rate (WACC). Please adjust.")
        st.stop()

    growth_path = build_growth_path(years=years_explicit, g1=g1/100.0, g2=g2/100.0)
    fcff_proj = project_fcff(base_fcf, growth_path)
    disc_factors = [(1.0 / ((1.0 + wacc) ** t)) for t in range(1, years_explicit + 1)]
    pv_fcff = discount_cashflows(fcff_proj, wacc)

    tv = terminal_value_gordon(fcff_proj[-1], gT/100.0, wacc)
    pv_tv = tv * disc_factors[-1]

    enterprise_val = pv_fcff + pv_tv
    equity_val = enterprise_to_equity(enterprise_val, net_debt)
    fair_value = equity_val / (shares_out if shares_out > 0 else 1.0)
    mos_price = fair_value * (1 - mos/100.0)

    st.subheader("Results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enterprise Value (₹ Cr)", fmt_inr(enterprise_val, cr=True))
    k2.metric("Equity Value (₹ Cr)", fmt_inr(equity_val, cr=True))
    k3.metric("Fair Value / Share (₹)", fmt_inr(fair_value, cr=False))
    k4.metric("MoS Price / Share (₹)", fmt_inr(mos_price, cr=False))

    if wacc is not None:
        st.caption(f"Using WACC = {wacc*100.0:.2f}%")

    if auto_wacc_details and st.checkbox("Show Auto WACC regression chart (beta estimation)", value=False):
        pts = auto_wacc_details.get("regression_points")
        if pts is not None and len(pts) > 0:
            import matplotlib.pyplot as plt
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            fig = plt.figure()
            plt.scatter(x, y, s=6)
            if len(x) >= 2:
                import numpy as np
                b = np.polyfit(x, y, 1)
                xx = np.linspace(min(x), max(x), 100)
                yy = b[0]*xx + b[1]
                plt.plot(xx, yy)
                plt.title(f"Beta regression (beta≈{b[0]:.2f})")
                plt.xlabel("NIFTY return"); plt.ylabel("Stock return")
            st.pyplot(fig)

    if current_price:
        try:
            upside = (fair_value - current_price) / current_price * 100.0
            st.metric("Upside vs Current", f"{upside:.1f}%")
        except Exception:
            pass

    df = pd.DataFrame({
        "Year": list(range(1, years_explicit + 1)),
        "FCFF (₹ Cr)": fcff_proj,
        "Discount Factor": disc_factors,
        "PV of FCFF (₹ Cr)": [fcff_proj[i] * disc_factors[i] for i in range(years_explicit)]
    })
    st.dataframe(df, use_container_width=True)

    fig = plt.figure()
    plt.plot(df["Year"], df["FCFF (₹ Cr)"])
    plt.xlabel("Year")
    plt.ylabel("FCFF (₹ Cr)")
    plt.title("Projected FCFF")
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Projections (CSV)", csv, file_name=f"{ticker or 'custom'}_dcf_projections.csv", mime="text/csv")

else:
    st.info("Choose a mode (Auto / Manual / Upload), set WACC mode, then click **Run Valuation**.")

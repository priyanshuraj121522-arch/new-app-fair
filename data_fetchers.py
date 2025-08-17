
import pandas as pd
import numpy as np
import yfinance as yf

def pick(df, candidates):
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None

def to_crore(x):
    if x is None:
        return None
    try:
        if hasattr(x, "astype"):
            return (x.astype(float) / 1e7).dropna()
        return float(x) / 1e7
    except Exception:
        return None

def fetch_from_yfinance(ticker: str) -> dict:
    info = {"error": False, "message": ""}
    try:
        t = yf.Ticker(ticker)
        prices_df = t.history(period="3mo", interval="1d")
        info["prices_df"] = prices_df

        cf_df = t.cashflow
        bs_df = t.balance_sheet
        is_df = t.income_stmt
        info["cf_df"] = cf_df
        info["bs_df"] = bs_df
        info["is_df"] = is_df
        info["info"] = t.info if hasattr(t, "info") else {}

        ocf_series = pick(cf_df, [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Total Cash From Operating Activities USD",
        ])
        capex_series = pick(cf_df, [
            "Capital Expenditures",
            "Capital Expenditure",
            "CapitalExpenditures",
        ])

        fcf_series = None
        if ocf_series is not None and capex_series is not None:
            fcf_series = ocf_series.astype(float) - capex_series.astype(float)
        fcf_cr = to_crore(fcf_series)

        base_ttm_cr = None
        try:
            qcf = t.quarterly_cashflow
            q_ocf = pick(qcf, [
                "Total Cash From Operating Activities",
                "Operating Cash Flow",
            ])
            q_capex = pick(qcf, [
                "Capital Expenditures",
                "Capital Expenditure",
                "CapitalExpenditures",
            ])
            if q_ocf is not None and q_capex is not None:
                fcf_q = (q_ocf.astype(float) - q_capex.astype(float)).fillna(0.0)
                if len(fcf_q) >= 4:
                    base_ttm_cr = float(fcf_q.iloc[:4].sum() / 1e7)
        except Exception:
            pass

        if base_ttm_cr is None and fcf_cr is not None and not fcf_cr.empty:
            base_ttm_cr = float(fcf_cr.iloc[0])

        if fcf_cr is not None and not fcf_cr.empty:
            vals = fcf_cr.iloc[:5].astype(float).tolist()
            avg3 = float(np.mean(vals[:3])) if len(vals) >= 3 else float(np.mean(vals))
            avg5 = float(np.mean(vals[:5])) if len(vals) >= 5 else float(np.mean(vals))
        else:
            avg3 = 0.0
            avg5 = 0.0

        total_debt = pick(bs_df, ["Total Debt"])
        if total_debt is not None:
            td = float(total_debt.iloc[0])
        else:
            short_debt = pick(bs_df, ["Short/Current Long Term Debt","Short Long Term Debt","Short Term Debt"])
            long_debt = pick(bs_df, ["Long Term Debt","Long Term Debt Total"])
            sd = float(short_debt.iloc[0]) if short_debt is not None else 0.0
            ld = float(long_debt.iloc[0]) if long_debt is not None else 0.0
            td = sd + ld

        cash_eq = pick(bs_df, ["Cash And Cash Equivalents","Cash","Cash And Cash Equivalents USD"])
        st_invest = pick(bs_df, ["Short Term Investments","Other Short Term Investments"])

        cash_val = float(cash_eq.iloc[0]) if cash_eq is not None else 0.0
        sti_val = float(st_invest.iloc[0]) if st_invest is not None else 0.0
        net_debt_cr = (td - (cash_val + sti_val)) / 1e7 if td is not None else None

        info.update({
            "base_fcf_ttm_cr": float(base_ttm_cr) if base_ttm_cr is not None else 0.0,
            "base_fcf_avg3_cr": float(avg3),
            "base_fcf_avg5_cr": float(avg5),
            "net_debt_cr": net_debt_cr,
            "error": False,
            "message": "OK",
        })
        return info
    except Exception as e:
        return {"error": True, "message": str(e)}

def _try_fetch_india_10y_yield():
    """Best-effort India 10Y G-Sec yield via yfinance.
    Tries a few likely symbols; returns decimal (e.g., 0.072) or None.
    """
    symbols = [
        # These may or may not exist depending on Yahoo schema; we try and ignore failures.
        "^IN10Y", "IN10Y.BOND", "IND10Y.BOND", "10YIND.BOND", "^GSEC10Y", "IN10Y:IND"
    ]
    for s in symbols:
        try:
            t = yf.Ticker(s)
            hist = t.history(period="1mo", interval="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                last = float(hist["Close"].dropna().iloc[-1])
                # If the series is in % points (like 7.2), convert to decimal
                return last/100.0 if last > 1.0 else last
        except Exception:
            continue
    return None

def _estimate_beta_vs_nifty(ticker: str, lookback_years: int = 2):
    """Estimate beta by regressing stock daily returns against NIFTY 50 (^NSEI)."""
    try:
        stock = yf.Ticker(ticker).history(period=f"{lookback_years}y", interval="1d")["Close"].pct_change().dropna()
        index = yf.Ticker("^NSEI").history(period=f"{lookback_years}y", interval="1d")["Close"].pct_change().dropna()
        df = pd.concat([stock.rename("s"), index.rename("m")], axis=1).dropna()
        if len(df) < 50:
            return None, None
        cov = np.cov(df["m"], df["s"])[0,1]
        var_m = np.var(df["m"])
        beta = cov/var_m if var_m != 0 else None
        pts = list(zip(df["m"].values.tolist(), df["s"].values.tolist()))
        return float(beta) if beta is not None else None, pts
    except Exception:
        return None, None

def auto_wacc_best_effort(ticker: str, fin: dict | None = None) -> dict:
    """Compute WACC using best-effort fetched/derived inputs with sensible fallbacks."""
    out = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info if hasattr(t, "info") else {}
        market_cap = info.get("marketCap")
        shares = info.get("sharesOutstanding")
        last_price = None
        try:
            px = t.history(period="1mo", interval="1d")["Close"]
            if px is not None and not px.empty:
                last_price = float(px.iloc[-1])
        except Exception:
            pass

        # Risk-free rate
        rf = _try_fetch_india_10y_yield()
        if rf is None:
            rf = 0.072  # default fallback 7.2%
            out["warning"] = out.get("warning","") + " rf fallback used; "

        # Beta
        beta = info.get("beta")
        reg_pts = None
        if beta is None:
            beta, reg_pts = _estimate_beta_vs_nifty(ticker)
            if beta is None:
                beta = 1.0
                out["warning"] = out.get("warning","") + " beta fallback=1.0; "

        # Market risk premium
        mrp = 0.06  # common India ERP assumption; editable
        # Cost of debt
        is_df = t.income_stmt
        bs_df = t.balance_sheet
        interest = None
        debt = None
        try:
            if is_df is not None and not is_df.empty:
                candidates = ["Interest Expense", "Net Interest Income", "Interest Expense Non Operating"]
                for c in candidates:
                    if c in is_df.index:
                        interest = float(is_df.loc[c].iloc[0])
                        break
            if bs_df is not None and not bs_df.empty:
                if "Total Debt" in bs_df.index:
                    debt = float(bs_df.loc["Total Debt"].iloc[0])
                else:
                    sd = bs_df.loc["Short/Current Long Term Debt"].iloc[0] if "Short/Current Long Term Debt" in bs_df.index else 0.0
                    ld = bs_df.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in bs_df.index else 0.0
                    debt = float(sd) + float(ld)
        except Exception:
            pass
        kd = None
        if debt and debt > 0 and interest:
            kd = abs(interest) / debt  # approximate yield on debt
        if kd is None:
            kd = 0.085  # fallback 8.5%
            out["warning"] = out.get("warning","") + " kd fallback used; "

        # Tax rate
        tax_rate = None
        try:
            if is_df is not None and not is_df.empty:
                tax_exp = None
                pre_tax = None
                if "Income Tax Expense" in is_df.index:
                    tax_exp = float(is_df.loc["Income Tax Expense"].iloc[0])
                elif "Provision for Income Taxes" in is_df.index:
                    tax_exp = float(is_df.loc["Provision for Income Taxes"].iloc[0])
                if "Pretax Income" in is_df.index:
                    pre_tax = float(is_df.loc["Pretax Income"].iloc[0])
                elif "Ebt" in is_df.index:
                    pre_tax = float(is_df.loc["Ebt"].iloc[0])
                if tax_exp is not None and pre_tax and pre_tax != 0:
                    tax_rate = max(0.0, min(0.35, tax_exp / pre_tax))
        except Exception:
            pass
        if tax_rate is None:
            tax_rate = 0.25  # fallback
            out["warning"] = out.get("warning","") + " tax fallback used; "

        # Capital structure weights
        if market_cap is None and last_price is not None and shares:
            market_cap = last_price * float(shares)
        if bs_df is not None and not bs_df.empty:
            if "Total Debt" in bs_df.index:
                debt_val = float(bs_df.loc["Total Debt"].iloc[0])
            else:
                sd = bs_df.loc["Short/Current Long Term Debt"].iloc[0] if "Short/Current Long Term Debt" in bs_df.index else 0.0
                ld = bs_df.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in bs_df.index else 0.0
                debt_val = float(sd) + float(ld)
        else:
            debt_val = None
        if market_cap is None or debt_val is None:
            out["warning"] = out.get("warning","") + " weights fallback 80/20; "
            we = 0.8; wd = 0.2
        else:
            E = float(market_cap)
            D = max(0.0, float(debt_val))
            if (E + D) <= 0:
                we = 0.8; wd = 0.2
                out["warning"] = out.get("warning","") + " weights fallback 80/20; "
            else:
                we = E / (E + D)
                wd = 1.0 - we

        ke = rf + beta * mrp
        wacc = we*ke + wd*kd*(1.0 - tax_rate)

        out.update({
            "rf": rf, "beta": beta, "mrp": mrp, "kd": kd, "tax_rate": tax_rate,
            "we": we, "wd": wd, "ke": ke, "wacc": wacc,
            "regression_points": reg_pts,
            "note": "Auto inputs are best-effort and may use fallbacks. You can override in the sidebar."
        })
        return out
    except Exception as e:
        return {"error": True, "message": str(e)}


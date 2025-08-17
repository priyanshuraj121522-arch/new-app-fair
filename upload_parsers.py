
import io
import pandas as pd
import numpy as np

TEMPLATE_COLUMNS = [
    "year",
    "operating_cash_flow",
    "capital_expenditures",
    "total_debt",
    "cash_and_equivalents",
    "short_term_investments",
    "shares_outstanding",
]

ALIASES = {
    "year": ["year", "fy", "fiscal_year"],
    "operating_cash_flow": ["operating_cash_flow","total_cash_from_operating_activities","ocf"],
    "capital_expenditures": ["capital_expenditures","capital_expenditure","capex"],
    "total_debt": ["total_debt","gross_debt","debt"],
    "cash_and_equivalents": ["cash_and_equivalents","cash","cash_and_cash_equivalents"],
    "short_term_investments": ["short_term_investments","sti","other_short_term_investments"],
    "shares_outstanding": ["shares_outstanding","shares","total_shares"],
}

def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        cn = _norm(str(col))
        for key, options in ALIASES.items():
            if cn in options:
                mapping[col] = key
                break
    for key in TEMPLATE_COLUMNS:
        if key in df.columns:
            mapping[key] = key
    df2 = df.rename(columns=mapping)
    return df2

def read_uploaded_financials(file) -> dict:
    if isinstance(file, (bytes, bytearray)):
        bio = io.BytesIO(file)
        df = pd.read_csv(bio)
    else:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

    df = _map_columns(df)
    for c in ["operating_cash_flow","capital_expenditures","total_debt","cash_and_equivalents","short_term_investments","shares_outstanding"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "operating_cash_flow" in df.columns and "capital_expenditures" in df.columns:
        df["fcf_inr"] = df["operating_cash_flow"] - df["capital_expenditures"]
        df["fcf_cr"] = df["fcf_inr"] / 1e7

    fcf_ttm_cr = None
    fcf_last_cr = None
    fcf_avg3_cr = None
    fcf_avg5_cr = None
    fcf_series = df["fcf_cr"].dropna() if "fcf_cr" in df.columns else pd.Series(dtype=float)
    if not fcf_series.empty:
        fcf_last_cr = float(fcf_series.iloc[-1])
        if len(fcf_series) >= 3:
            fcf_avg3_cr = float(fcf_series.iloc[-3:].mean())
        if len(fcf_series) >= 5:
            fcf_avg5_cr = float(fcf_series.iloc[-5:].mean())

    net_debt_cr = None
    if "total_debt" in df.columns:
        cash_val = float(df["cash_and_equivalents"].iloc[-1]) if "cash_and_equivalents" in df.columns and not df["cash_and_equivalents"].empty else 0.0
        sti_val = float(df["short_term_investments"].iloc[-1]) if "short_term_investments" in df.columns and not df["short_term_investments"].empty else 0.0
        debt_val = float(df["total_debt"].iloc[-1])
        net_debt_cr = (debt_val - (cash_val + sti_val)) / 1e7

    shares_crore = None
    if "shares_outstanding" in df.columns:
        shares_crore = float(df["shares_outstanding"].iloc[-1]) / 1e7

    return {
        "df": df,
        "fcf_ttm_cr": fcf_ttm_cr,
        "fcf_last_cr": fcf_last_cr,
        "fcf_avg3_cr": fcf_avg3_cr,
        "fcf_avg5_cr": fcf_avg5_cr,
        "net_debt_cr": net_debt_cr,
        "shares_crore": shares_crore,
    }

import pandas as pd
_template_df = pd.DataFrame({
    "year": [2021, 2022, 2023],
    "operating_cash_flow": [None, None, None],
    "capital_expenditures": [None, None, None],
    "total_debt": [None, None, None],
    "cash_and_equivalents": [None, None, None],
    "short_term_investments": [None, None, None],
    "shares_outstanding": [None, None, None],
})
TEMPLATE_CSV_BYTES = _template_df.to_csv(index=False).encode("utf-8")

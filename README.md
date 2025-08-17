# Indian Stocks DCF — v4 (Auto / Manual / Upload + **Full Auto WACC**)

Now includes **Full Auto WACC** that best‑effort fetches:
- Risk‑free rate (India 10Y G‑Sec, with fallback)
- Beta (Yahoo beta; else regression vs NIFTY ^NSEI)
- Market risk premium (default 6%)
- Cost of debt from interest expense ÷ total debt (fallback 8.5%)
- Tax rate from tax expense ÷ pretax income (fallback 25%)
- Capital structure weights from market cap and debt (fallback 80/20)

You can review the breakdown and override values in the sidebar.
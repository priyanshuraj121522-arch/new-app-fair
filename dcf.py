
def build_growth_path(years: int, g1: float, g2: float):
    n1 = years // 2
    n2 = years - n1
    return [g1]*n1 + [g2]*n2

def project_fcff(base_fcf: float, growth_path: list[float]) -> list[float]:
    fcff = []
    level = base_fcf
    for g in growth_path:
        level = level * (1.0 + g)
        fcff.append(level)
    return fcff

def discount_cashflows(fcff_list, discount_rate):
    pv = 0.0
    for t, cf in enumerate(fcff_list, start=1):
        pv += cf / ((1.0 + discount_rate) ** t)
    return pv

def terminal_value_gordon(last_fcff, g_terminal, discount_rate):
    if discount_rate <= g_terminal:
        raise ValueError("Terminal growth must be less than discount rate.")
    return last_fcff * (1.0 + g_terminal) / (discount_rate - g_terminal)

def enterprise_to_equity(enterprise_value, net_debt):
    return enterprise_value - net_debt

def compute_cost_of_equity_capm(rf: float, beta: float, mrp: float) -> float:
    return rf + beta * mrp

def compute_wacc(ke: float, kd: float, tax_rate: float, we: float, wd: float) -> float:
    return we*ke + wd*kd*(1.0 - tax_rate)

def fmt_inr(x: float, cr: bool = False) -> str:
    try:
        if cr:
            return f"₹ {x:,.2f} Cr"
        else:
            return f"₹ {x:,.2f}"
    except Exception:
        return str(x)

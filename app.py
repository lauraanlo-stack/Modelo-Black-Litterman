
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Black-Litterman | Gestión de Proveedores",
    page_icon="📊",
    layout="wide"
)

# ---------- Helpers ----------
@st.cache_data
def load_data():
    return pd.read_csv("proveedores.csv")

def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / len(w)

def build_returns(data):
    pivot = (
        data.pivot(index="month", columns="supplier", values="performance_score")
        .sort_index()
    )
    # Use monthly percentage changes as the performance "return"
    returns = pivot.pct_change().dropna()
    return returns, pivot

def black_litterman(
    market_weights,
    cov,
    delta,
    tau,
    P,
    Q,
    confidence,
):
    n = len(market_weights)

    # Equilibrium / implied returns
    pi = delta * cov @ market_weights

    # Omega: confidence controls the uncertainty of each view.
    # Higher confidence -> smaller uncertainty -> greater influence.
    tau_cov = tau * cov
    omega_diag = []
    for i in range(len(Q)):
        row = P[i]
        view_var = float(row @ tau_cov @ row.T)
        omega_diag.append(view_var * (1.0 - confidence[i] + 0.05))
    omega = np.diag(np.maximum(omega_diag, 1e-8))

    inv_tau_cov = np.linalg.pinv(tau_cov)
    inv_omega = np.linalg.pinv(omega)

    posterior_cov = np.linalg.pinv(inv_tau_cov + P.T @ inv_omega @ P)
    posterior_mean = posterior_cov @ (
        inv_tau_cov @ pi + P.T @ inv_omega @ Q
    )

    return pi, posterior_mean, posterior_cov, omega

def optimize_long_only(mu, cov, min_w, max_w, risk_aversion=4.0):
    """
    Lightweight projected-gradient optimizer:
    maximize mu'w - risk_aversion * w'Σw
    subject to sum(w)=1 and min/max bounds.
    """
    n = len(mu)
    w = np.ones(n) / n

    # Start from a feasible point based on market weights
    w = np.clip(w, min_w, max_w)
    w = normalize_weights(w)

    for _ in range(4000):
        grad = mu - 2 * risk_aversion * (cov @ w)
        step = 0.02 / (1 + _ / 500)
        candidate = w + step * grad
        candidate = np.clip(candidate, min_w, max_w)

        # Project approximately onto sum=1 while respecting bounds
        for _ in range(30):
            diff = 1.0 - candidate.sum()
            if abs(diff) < 1e-10:
                break
            free = (candidate > min_w + 1e-8) & (candidate < max_w - 1e-8)
            if diff > 0:
                free = candidate < max_w - 1e-8
            else:
                free = candidate > min_w + 1e-8
            if not free.any():
                break
            candidate[free] += diff / free.sum()
            candidate = np.clip(candidate, min_w, max_w)

        if np.max(np.abs(candidate - w)) < 1e-8:
            break
        w = candidate

    return normalize_weights(w)

# ---------- Data ----------
data = load_data()
returns, scores = build_returns(data)
suppliers = list(returns.columns)
n = len(suppliers)

market_weights = (
    data[["supplier", "spend_share"]]
    .drop_duplicates()
    .set_index("supplier")
    .reindex(suppliers)["spend_share"]
    .values
)
market_weights = normalize_weights(market_weights)

cov = returns.cov().values * 12  # annualized covariance
# Small regularization improves numerical stability
cov = cov + np.eye(n) * 1e-7

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Parámetros del modelo")

delta = st.sidebar.slider(
    "Aversión al riesgo (δ)",
    min_value=0.5, max_value=10.0, value=2.5, step=0.5,
    help="Mayor valor = mayor penalización al riesgo."
)

tau = st.sidebar.slider(
    "Incertidumbre del equilibrio (τ)",
    min_value=0.01, max_value=1.0, value=0.05, step=0.01,
    help="Controla cuánto nos alejamos del equilibrio para incorporar las views."
)

risk_aversion = st.sidebar.slider(
    "Penalización de riesgo en la optimización",
    min_value=0.5, max_value=15.0, value=4.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Views del gestor")

view_supplier = st.sidebar.selectbox(
    "Proveedor sobre el que tienes una expectativa",
    suppliers,
    index=suppliers.index("Proveedor C") if "Proveedor C" in suppliers else 0
)

view_pct = st.sidebar.slider(
    "Expectativa relativa del proveedor",
    min_value=-20.0, max_value=20.0, value=8.0, step=1.0,
    help="Ejemplo: +8% significa que esperas un desempeño superior al equilibrio."
)

confidence_pct = st.sidebar.slider(
    "Confianza en la view",
    min_value=5, max_value=95, value=75, step=5
)

relative_view = st.sidebar.checkbox(
    "Agregar una segunda view relativa",
    value=True
)

if relative_view:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        supplier_a = st.selectbox("Proveedor A", suppliers, index=0)
    with col2:
        supplier_b = st.selectbox(
            "Proveedor B",
            suppliers,
            index=1 if n > 1 else 0
        )
    relative_pct = st.sidebar.slider(
        "A supera a B (%)",
        min_value=-20.0, max_value=20.0, value=5.0, step=1.0
    )
    relative_conf = st.sidebar.slider(
        "Confianza",
        min_value=5, max_value=95, value=65, step=5
    )

# ---------- Views matrix ----------
P_rows = []
Q_vals = []
confidences = []

# Absolute view: supplier's expected return relative to equilibrium.
row = np.zeros(n)
idx = suppliers.index(view_supplier)
row[idx] = 1.0
P_rows.append(row)
Q_vals.append(float(view_pct) / 100.0)
confidences.append(confidence_pct / 100.0)

if relative_view and supplier_a != supplier_b:
    row = np.zeros(n)
    row[suppliers.index(supplier_a)] = 1.0
    row[suppliers.index(supplier_b)] = -1.0
    P_rows.append(row)
    Q_vals.append(float(relative_pct) / 100.0)
    confidences.append(relative_conf / 100.0)

P = np.vstack(P_rows)
Q = np.array(Q_vals)
confidence = np.array(confidences)

pi, mu_bl, posterior_cov, omega = black_litterman(
    market_weights, cov, delta, tau, P, Q, confidence
)

# ---------- Constraints ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Restricciones")
min_pct = st.sidebar.slider(
    "Mínimo por proveedor (%)", 0, 30, 5, 1
)
max_pct = st.sidebar.slider(
    "Máximo por proveedor (%)", 20, 60, 40, 1
)

min_w = min_pct / 100
max_w = max_pct / 100

if min_w * n > 1:
    st.error("El mínimo seleccionado es incompatible con el número de proveedores.")
    st.stop()

if max_w * n < 1:
    st.error("El máximo seleccionado es incompatible con el número de proveedores.")
    st.stop()

optimized_weights = optimize_long_only(
    mu_bl, posterior_cov, min_w, max_w, risk_aversion
)

# ---------- Metrics ----------
market_return = float(market_weights @ mu_bl)
optimized_return = float(optimized_weights @ mu_bl)

market_vol = float(np.sqrt(market_weights @ posterior_cov @ market_weights))
optimized_vol = float(np.sqrt(optimized_weights @ posterior_cov @ optimized_weights))

market_score = float(np.average(
    data.groupby("supplier")["performance_score"].mean().reindex(suppliers).values,
    weights=market_weights
))
optimized_score = float(np.average(
    data.groupby("supplier")["performance_score"].mean().reindex(suppliers).values,
    weights=optimized_weights
))

# ---------- UI ----------
st.title("📊 Black-Litterman aplicado a la Gestión de Proveedores")
st.markdown(
    "### Optimización de la cartera de abastecimiento mediante datos históricos, "
    "views del gestor y restricciones."
)

st.info(
    "La aplicación adapta la lógica Black-Litterman: el equilibrio actual del "
    "abastecimiento se combina con las expectativas del gestor y su nivel de "
    "confianza para obtener una asignación optimizada."
)

st.markdown("## 1. ¿Qué estamos optimizando?")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proveedores", n)
c2.metric("Participación actual", "100%")
c3.metric("Volatilidad BL actual", f"{market_vol:.2%}")
c4.metric("Volatilidad optimizada", f"{optimized_vol:.2%}")

st.markdown("### Variables utilizadas")
st.markdown(
    "- **Cumplimiento de PO**  \n"
    "- **Entregas a tiempo**  \n"
    "- **Tasa de excepciones**  \n"
    "- **Horas de procesamiento**  \n"
    "- **Calidad**  \n"
    "- **Estabilidad de precio**"
)

st.markdown("## 2. Equilibrio actual vs. asignación optimizada")

result = pd.DataFrame({
    "Proveedor": suppliers,
    "Asignación actual": market_weights,
    "Asignación BL": optimized_weights,
    "Cambio": optimized_weights - market_weights,
})

result_display = result.copy()
result_display["Asignación actual"] = result_display["Asignación actual"].map(lambda x: f"{x:.1%}")
result_display["Asignación BL"] = result_display["Asignación BL"].map(lambda x: f"{x:.1%}")
result_display["Cambio"] = result_display["Cambio"].map(lambda x: f"{x:+.1%}")

st.dataframe(result_display, use_container_width=True, hide_index=True)

chart_df = result.melt(
    id_vars="Proveedor",
    value_vars=["Asignación actual", "Asignación BL"],
    var_name="Tipo",
    value_name="Participación"
)

fig = px.bar(
    chart_df,
    x="Proveedor",
    y="Participación",
    color="Tipo",
    barmode="group",
    text_auto=".0%"
)
fig.update_layout(yaxis_tickformat=".0%", legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.markdown("## 3. ¿Qué dice Black-Litterman?")

bl_table = pd.DataFrame({
    "Proveedor": suppliers,
    "Retorno implícito (π)": pi,
    "Retorno ajustado BL (μBL)": mu_bl,
})

bl_display = bl_table.copy()
bl_display["Retorno implícito (π)"] = bl_display["Retorno implícito (π)"].map(lambda x: f"{x:.2%}")
bl_display["Retorno ajustado BL (μBL)"] = bl_display["Retorno ajustado BL (μBL)"].map(lambda x: f"{x:.2%}")

st.dataframe(bl_display, use_container_width=True, hide_index=True)

st.markdown("### Views incorporadas")
view_rows = []
for i, row in enumerate(P):
    if np.count_nonzero(row) == 1:
        supplier = suppliers[np.argmax(row)]
        description = f"{supplier}: expectativa {Q[i]:+.1%}"
    else:
        pos = np.where(row == 1)[0]
        neg = np.where(row == -1)[0]
        description = f"{suppliers[pos[0]]} vs. {suppliers[neg[0]]}: {Q[i]:+.1%}"
    view_rows.append({
        "View": description,
        "Confianza": confidence[i]
    })

views_display = pd.DataFrame(view_rows)
views_display["Confianza"] = views_display["Confianza"].map(lambda x: f"{x:.0%}")
st.dataframe(views_display, use_container_width=True, hide_index=True)

st.markdown("## 4. Indicadores de la cartera")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score histórico actual", f"{market_score:.1f}/100")
m2.metric("Score con asignación BL", f"{optimized_score:.1f}/100")
m3.metric("Desempeño esperado BL", f"{optimized_return:.2%}")
m4.metric("Cambio de volatilidad", f"{optimized_vol - market_vol:+.2%}")

st.markdown("## 5. Desempeño histórico")

history = (
    data.groupby("month", as_index=False)["performance_score"]
    .mean()
)
fig2 = px.line(
    history,
    x="month",
    y="performance_score",
    markers=True,
    labels={"month": "Mes", "performance_score": "Score promedio"}
)
fig2.update_yaxes(range=[70, 100])
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(
    "Proyecto académico. Los datos incluidos son sintéticos y sirven para demostrar "
    "la metodología. La aplicación adapta Black-Litterman a una cartera de proveedores; "
    "no representa una recomendación financiera ni una decisión real de compras."
)

with st.expander("🧮 Ver lógica matemática"):
    st.latex(r"\pi = \delta \Sigma w_m")
    st.latex(
        r"\mu_{BL} = "
        r"\left[(\tau\Sigma)^{-1}+P^T\Omega^{-1}P\right]^{-1}"
        r"\left[(\tau\Sigma)^{-1}\pi+P^T\Omega^{-1}Q\right]"
    )
    st.write(
        "π = rendimiento/desempeño implícito del equilibrio; "
        "Σ = matriz de covarianzas; wm = pesos actuales; "
        "P = estructura de las views; Q = expectativas; "
        "Ω = incertidumbre de las views; τ = incertidumbre del equilibrio."
    )

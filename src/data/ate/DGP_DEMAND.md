# Data Generating Process: Demand Design

This document describes the **demand** DGP used in the ATE experiments, including the
base (fully-observed) design and the MAR-modified extension where the outcome proxy
$W$ can be missing at random.

---

## 1. Notation and Variable Roles

| Symbol | Code variable | Role | Dimension |
|--------|--------------|------|-----------|
| $S$ | `demand` | Latent confounder (unobserved "true demand") | scalar |
| $Z = (Z_1, Z_2)$ | `(cost1, cost2)` | **Treatment proxy** — noisy signals of $S$ | $\mathbb{R}^2$ |
| $A$ | `price` | **Treatment** | scalar |
| $W$ | `views` | **Outcome proxy** — noisy signal of $S$ | scalar |
| $Y$ | `outcome` | **Outcome** (observed, noisy) | scalar |
| $X$ | — | **Backdoor covariates** (not used in demand; `None`) | — |
| $\delta_W$ | `delta_w` | **Missingness indicator** (MAR extension only): 1 = observed, 0 = missing | scalar |

The proxy-variable framework treats $S$ as unobserved.  $Z$ and $W$ serve as
proxies that satisfy the completeness conditions required for identification of the
structural function $\mathbb{E}[Y \mid do(A=a)]$.

---

## 2. Base DGP (Fully Observed)

Implemented in `demand_pv.py`.

### 2.1 Auxiliary function $\psi$

A non-linear transformation of demand that shapes all structural relationships:

$$
\psi(t) = 2\!\left(\frac{(t-5)^4}{600} + \exp\!\bigl(-4(t-5)^2\bigr) + \frac{t}{10} - 2\right)
$$

```python
def psi(t):
    return 2 * ((t - 5)**4 / 600 + np.exp(-4*(t - 5)**2) + t/10 - 2)
```

This function has a distinctive W-shape on $[0,10]$: it is large near the boundaries,
has a sharp peak centred at $t=5$, and dips in between.  It controls both the strength
of confounding and the curvature of the structural function.

### 2.2 Generative equations

All variables are generated for $i = 1, \ldots, n$:

1. **Latent confounder:**

$$
S_i \sim \text{Uniform}(0, 10)
$$

2. **Treatment proxies (cost shifters):**

$$
Z_{1,i} = 2\sin\!\left(\frac{2\pi\, S_i}{10}\right) + \varepsilon_{Z_1,i}, \qquad \varepsilon_{Z_1,i} \sim \mathcal{N}(0, 1)
$$

$$
Z_{2,i} = 2\cos\!\left(\frac{2\pi\, S_i}{10}\right) + \varepsilon_{Z_2,i}, \qquad \varepsilon_{Z_2,i} \sim \mathcal{N}(0, 1)
$$

The sine/cosine pair provides two complementary noisy views of the periodic structure
in demand. They are instrumental for price but not direct causes of the outcome
conditional on demand.

3. **Treatment (price):**

$$
A_i = 35 + (Z_{1,i} + 3)\,\psi(S_i) + Z_{2,i} + \varepsilon_{A,i}, \qquad \varepsilon_{A,i} \sim \mathcal{N}(0, 1)
$$

Price depends on demand $S$ (through $\psi$) and on costs $Z$.
The interaction $(Z_1+3)\,\psi(S)$ makes the price–demand relationship
non-separable.

4. **Outcome proxy (page views):**

$$
W_i = 7\,\psi(S_i) + 45 + \varepsilon_{W,i}, \qquad \varepsilon_{W,i} \sim \mathcal{N}(0, 1)
$$

Views depend only on demand through $\psi$, not on price or costs.
This is a noisy proxy for the latent confounder.

5. **Noiseless outcome:**

$$
Y_i^* = \min\!\bigl(\exp\!\bigl((W_i - A_i)/10\bigr),\; 5\bigr) \cdot A_i \;-\; 5\,\psi(S_i)
$$

```python
def cal_outcome(price, views, demand):
    return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)
```

The clipping at 5 prevents the exponential from exploding when views
greatly exceed price.

6. **Observed outcome (training only):**

$$
Y_i = Y_i^* + \varepsilon_{Y,i}, \qquad \varepsilon_{Y,i} \sim \mathcal{N}(0, 1)
$$

An additive measurement noise term is applied only to training data.

### 2.3 Confounding structure

The latent demand $S$ is a common cause of:
- treatment $A$ (through $\psi(S)$ in the price equation), and
- outcome $Y$ (through both $W$ and the direct $-5\psi(S)$ term).

Because $S$ is unobserved, a naive regression of $Y$ on $A$ is confounded.
The proxy variables $Z$ and $W$ enable identification via the proxy-variable
framework: $Z \perp\!\!\!\perp (W, Y) \mid S$ and $W \perp\!\!\!\perp (Z, A) \mid S$.

### 2.4 DAG

```
        S (latent)
       / | \
      /  |  \
     v   v   v
    Z    W    A ──> Y
              ^    ^
              |   /
              W -/
```

- $S \to Z$: demand drives cost shifters
- $S \to W$: demand drives page views
- $S \to A$: demand (via costs and $\psi$) drives price
- $A \to Y$: price directly affects revenue
- $W \to Y$: views affect revenue (through the exp term)
- $S \to Y$: demand has a direct effect on outcome ($-5\psi(S)$)

### 2.5 Structural function (ground truth)

The structural (causal) function is the expected outcome when price is intervened upon:

$$
h_0(a) \;=\; \mathbb{E}\bigl[Y \mid do(A = a)\bigr] \;=\; \mathbb{E}_{S,\,\varepsilon_W}\!\Big[\min\!\bigl(\exp\!\bigl((W - a)/10\bigr),\, 5\bigr)\cdot a - 5\,\psi(S)\Big]
$$

This is estimated by Monte Carlo over 10,000 draws of $S$ and $\varepsilon_W$ with
a fixed seed:

```python
def cal_structural(p: float):
    rng = default_rng(seed=42)
    demand = rng.uniform(0, 10.0, 10000)
    views = 7 * psi(demand) + 45 + rng.normal(0, 1.0, 10000)
    outcome = cal_outcome(p, views, demand)
    return np.mean(outcome)
```

The test set evaluates this at 10 equally spaced price points in $[10, 30]$.

### 2.6 Returned data objects

**Training** (`PVTrainDataSet`):

| Field | Contents | Shape |
|-------|----------|-------|
| `treatment` | $A$ (price) | $(n, 1)$ |
| `treatment_proxy` | $(Z_1, Z_2)$ (costs) | $(n, 2)$ |
| `outcome_proxy` | $W$ (views) | $(n, 1)$ |
| `outcome` | $Y$ (noisy outcome) | $(n, 1)$ |
| `backdoor` | `None` | — |

**Test** (`PVTestDataSet`):

| Field | Contents | Shape |
|-------|----------|-------|
| `treatment` | 10 price grid points | $(10, 1)$ |
| `structural` | $h_0(a)$ at each grid point | $(10, 1)$ |

---

## 3. MAR-Modified DGP

Implemented in `__init__.py :: generate_train_data_ate_mar()`.

This extension takes the fully-observed demand data and introduces **Missing At Random
(MAR) missingness** on the outcome proxy $W$.  Everything else (treatment, treatment
proxy, outcome) remains fully observed.

### 3.1 Motivation

In many real-world proxy-variable settings, one of the proxy measurements may be
partially missing.  For example, page-view data ($W$) might only be recorded for a
subset of transactions.  The MAR assumption states that the probability of observing
$W$ depends on the other observed variables but not on $W$ itself (conditional on
those variables).  This permits consistent estimation via inverse-probability weighting
(IPW) or doubly-robust (DR) corrections.

### 3.2 MAR mechanism

Given the fully-observed base data, the missingness indicator $\delta_{W,i}$ is
generated as follows.

**Step 1 — Construct the conditioning set $L^+$:**

$$
L^+_i = (A_i,\; Z_i,\; Y_i)
$$

(If backdoor covariates $X$ were present, they would also be included. For the demand
design, $X = \varnothing$.)

**Step 2 — Standardise:**

$$
\tilde{L}^+_i = \frac{L^+_i - \bar{L}^+}{\text{std}(L^+) + 10^{-8}}
$$

Column-wise zero-mean, unit-variance normalisation ensures the coefficients operate
on a comparable scale across variables.

**Step 3 — Compute linear score:**

$$
s_i = \tilde{L}^+_i \cdot \boldsymbol{\alpha}, \qquad \alpha_j = \texttt{mar\_alpha\_value} \;\;\forall j
$$

The default `mar_alpha_value` is **1.6** (configurable). All dimensions of $L^+$
contribute equally to the missingness score. Larger values of `mar_alpha_value`
produce more heterogeneous missingness probabilities (stronger MAR dependence on
observed variables).

**Step 4 — Calibrate intercept $\alpha_0$ via binary search:**

Solve for $\alpha_0$ such that the marginal missing rate matches the target:

$$
\frac{1}{n}\sum_{i=1}^{n} \sigma(\alpha_0 + s_i) = \texttt{missing\_rate}
$$

where $\sigma(\cdot)$ is the logistic sigmoid.  A 60-iteration bisection over
$[-20, 20]$ is used — this converges to machine precision.

**Step 5 — Sample missingness:**

$$
\Pr(\delta_{W,i} = 0 \mid L^+_i) = \sigma(\alpha_0 + s_i)
$$

$$
\delta_{W,i} = \begin{cases}1 & \text{if } U_i \geq \sigma(\alpha_0 + s_i) \\
0 & \text{otherwise}\end{cases}, \qquad U_i \sim \text{Uniform}(0,1)
$$

The RNG for the uniform draws uses `rand_seed + 1000` (separate from the base data
seed) to ensure missingness is independent of the data-generation randomness.

**Step 6 — Mask $W$ for missing rows:**

$$
W_i^{\text{MAR}} = \begin{cases} W_i & \text{if } \delta_{W,i} = 1 \\ 0 & \text{if } \delta_{W,i} = 0 \end{cases}
$$

The zero-fill is a placeholder; downstream models use $\delta_{W,i}$ to know which
entries are genuine observations.

### 3.3 Key properties of the MAR mechanism

1. **MAR, not MCAR:** Missingness depends on observed $(A, Z, Y)$ through the
   logistic model. It does **not** depend on $W$ itself (conditional on $L^+$),
   satisfying the MAR condition:
   $\delta_W \perp\!\!\!\perp W \mid A, Z, Y$.

2. **Controllable missing rate:** The `missing_rate` parameter (default 0.3) sets
   the marginal proportion of missing outcome-proxy values.

3. **Controllable MAR strength:** `mar_alpha_value` (default 1.6) controls how
   heterogeneous the per-observation missing probabilities are. At 0 the mechanism
   degenerates to MCAR; at large values the probabilities spread toward 0 and 1.

### 3.4 Returned data object

`PVTrainDataSetMAR` extends `PVTrainDataSet` with one additional field:

| Field | Contents | Shape |
|-------|----------|-------|
| `treatment` | $A$ (price) — unchanged | $(n, 1)$ |
| `treatment_proxy` | $(Z_1, Z_2)$ (costs) — unchanged | $(n, 2)$ |
| `outcome_proxy` | $W^{\text{MAR}}$ (views; 0 for missing rows) | $(n, 1)$ |
| `outcome` | $Y$ (noisy outcome) — unchanged | $(n, 1)$ |
| `backdoor` | `None` | — |
| **`delta_w`** | $\delta_W$ (1 = observed, 0 = missing) | $(n, 1)$ |

---

## 4. How the MAR Data is Consumed Downstream

The MAR dataset feeds into `DFPVTrainerMAR` (Algorithm 5.1 in the thesis).  The key
downstream components that rely on the MAR structure are:

### 4.1 Nuisance models (`NuisanceModels`)

Two nuisance models are trained on each cross-fitting fold:

| Model | Input | Target | Trained on |
|-------|-------|--------|-----------|
| **Propensity** $\hat{e}(L^+)$ | $L^+ = (A, Z, Y)$ | $\delta_W$ (binary) | All $n$ samples |
| **Imputation** $\hat{m}_\psi(L^+)$ | $L^+ = (A, Z, Y)$ | $\psi_{\theta_W}(W_i)$ (network output) | Complete cases only ($\delta_W = 1$) |

The propensity model estimates $\Pr(\delta_W = 1 \mid A, Z, Y)$.
The imputation model estimates $\mathbb{E}[\psi_{\theta_W}(W) \mid A, Z, Y]$.

### 4.2 Doubly-robust pseudo-outcome

The DR pseudo-outcome replaces the raw $\psi_{\theta_W}(W)$ in the Stage-1 regression:

$$
\hat{\varphi}_{DR,i} = \hat{m}_\psi(L^+_i) + \frac{\delta_{W,i}}{\hat{e}(L^+_i)}\Big(\psi_{\theta_W}(W_i) - \hat{m}_\psi(L^+_i)\Big)
$$

- When $\delta_{W,i} = 1$: the IPW-weighted residual corrects the imputation.
- When $\delta_{W,i} = 0$: the correction vanishes and only the imputation is used.
- The propensity score is clipped to $[\epsilon, 1-\epsilon]$ (default $\epsilon = 10^{-3}$) to stabilise weights.

This pseudo-outcome is **doubly robust**: it is consistent if *either* the propensity
model *or* the imputation model is correctly specified.

### 4.3 Modified two-stage least squares

- **Stage 1:** Regress $\hat{\varphi}_{DR}$ on $\varphi_A(A) \otimes \varphi_Z(Z) \otimes \varphi_X(X)$ to obtain $\hat{V}$.
- **Stage 2:** Regress $Y$ on $\hat{\mu}(L) = \hat{V} \cdot (\varphi_A(A) \otimes \varphi_Z(Z) \otimes \varphi_X(X))$ to obtain the structural estimate.

The outcome proxy network $\theta_W$ is updated via the Stage-2 loss (backpropagation
through the 2SLS), while Stage-1 treats $\psi_{\theta_W}$ targets as detached.

---

## 5. Configuration Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `n_sample` | `data_config` | — | Number of training observations |
| `missing_rate` | `data_config` | 0.3 | Target marginal fraction of missing $W$ |
| `mar_alpha_value` | `data_config` | 1.6 | Logistic coefficient magnitude (MAR strength) |
| `seed` / `rand_seed` | function arg | 42 | Base RNG seed; missingness uses `seed + 1000` |

---

## 6. Source Files

| File | Purpose |
|------|---------|
| `demand_pv.py` | Base demand DGP: `psi`, `generatate_demand_core`, `generate_train_demand_pv`, `generate_test_demand_pv`, `cal_outcome`, `cal_structural` |
| `__init__.py` | Entry points `generate_train_data_ate` / `generate_test_data_ate`, MAR wrapper `generate_train_data_ate_mar` |
| `data_class.py` | `PVTrainDataSet`, `PVTestDataSet` (fully-observed NamedTuples) |
| `data_class_mar.py` | `PVTrainDataSetMAR`, `PVTrainDataSetMARTorch`, K-fold utilities |

# Modified DFPV under MAR - Implementation Plan

**Created**: February 2026  
**Project**: Implementing Modified Deep Feature Proxy Variable (DFPV) for Missing At Random (MAR) outcome proxy data  
**Status**: Planning Phase - Ready for Implementation

---

## 📋 Executive Summary

### What We Want to Achieve

Extend the existing DFPV codebase to handle **missing outcome proxy ($W$) data** under a Missing At Random (MAR) mechanism, as described in the thesis Section 5.1.3 "Modified DFPV under MAR (DR Stage-1, LS Stage-2)".

### Why This Matters

- Original DFPV requires fully observed outcome proxy $W$
- In real applications, $W$ often has missing values
- Complete case analysis (discarding missing) loses efficiency and may introduce bias
- Modified DFPV uses Doubly Robust (DR) estimation to handle missingness while maintaining theoretical guarantees

### Key Innovation

Replace Stage 1's target from raw $\psi_W(W)$ to a **Doubly Robust pseudo-outcome** $\phi_{\text{DR}}$ that:
1. Uses all samples (not just complete cases)
2. Combines propensity score weighting and imputation
3. Provides consistent estimation if either model is correct

---

## 📚 Background Materials

### Primary Thesis Reference

**Location**: `/Users/apple/2602_WenPaper/Thesis_latex/Sections/methodology_DFPV.tex`

**Relevant Sections**:
- **Lines 1-150**: General Two-Stage Regression (baseline theory)
- **Lines 150-300**: Original DFPV Model (what we're extending)
- **Lines 300-end**: Modified DFPV under MAR (**our implementation target**)

**Key Equations**:
- **Eq. 5.12**: DR pseudo-outcome definition
- **Eq. 5.13**: Modified Stage 1 optimization
- **Eq. 5.14**: Modified Stage 2 optimization
- **Eq. 5.15**: Modified ATE estimation
- **Algorithm 5.1**: Cross-fitting procedure

### Existing Codebase Structure

**Base Path**: `/Users/apple/DeepFeatureProxyVariable/`

**Core Files to Understand**:

| File Path | What It Does | Why Relevant |
|-----------|-------------|--------------|
| `src/models/DFPV/model.py` | Original DFPV model logic | Contains `fit_2sls()`, `augment_stage1_feature()`, `augment_stage2_feature()` that we'll adapt |
| `src/models/DFPV/trainer.py` | Original training loop | Shows alternating Stage 1/Stage 2 updates we'll modify |
| `src/models/DFPV/nn_structure/*.py` | Neural network architectures | Reusable - no changes needed |
| `src/utils/pytorch_linear_reg_utils.py` | Ridge regression utilities | Reusable - `fit_linear()`, `linear_reg_pred()` |
| `src/data/ate/data_class.py` | Data structures | Need to extend for missing data |

---

## 🔍 Theoretical Understanding: Original vs Modified

### Original DFPV Pipeline

```
Stage 1: Learn to predict proxy representation
  Input:  (A, Z, X) → Neural networks → Features
  Target: ψ_W(W)  [only complete cases]
  Output: V̂, trained φ networks
  
Stage 2: Learn outcome bridge function  
  Input:  (A, X) → Neural networks → Features
  Target: Y
  Bridge: Use V̂ from Stage 1 to compute m̂_ψ(A,Z,X)
  Output: û, trained ψ and φ networks
  
  Note: ψ_W network is TRAINED in Stage 2

Prediction: β̂(a) = E[ĥ(a,W,X)] using averaged W,X features
```

### Modified DFPV Pipeline  

```
Stage 1: Learn to predict proxy representation (with DR adjustment)
  Input:  (A, Z, X) → Neural networks → Features
  Target: φ_DR  [doubly robust pseudo-outcome, ALL samples]
  Output: V̂, trained φ networks
  
  New Step: Compute φ_DR using:
    - Propensity score e(L^+) 
    - Imputation model m_ψ(L^+)
    - Cross-fitting (K-fold)
  
Stage 2: Learn outcome bridge function
  Input:  (A, X) → Neural networks → Features  
  Target: Y
  Bridge: Use V̂ from Stage 1 to compute μ̂(L)
  Output: û, trained ψ_{A(2)} and φ_{X(2)} networks
  
  Note: ψ_W network is FROZEN in Stage 2 (key difference!)

Prediction: β̂(a) = E[ĥ(a,W,X)] using μ̂(a,Z,X) (no raw W needed)
```

### Critical Differences Summary

| Aspect | Original DFPV | Modified DFPV |
|--------|--------------|---------------|
| **Missing Data** | Cannot handle | Handles via DR |
| **Stage 1 Target** | $\psi_W(W)$ (complete cases only) | $\phi_{\text{DR}}$ (all cases) |
| **Stage 1 Samples** | Complete cases | All samples |
| **Nuisance Models** | None | Propensity $e(L^+)$ + Imputation $m_\psi(L^+)$ |
| **Cross-Fitting** | Not needed | Required (K-fold) |
| **$\theta_W$ in Stage 2** | Trained (updated) | Frozen (not updated) |
| **Stage 2 Input** | $\hat{m}_\psi(A,Z,X)$ | $\hat{\mu}(L) = \hat{\mu}(A,Z,X)$ |
| **Prediction** | Uses mean W features | Uses imputed $\hat{\mu}$ |

---

## 🎯 Implementation Strategy

### Design Principles

1. **Maximize Code Reuse**: Don't modify existing DFPV code, create parallel MAR versions
2. **Clear Separation**: New files clearly marked with `_mar` suffix
3. **Backward Compatibility**: Original DFPV should work as before
4. **Modular Design**: Nuisance models, DR computation as separate, testable modules
5. **Theoretical Correctness**: Follow thesis Algorithm 5.1 exactly

### File Structure Plan

```
src/models/DFPV/
├── model.py                    # ✅ Keep as-is (original DFPV)
├── trainer.py                  # ✅ Keep as-is (original trainer)
├── model_mar.py                # 🆕 NEW: Modified DFPV model
├── trainer_mar.py              # 🆕 NEW: Modified DFPV trainer  
├── nuisance_models.py          # 🆕 NEW: Propensity + Imputation
├── dr_utils.py                 # 🆕 NEW: DR computation utilities
└── nn_structure/               # ✅ Keep as-is (reuse)
    └── *.py

src/data/ate/
├── data_class.py               # ✅ Keep as-is (original)
└── data_class_mar.py           # 🆕 NEW: Data structures with δ_W

experiments/
├── main_mar.py                 # 🆕 NEW: Experiment script
└── compare_original_vs_mar.py  # 🆕 NEW: Comparison script
```

---

## 🔧 Detailed Implementation Roadmap

### Phase 1: Foundational Components

#### 1.1 Data Structures for MAR

**New File**: `src/data/ate/data_class_mar.py`

**What to Create**:
- `PVTrainDataSetMAR`: NumPy version with `delta_W` field
- `PVTrainDataSetMARTorch`: PyTorch tensor version
- `create_k_folds()`: Split data into K folds for cross-fitting
- `merge_folds()`: Combine multiple folds

**Key Requirements**:
- `delta_W`: Binary indicator (1=observed, 0=missing)
- `outcome_proxy`: May contain NaN or masked values for missing entries
- Support slicing and indexing for fold operations

**Reference**:
- Original: `src/data/ate/data_class.py` classes `PVTrainDataSet`, `PVTrainDataSetTorch`
- Thesis: Section 5.1.3, paragraph "Setup"

**Why Needed**: Original data classes don't track missingness, new structure needed for MAR handling

---

#### 1.2 Nuisance Models Module

**New File**: `src/models/DFPV/nuisance_models.py`

**What to Create**:

**Class: `NuisanceModels`**
- **Purpose**: Manage propensity score and imputation models
- **Components**:
  - `propensity_net`: Binary classifier for $e(L^+) = \Pr(\delta_W=1 \mid A,Z,X,Y)$ — uses $L^+=(A,Z,X,Y)$
  - `imputation_net`: Regression model for $m_\psi(L) = \mathbb{E}[\psi_W(W) \mid A,Z,X]$ — uses $L=(A,Z,X)$ **NOT** $L^+$

**Key Methods**:
- `fit_propensity()`: Train on all samples with binary cross-entropy, input is $L^+$
- `fit_imputation()`: Train on complete cases only with MSE, input is $L$ (not $L^+$)
- `predict_propensity()`: Return $\hat{e}(L^+) \in [0,1]$
- `predict_imputation()`: Return $\hat{m}_\psi(L) \in \mathbb{R}^{d_W}$

**IMPORTANT — Thesis Correction**:
- Algorithm 5.1 line 6 explicitly trains imputation on `{ ψ_{θ_W}(W_i), L_i }` where `L_i=(A_i,Z_i,X_i)`
- Imputation input is **L**, propensity input is **L+**: these are different conditioning sets

**Design Considerations**:
- Use dropout for regularization (avoid overfitting)
- Moderate network depth (2-3 hidden layers, 32-64 units)
- Separate optimizers for independent training

**Reference**:
- Thesis: Algorithm 5.1 lines 4–6 (nuisance fitting step)
- Propensity: Standard logistic regression extension
- Imputation: Regression to feature space $\psi_W(W)$, not raw $W$; conditioned on $L$ only

**Why Needed**: DR pseudo-outcome requires these nuisance estimates

---

#### 1.3 DR Utilities Module

**New File**: `src/models/DFPV/dr_utils.py`

**What to Create**:

**Function: `construct_L_plus()`**
- **Purpose**: Concatenate $(A, Z, X, Y)$ into $L^+$
- **Handles**: Optional backdoor $X$ (may be None)

**Function: `compute_dr_pseudo_outcome()`**
- **Purpose**: Compute $\phi_{\text{DR}}$ from components
- **Formula** (Thesis Eq. 5.12):
  $$\phi_{\text{DR}} = \hat{m}_\psi(L^+) + \frac{\delta_W}{\hat{e}(L^+)} \{\psi_W(W) - \hat{m}_\psi(L^+)\}$$
- **Critical Details**:
  - Clip propensity to $[\epsilon, 1-\epsilon]$ (avoid division by zero)
  - For missing cases ($\delta_W=0$): DR correction = 0, use imputation only
  - For observed cases ($\delta_W=1$): Add IPW-weighted residual

**Function: `compute_effective_sample_size()`**
- **Purpose**: Diagnostic for propensity weight quality
- **Formula**: $\text{ESS} = (\sum \delta_i/e_i)^2 / \sum (\delta_i/e_i)^2$
- **Use**: Warn if ESS too low (extreme weights)

**Reference**:
- Thesis: Section 5.1.3, paragraph "Stage~1: DR regression"
- Standard DR estimation literature (Kennedy 2016, Chernozhukov et al. 2018)

**Why Needed**: Core DR computation isolated for testing and reuse

---

### Phase 2: Modified DFPV Model

#### 2.1 Core Model Class

**New File**: `src/models/DFPV/model_mar.py`

**What to Create**:

**Class: `DFPVModelMAR`**

**Purpose**: Parallel to `DFPVModel` but for MAR setting

**Key Attributes**:
- Same neural networks as original (treatment, proxy, backdoor, outcome_proxy)
- **New**: `nuisance_models` instance
- `stage1_weight`: $V$ (Stage 1 linear weights)
- `stage2_weight`: $u$ (Stage 2 linear weights)
- `mean_backdoor_feature`: For ATE marginalization

**Key Methods to Implement**:

1. **`fit_2sls_mar()`** (static method)
   - **Replaces**: Original `fit_2sls()` 
   - **Input Differences**:
     - Takes `phi_DR_1st` instead of `outcome_proxy_feature_1st`
     - No separate Stage 1/Stage 2 data split (cross-fitting handles this)
     - No `outcome_proxy_feature_2nd` input
   - **Stage 1**: Regress $\phi_{\text{DR}}$ on $(A,Z,X)$ features → solve for $V$
   - **Stage 2**: Regress $Y$ on $(A, \hat{\mu}(L), X)$ features → solve for $u$
   - **Reference**: Thesis Eq. 5.13 and 5.14

2. **`fit_t_mar()`**
   - **Purpose**: Final model fitting after all epochs
   - **Inputs**: Full training data, precomputed $\phi_{\text{DR}}$ for all samples
   - **Does**: Extract features, call `fit_2sls_mar()`, save weights

3. **`predict_t()`**
   - **Signature Change**: Takes `(treatment, treatment_proxy, backdoor)`
   - **Why**: Need $Z$ and $X$ to compute $\hat{\mu}(a, Z, X)$ per Eq. beta_plugin
   - **IMPORTANT**: The intervention treatment value `a` is used **both** in the Stage-2 term `ψ_{A(2)}(a)` AND inside `μ̂(a, Z_i, X_i)` as the A-argument of Stage-1 networks. This is the counterfactual: what would be predicted if A were set to `a` for each observed (Z_i, X_i)?
   - **Process**:
     1. Compute Stage-1 features with A=a: `φ_{A(1)}(a) ⊗ φ_Z(Z_i) ⊗ φ_{X(1)}(X_i)`
     2. Compute `μ̂(a, Z_i, X_i) = V̂ · (Stage-1 features)`
     3. Compute Stage-2 features: `ψ_{A(2)}(a) ⊗ μ̂(a, Z_i, X_i) ⊗ φ_{X(2)}(X_i)`
     4. Predict: `β̂(a) = (1/n) Σ_i û^T · (Stage-2 features)`
   - **Reference**: Thesis Eq. beta_plugin (ATE estimator section)

**What to Reuse**:
- `DFPVModel.augment_stage1_feature()` (Kronecker product logic)
- `DFPVModel.augment_stage2_feature()` (Kronecker product logic)
- `fit_linear()`, `linear_reg_pred()` from utils

**Critical Design Decision — θ_W Initialization (confirmed from thesis)**:
- $\theta_W$ (outcome_proxy_net) starts from **random initialization** — it is **never pre-trained**
- Algorithm 5.1 uses it as `θ_W^{(t)}` at each epoch `t`, meaning whatever value it currently holds
- It is **never updated** at any point: not in Stage 1 (target is `φ_DR`), not in Stage 2 (Eq. 5.16 optimizes only `(u, θ_{A(2)}, θ_{X(2)})`)
- Acts as a fixed random feature extractor throughout all training

**Reference**:
- Original: `src/models/DFPV/model.py` class `DFPVModel`
- Thesis: Section 5.1.3, paragraphs "Stage~2" and "Bridge and ATE estimators"

---

### Phase 3: Modified DFPV Trainer with Cross-Fitting

#### 3.1 Trainer Implementation

**New File**: `src/models/DFPV/trainer_mar.py`

**What to Create**:

**Class: `DFPVTrainerMAR`**

**Purpose**: Orchestrate training with K-fold cross-fitting

**Constructor Changes**:
- **New Parameters**:
  - `n_folds`: Number of folds (default: 5)
  - `nuisance_n_epochs`: Epochs for nuisance training
  - `nuisance_lr`: Learning rate for nuisance models
  - `propensity_clip`: Clipping threshold for propensity
- **New Attribute**: `nuisance_models` instance
- **Removed**: No optimizer for `outcome_proxy_net` in Stage 2

**Key Methods to Implement**:

1. **`train()`** - Main Training Loop
   - **Structure**: Nested loops (epochs → folds)
   - **Pseudo-code**:
     ```
     Create K folds
     For each epoch:
         For each fold k:
             1. Fit nuisance on folds {1,...,K}\{k}
             2. Compute φ_DR on fold k
             3. Update Stage 1 on fold k
             4. Update Stage 2 on fold k
     
     Final fit on all data with final nuisance models
     ```
   - **Reference**: Thesis Algorithm 5.1

2. **`_fit_nuisance_models()`**
   - **Input**: Training folds (excluding current validation fold)
   - **Does**:
     - Construct $L^+$ from data
     - Train propensity model on all samples
     - Train imputation model on complete cases only
   - **Note**: Uses current (frozen) $\theta_W^{(t)}$ to extract $\psi_W(W_i)$ targets for complete cases; imputation input is $L=(A,Z,X)$, propensity input is $L^+=(A,Z,X,Y)$

3. **`_compute_dr_pseudo_outcome()`**
   - **Input**: Validation fold, trained nuisance models
   - **Does**:
     - Predict $\hat{e}(L^+)$ and $\hat{m}_\psi(L^+)$
     - Extract $\psi_W(W)$ for observed cases
     - Call `compute_dr_pseudo_outcome()` from utils
   - **Returns**: $\phi_{\text{DR}}$ for validation fold

4. **`stage1_update_mar()`**
   - **Similar to**: Original `stage1_update()`
   - **Key Differences**:
     - Target is $\phi_{\text{DR}}$ (not $\psi_W(W)$)
     - Uses ALL samples in fold (not just complete)
     - `outcome_proxy_net` is always frozen (never trained)
   - **Reference**: Original `trainer.py` line ~60-100

5. **`stage2_update_mar()`**
   - **Similar to**: Original `stage2_update()`
   - **Key Differences**:
     - `outcome_proxy_net` remains frozen (critical!)
     - No `outcome_proxy_opt.step()` call
     - Calls `fit_2sls_mar()` instead of `fit_2sls()`
   - **Reference**: Original `trainer.py` line ~100-150

**Training Mode Management**:

| Network | Stage 1 Update | Stage 2 Update | Why |
|---------|---------------|---------------|-----|
| `treatment_1st_net` | `.train(True)` | `.train(False)` | Stage 1 only |
| `treatment_2nd_net` | `.train(False)` | `.train(True)` | Stage 2 only |
| `treatment_proxy_net` | `.train(True)` | `.train(False)` | Stage 1 only |
| `outcome_proxy_net` | `.train(False)` | `.train(False)` | **Never updated!** |
| `backdoor_1st_net` | `.train(True)` | `.train(False)` | Stage 1 only |
| `backdoor_2nd_net` | `.train(False)` | `.train(True)` | Stage 2 only |

**Why Cross-Fitting is Critical**:
- Prevents overfitting bias in nuisance models
- Ensures $\phi_{\text{DR}}^{(i)}$ uses models trained without sample $i$
- Necessary for double robustness property in finite samples
- Standard in modern causal inference (Chernozhukov et al. 2018)

**Reference**:
- Original: `src/models/DFPV/trainer.py` class `DFPVTrainer`
- Thesis: Algorithm 5.1 (complete algorithm)

---

### Phase 4: Integration & Testing

#### 4.1 Experiment Scripts

**New File**: `experiments/main_mar.py`

**Purpose**: Run Modified DFPV experiments

**Key Functions**:
- `run_mar_experiment()`: Single experiment with specified missingness rate
- Varies missingness rates: 0%, 30%, 50%, 70%
- Saves results and predictions

**Reference**: 
- Original: `main.py` in root
- Adapt for MAR data generation and modified trainer

---

**New File**: `experiments/compare_original_vs_mar.py`

**Purpose**: Compare Original DFPV (complete case) vs Modified DFPV (DR)

**Key Comparisons**:
- Vary missingness rates
- Plot MSE/MAE vs missingness
- Show efficiency gains of Modified DFPV

---

#### 4.2 Data Generation

**Extend**: Existing data generation to support MAR

**What to Add**:
- `generate_train_data_ate_mar()`: Generate data with missing $W$
- MAR mechanism: $\Pr(\delta_W = 0 \mid L^+) = \text{logistic}(\alpha_0 + \alpha^\top L^+)$
- Control missingness rate via $\alpha_0$

**Reference**:
- Existing: `src/data/ate/__init__.py`

---

### Phase 5: Testing Strategy

#### Unit Tests

1. **Test Nuisance Models**
   - Propensity predictions in $[0,1]$
   - Imputation reduces MSE on held-out complete cases
   - Correlation with true propensity (synthetic data)

2. **Test DR Pseudo-Outcome**
   - With perfect propensity → recovers true mean
   - With perfect imputation → recovers true mean
   - With both misspecified → no crash, reasonable output
   - Clipping behavior for extreme propensities

3. **Test Cross-Fitting**
   - Fold sizes balanced
   - No overlap between folds
   - Reconstructs full data when merged

4. **Test Data Structures**
   - `delta_W` correctly tracks missingness
   - Slicing preserves all fields
   - GPU transfer works

#### Integration Tests

1. **Modified vs Original (No Missing)**
   - Modified DFPV ≈ Original DFPV when missingness rate = 0%
   - Predictions differ by < 10%

2. **Varying Missingness Rates**
   - Test: 0%, 20%, 40%, 60%, 80% missing
   - Modified DFPV should:
     - Handle all rates without error
     - Outperform complete case at high missingness
     - Show reasonable MSE (<10 for demand dataset)


## ✅ Success Criteria

### Functional Requirements

- ✅ Code runs without errors on demand dataset
- ✅ Handles 0%, 30%, 50%, 70% missingness
- ✅ Produces reasonable predictions (MSE < 10)
- ✅ All unit tests pass
- ✅ Integration tests pass

### Performance Requirements

- ✅ Modified DFPV ≈ Original DFPV at 0% missing (difference < 10%)
- ✅ Modified DFPV < Complete Case at 50% missing 
- ✅ Training time < 2x original DFPV (cross-fitting overhead)

### Code Quality Requirements

- ✅ Minimal Implementation, only do what told
- ✅ Follows existing code style
- ✅ Clear variable names matching thesis notation
- ✅ No hardcoded magic numbers
- ✅ Modular, testable design
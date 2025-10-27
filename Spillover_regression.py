# ====================================================
# Robust regression workflow to handle singular design
# Variables assumed: npi_id, test_or_control (Target/Control),
# target_members, control_members, total_members,
# EDS_conversion_flag_150d, member_status, star_pdc_py, (optional) condition
# ====================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Example data (comment out when using real df)
# ---------------------------
# np.random.seed(0)
# df = pd.DataFrame({
#     'npi_id': np.random.randint(100, 200, 3000),
#     'test_or_control': np.random.choice(['Target','Control'], 3000, p=[0.5,0.5]),
#     'target_members': np.random.randint(1, 20, 3000),
#     'control_members': np.random.randint(1, 20, 3000),
#     'total_members': np.random.randint(5, 30, 3000),
#     'EDS_conversion_flag_150d': np.random.binomial(1, 0.35, 3000),
#     'member_status': np.random.choice(['new','returning'], 3000, p=[0.2,0.8]),
#     'star_pdc_py': np.random.choice([np.nan, 0.6, 0.7, 0.8, 0.9], 3000, p=[0.2,0.2,0.2,0.2,0.2])
# })
# df['total_members'] = df['target_members'] + df['control_members']  # realistic

# ---------------------------
# 1) Basic preprocessing & checks
# ---------------------------
required_cols = ['npi_id','test_or_control','target_members','control_members','total_members',
                 'EDS_conversion_flag_150d','member_status','star_pdc_py']

# If any required missing, stop early
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    raise ValueError(f"Missing required columns: {missing_req}")

# Create prescriber exposure share
df['prescriber_target_share'] = df['target_members'] / df['total_members']
df['prescriber_target_share'] = df['prescriber_target_share'].replace([np.inf, -np.inf], np.nan)

# Standardize target flag to numeric (0/1)
df['test_or_control_num'] = df['test_or_control'].map({'Control':0, 'Target':1})
if df['test_or_control_num'].isna().any():
    # fallback: try uppercase/lowercase mapping
    df['test_or_control_num'] = df['test_or_control'].str.lower().map({'control':0,'target':1})

# Standardize outcome to 0/1 ints
df['EDS_conversion_flag_150d'] = df['EDS_conversion_flag_150d'].map({True:1, False:0}).fillna(df['EDS_conversion_flag_150d'])
# If still not numeric, try coercion
if not np.issubdtype(df['EDS_conversion_flag_150d'].dtype, np.number):
    df['EDS_conversion_flag_150d'] = pd.to_numeric(df['EDS_conversion_flag_150d'], errors='coerce')

# Fill PDC missing with 0 (new members)
df['star_pdc_py_filled'] = df['star_pdc_py'].fillna(0)
df['is_returning'] = (df['member_status'] == 'returning').astype(int)

# Drop rows missing essential predictors or outcome
keep_cols = ['EDS_conversion_flag_150d','test_or_control_num','prescriber_target_share','star_pdc_py_filled','is_returning','npi_id']
df_clean = df.dropna(subset=keep_cols).copy()
print(f"Rows starting: {len(df)}, after dropping missing essential cols: {len(df_clean)}")
df = df_clean

# ---------------------------
# 2) Quick diagnostics: zero-variance and correlation
# ---------------------------
def zero_variance_cols(df, tolerance=1e-8):
    return [c for c in df.columns if df[c].nunique() <= 1]

numeric_for_vif = df[['test_or_control_num','prescriber_target_share','star_pdc_py_filled','is_returning']].copy()
zv = zero_variance_cols(numeric_for_vif)
if zv:
    print("Zero-variance columns detected (will drop):", zv)
    numeric_for_vif = numeric_for_vif.drop(columns=zv)

print("\nPredictor summaries:")
print(numeric_for_vif.describe().T)

# ---------------------------
# 3) VIF check
# ---------------------------
# Add constant for VIF computations
X = sm.add_constant(numeric_for_vif)
vif = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\nVIFs:")
print(vif)

# If VIFs are huge (>10), we'll try to drop the most collinear variables iteratively
high_vif = vif[vif['VIF'] > 10]['variable'].tolist()
high_vif = [v for v in high_vif if v != 'const']
if high_vif:
    print("\nHigh VIF variables detected:", high_vif)
    # Strategy: drop in order of highest VIF until all VIF <= 10
    drop_list = []
    X_vif = X.copy()
    while True:
        vif_vals = pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])], index=X_vif.columns)
        vif_nonconst = vif_vals.drop('const')
        if vif_nonconst.max() <= 10 or vif_nonconst.shape[0] <= 1:
            break
        to_drop = vif_nonconst.idxmax()
        print(f"Dropping {to_drop} (VIF={vif_nonconst.max():.2f}) to reduce multicollinearity.")
        drop_list.append(to_drop)
        X_vif = X_vif.drop(columns=[to_drop])
    numeric_for_vif = numeric_for_vif.drop(columns=drop_list)
    print("Dropped due to VIF:", drop_list)
    print("Remaining predictors for regression:", numeric_for_vif.columns.tolist())

# ---------------------------
# 4) Build formula (no redundant columns)
# ---------------------------
# use test_or_control_num (0/1) not a full set of dummies to avoid collinearity
# include optional condition as categorical if present but drop if causes collinearity
condition_term = " + C(condition)" if 'condition' in df.columns else ""
formula = f"""
EDS_conversion_flag_150d ~ test_or_control_num
                         + prescriber_target_share
                         + test_or_control_num:prescriber_target_share
                         + star_pdc_py_filled
                         + is_returning
                         {condition_term}
"""
formula = " ".join(formula.split())  # tidy spacing
print("\nModel formula:\n", formula)

# ---------------------------
# 5) Check design matrix rank to pre-empt SVD errors
# ---------------------------
# Create design matrix via patsy to inspect rank
import patsy
y, X_design = patsy.dmatrices(formula, data=df, return_type='dataframe')
rank = np.linalg.matrix_rank(X_design.values)
cols = X_design.shape[1]
print(f"\nDesign matrix rank: {rank} | columns: {cols}")
if rank < cols:
    # find linearly dependent columns via QR
    print("Design matrix is rank-deficient (singular). Attempting to identify dependent columns...")
    # compute SVD and tiny singular values
    u, s, vh = np.linalg.svd(X_design.values, full_matrices=False)
    tol = 1e-10
    near_zero_idx = np.where(s < tol)[0]
    print("Number of near-zero singular values:", len(near_zero_idx))
    # Show candidate columns (warning: approximate)
    # We'll try dropping interaction term as first remedy
    if 'test_or_control_num:prescriber_target_share' in X_design.columns:
        print("Dropping interaction term 'test_or_control_num:prescriber_target_share' to fix singularity.")
        formula = formula.replace("+ test_or_control_num:prescriber_target_share", "")
        y, X_design = patsy.dmatrices(formula, data=df, return_type='dataframe')
        rank = np.linalg.matrix_rank(X_design.values)
        print(f"New design matrix rank: {rank} | columns: {X_design.shape[1]}")
        if rank < X_design.shape[1]:
            print("Still singular after dropping interaction; will use pseudoinverse in OLS and attempt logit regularized.")
    else:
        print("Interaction term not present; consider simplifying categorical encodings or dropping a condition level.")

# ---------------------------
# 6) Fit OLS robustly (try pinv if singular)
# ---------------------------
ols_model = None
try:
    ols_model = smf.ols(formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['npi_id']}
    )
    print("\nOLS converged normally.")
except Exception as e:
    print("\nOLS standard fit failed with error:", e)
    print("Attempting OLS with pseudoinverse (pinv) to handle singular matrix.")
    try:
        ols_model = smf.ols(formula, data=df).fit(method='pinv')
        # re-calc clustered cov
        ols_model = ols_model.get_robustcov_results(cov_type='cluster', groups=df['npi_id'])
        print("OLS using pinv succeeded (results have cluster-robust SEs applied).")
    except Exception as e2:
        print("OLS even with pinv failed:", e2)
        ols_model = None

# ---------------------------
# 7) Fit Logit robustly (try solver/regularized)
# ---------------------------
logit_result = None
try:
    # first ensure binary outcome is strict 0/1
    df['EDS_conversion_flag_150d'] = df['EDS_conversion_flag_150d'].astype(int)
    logit_model = smf.logit(formula, data=df)
    logit_result = logit_model.fit(method='bfgs', maxiter=500, disp=False)
    # apply cluster-robust cov
    logit_cluster = logit_result.get_robustcov_results(cov_type='cluster', groups=df['npi_id'])
    logit_result = logit_cluster
    print("\nLogit converged with BFGS and cluster-robust SEs applied.")
except Exception as e:
    print("\nLogit initial fit failed with error:", e)
    print("Attempting penalized logistic regression (L2) to handle separation/collinearity.")
    try:
        # Build X and y for sklearn-like regularized fit via statsmodels fit_regularized
        logit_model = smf.logit(formula, data=df)
        # alpha small -> mild regularization; increase if still unstable
        logit_regularized = logit_model.fit_regularized(method='l1', alpha=0.01, maxiter=1000)
        # fit_regularized returns params but not straightforward robust cov - keep params only
        logit_result = logit_regularized
        print("Penalized Logit succeeded (note: p-values not available for penalized fit).")
    except Exception as e2:
        print("Penalized Logit also failed:", e2)
        logit_result = None

# ---------------------------
# 8) Build summary table for key effects
# ---------------------------
effects = ['Direct Campaign Effect', 'Spillover Effect', 'Amplification (Interaction)']
vars_list = ['test_or_control_num', 'prescriber_target_share', 'test_or_control_num:prescriber_target_share']

summary_rows = []
for eff, var in zip(effects, vars_list):
    row = {'Effect': eff, 'Variable': var}
    # OLS
    if ols_model is not None:
        beta_ols = ols_model.params.get(var, np.nan)
        p_ols = ols_model.pvalues.get(var, np.nan) if hasattr(ols_model, 'pvalues') else np.nan
    else:
        beta_ols = p_ols = np.nan
    row['Beta (OLS)'] = beta_ols
    row['p-value (OLS)'] = p_ols
    # Logit
    if logit_result is not None:
        try:
            beta_logit = logit_result.params.get(var, np.nan)
            p_logit = logit_result.pvalues.get(var, np.nan)
        except Exception:
            # penalized fit may not have pvalues
            beta_logit = getattr(logit_result, 'params', pd.Series()).get(var, np.nan)
            p_logit = np.nan
    else:
        beta_logit = p_logit = np.nan
    row['Beta (Logit)'] = beta_logit
    row['p-value (Logit)'] = p_logit

    # Interpretation
    sig = "Significant" if (not np.isnan(p_ols) and p_ols < 0.05) else "Not significant/unknown"
    if eff == 'Direct Campaign Effect':
        interp = f"Direct effect of being targeted ({sig})"
    elif eff == 'Spillover Effect':
        interp = f"Effect of prescriber target share on members ({sig})"
    else:
        interp = f"Interaction effect (Target x PrescriberShare) ({sig})"
    row['Interpretation'] = interp
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.round(4)
print("\nSummary table (key effects):")
print(summary_df)

# ---------------------------
# 9) Marginal effects plot (OLS-based predictions, fallback even if OLS used pinv)
# ---------------------------
if ols_model is not None:
    pred_df = df.copy()
    pred_df['pred_ols'] = ols_model.predict(pred_df)
    # Create smoothed lines for target/control groups
    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=pred_df,
        x='prescriber_target_share',
        y='pred_ols',
        hue='test_or_control',
        ci=95,
        estimator='mean',
        lw=2
    )
    plt.title("Predicted EDS conversion (OLS) by Prescriber Target Share\n(Controls vs Target)")
    plt.xlabel("Prescriber Target Share")
    plt.ylabel("Predicted EDS conversion (probability)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping marginal effects plot since OLS model not available.")

# ---------------------------
# 10) Final notes printed to user
# ---------------------------
print("\n---- Diagnostic Notes ----")
print("- If the interaction term was dropped automatically above, re-run with it separately on a subset if you need it.")
print("- If VIF-based dropping removed predictors, review which were dropped and whether they are theoretically important.")
print("- If logit had to be penalized, p-values are not directly available; consider reporting OLS effect sizes with cluster-robust SEs and logistic coefficients as sensitivity.")
print("- For publication-grade inference, consider a mixed-effects model with prescriber random intercept if many prescribers and enough per-prescriber observations.")

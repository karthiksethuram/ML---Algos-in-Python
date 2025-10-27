# Requirements:
# pip install pandas numpy statsmodels scikit-learn matplotlib patsy

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.preprocessing import OneHotEncoder

# -----------------------
# 1) INPUT: your dataframe
# -----------------------
# df = pd.read_csv("your_data.csv")  # <- load your data
# Expected columns: npi_id, target_or_control (Target/Control), target_members, control_members,
# total_members, EDS_conversion_flag_150d (0/1), star_pdc_py (0-1 or NaN), member_status ('New'/'Returning'),
# condition_list (could be e.g. "DIAB;STATIN" or ["DIAB","STATIN"]) OR separate flags already.

# For demonstration I'll assume df already exists.

# -----------------------
# 2) Feature engineering
# -----------------------
df = df.copy()

# Ensure binary outcome is numeric 0/1
df['EDS_conv'] = df['EDS_conversion_flag_150d'].astype(int)

# Clean assignment
df['is_target_member'] = df['target_or_control'].astype(str).str.lower().map({'target':1,'t':1,'1':1,'control':0,'c':0,'0':0})
# fall back if mapping fails:
df['is_target_member'] = df['is_target_member'].fillna(df['target_or_control'].apply(lambda x: 1 if str(x).lower() in ['target','t','1'] else 0))

# Prescriber-level exposure: fraction of that prescriber's members that were randomized to target
# If you already have a prescriber-level summary, use it; otherwise compute:
# Some rows may already have target_members/total_members. We'll create a robust share:
df['total_members'] = df['total_members'].replace(0, np.nan)  # avoid zero division
df['presc_target_share'] = df['target_members'] / df['total_members']
# If target_members / total_members not provided, compute at npi_id level:
if df['presc_target_share'].isnull().all():
    grp = df.groupby('npi_id')['is_target_member'].agg(['mean','count']).rename(columns={'mean':'presc_target_share','count':'presc_member_count'})
    df = df.merge(grp['presc_target_share'], left_on='npi_id', right_index=True, suffixes=('','_computed'))
    df['presc_target_share'] = df['presc_target_share'].fillna(df['presc_target_share_computed'])

# indicator whether prescriber was targeted at all (useful for simple spillover)
df['presc_any_target'] = (df['presc_target_share'] > 0).astype(int)

# Member status dummy
df['is_new_member'] = (df['member_status'].astype(str).str.lower() == 'new').astype(int)

# Handle star_pdc_py: flag missingness and impute zeros for new members (or use mean)
df['star_pdc_py_missing'] = df['star_pdc_py'].isnull().astype(int)
# Impute with 0 for missing (because new members have no prior adherence) – we also keep the missing flag.
df['star_pdc_py_imputed'] = df['star_pdc_py'].fillna(0.0)

# Condition flags: adapt depending on how conditions are stored
# If they are a delimited string like "DIAB;STATIN"
def make_condition_flags(series):
    # returns DataFrame with columns DIAB, ACE_ARB, STATIN binary
    conds = {'DIAB':[], 'ACE_ARB':[], 'STATIN':[]}
    for val in series.fillna(''):
        tokens = [t.strip().upper() for t in (val if isinstance(val, list) else str(val)).replace(',', ';').split(';') if t.strip()]
        conds['DIAB'].append(int('DIAB' in tokens))
        conds['ACE_ARB'].append(int('ACE_ARB' in tokens or 'ACE-ARB' in tokens))
        conds['STATIN'].append(int('STATIN' in tokens))
    return pd.DataFrame(conds, index=series.index)

if 'DIAB' not in df.columns and 'condition_list' in df.columns:
    cond_df = make_condition_flags(df['condition_list'])
    df = pd.concat([df, cond_df], axis=1)
else:
    # If separate flags already exist, ensure they are binary 0/1
    for c in ['DIAB','ACE_ARB','STATIN']:
        if c in df.columns:
            df[c] = df[c].astype(int)

# If members can be targeted for more than one condition, these are non-exclusive flags (ok).

# Interaction: target member * prescriber target share to detect spillover on controls
df['target_x_presc_share'] = df['is_target_member'] * df['presc_target_share']

# -----------------------
# 3) Prepare regression dataframe & drop bad rows
# -----------------------
# Keep necessary cols
model_df = df[[
    'npi_id','EDS_conv','is_target_member','presc_target_share','presc_any_target','target_x_presc_share',
    'star_pdc_py_imputed','star_pdc_py_missing','is_new_member','DIAB','ACE_ARB','STATIN'
]].copy()

# Drop rows with missing outcome or missing npi_id
model_df = model_df.dropna(subset=['EDS_conv','npi_id'])

# -----------------------
# 4) Multicollinearity check (VIF) and automatic dropping
# -----------------------
def calculate_vif(X_df):
    X = X_df.copy()
    X = X.astype(float)
    X['intercept'] = 1.0
    vif_df = pd.DataFrame()
    vif_df['variable'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df

# Choose candidate regressors (no intercept)
candidate_vars = [
    'is_target_member','presc_target_share','target_x_presc_share',
    'star_pdc_py_imputed','star_pdc_py_missing','is_new_member',
    'DIAB','ACE_ARB','STATIN'
]
X_for_vif = model_df[candidate_vars].fillna(0.0)
vif = calculate_vif(X_for_vif)

# Drop any variable (except is_target_member and presc_target_share) with VIF > 10 (heuristic).
# We preserve main vars is_target_member and presc_target_share and their interaction if possible.
to_drop = []
for _, row in vif.iterrows():
    var, v = row['variable'], row['VIF']
    if var in ['is_target_member','presc_target_share','target_x_presc_share']: 
        continue
    if v > 10:
        to_drop.append(var)

if to_drop:
    print("Dropping high-VIF variables to avoid singularity:", to_drop)
    candidate_vars = [v for v in candidate_vars if v not in to_drop]

# Final regressors
regressors = candidate_vars.copy()

# Build design matrices (add constant)
X = sm.add_constant(model_df[regressors], has_constant='add')
y = model_df['EDS_conv']

# Check matrix rank to avoid singularities
rank = np.linalg.matrix_rank(X.values)
cols = X.columns.tolist()
if rank < X.shape[1]:
    # Drop columns causing rank deficiency: use QR or drop near-constant columns
    print(f"Design matrix rank deficient (rank {rank} < cols {X.shape[1]}). Attempting to drop collinear columns.")
    # heuristic: drop one condition if present
    for col in ['STATIN','ACE_ARB','DIAB','star_pdc_py_imputed','star_pdc_py_missing','is_new_member']:
        if col in X.columns and X.shape[1] - 1 >= rank:
            X = X.drop(columns=[col])
            regressors.remove(col)
            print("Dropped", col)
            rank = np.linalg.matrix_rank(X.values)
            if rank == X.shape[1]:
                break
    if rank < X.shape[1]:
        raise np.linalg.LinAlgError("Unable to fix singular design matrix automatically. Check predictors for perfect multicollinearity.")

# -----------------------
# 5) Fit OLS Linear Probability Model with prescriber-clustered SE
# -----------------------
ols_model = sm.OLS(y, X)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': model_df['npi_id']})
print(ols_res.summary())

# -----------------------
# 6) Fit Logistic (Logit) with clustered robust SE
# -----------------------
# Statsmodels Logit may fail to converge in some extreme separation cases; we catch exceptions.
logit_model = sm.Logit(y, X)

try:
    logit_res = logit_model.fit(disp=False, method='newton', maxiter=100)
except Exception as e:
    print("Logit failed to converge using default solver:", e)
    # Try another solver
    logit_res = logit_model.fit(disp=False, method='bfgs', maxiter=200)

# Compute cluster-robust covariance (clusters by npi_id)
try:
    logit_clus = logit_res.get_robustcov_results(cov_type='cluster', groups=model_df['npi_id'])
except Exception as e:
    print("Failed to compute clustered cov for Logit; falling back to sandwich (HC1):", e)
    logit_clus = logit_res.get_robustcov_results(cov_type='HC1')

print(logit_clus.summary())

# -----------------------
# 7) Marginal effects: effect of presc_target_share (spillover) by member assignment
# We'll compute predicted probabilities across presc_target_share grid for both target & control members,
# holding continuous covariates at their means and categorical at reference.
# -----------------------
# Find if presc_target_share and is_target_member present
if 'presc_target_share' in X.columns and 'is_target_member' in X.columns:
    grid = np.linspace(0, 1, 50)
    # Base vector: set all regressors to their column means
    base = X.drop(columns=['const']).mean()
    # Create DataFrame to hold predictions
    preds = []
    for t in [0,1]:
        for s in grid:
            row = base.copy()
            # set presc_target_share and is_target_member
            if 'presc_target_share' in row.index:
                row['presc_target_share'] = s
            if 'is_target_member' in row.index:
                row['is_target_member'] = t
            # set interaction if included
            if 'target_x_presc_share' in row.index:
                row['target_x_presc_share'] = t * s
            # Ensure column order
            row_with_const = sm.add_constant(pd.DataFrame([row]), has_constant='add')
            # Reindex to X columns
            row_with_const = row_with_const.reindex(columns=X.columns).fillna(0)
            # Predict using logit clustered result (use params from logit_res)
            linpred = np.dot(row_with_const.values, logit_res.params.values)
            prob = 1 / (1 + np.exp(-linpred))
            preds.append({'is_target_member':t, 'presc_target_share':s, 'pred_prob':prob[0]})
    pred_df = pd.DataFrame(preds)

    # Plot
    plt.figure(figsize=(8,5))
    for t,label in zip([0,1],['Control member','Target member']):
        sub = pred_df[pred_df['is_target_member']==t]
        plt.plot(sub['presc_target_share'], sub['pred_prob'], label=label)
    plt.xlabel('Prescriber target share (proportion of members randomized to target)')
    plt.ylabel('Predicted probability EDS conversion (logit model)')
    plt.title('Marginal effect / predicted probability across prescriber target share')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("presc_target_share or is_target_member not available in model; skipping marginal effects plot.")

# -----------------------
# 8) Build combined summary table with interpretations
# -----------------------
def summarize_results(res_obj, clustered=True, cluster_groups=None, label='Model'):
    params = res_obj.params
    bse = res_obj.bse
    tstat = params / bse
    pvals = res_obj.pvalues
    summary = pd.DataFrame({
        'coef': params,
        'std_err': bse,
        't_or_z': tstat,
        'pval': pvals
    })
    summary['model'] = label
    return summary

ols_sum = summarize_results(ols_res, label='OLS_LPM_clustered')
logit_sum = summarize_results(logit_clus, label='Logit_clustered')

combined = pd.concat([ols_sum, logit_sum], axis=0).reset_index().rename(columns={'index':'term'})

# Simple interpretation helper (focus on three terms)
def interpret_row(row):
    t = row['term']
    coef = row['coef']
    pv = row['pval']
    if 'is_target_member' in t:
        return f"{'Higher' if coef>0 else 'Lower'} EDS conversion for members randomized to target vs control by {abs(coef):.3f} (LPM probability points) — p={pv:.3g}"
    if 'presc_target_share' in t and 'target_x' not in t:
        return f"Spillover: prescribers with higher share of targeted members assoc. with {'higher' if coef>0 else 'lower'} conversion by {abs(coef):.3f} (per 1.0 increase) — p={pv:.3g}"
    if 'target_x_presc_share' in t:
        return f"Interaction: being a target member when prescriber has more targeted members changes conversion by {coef:.3f} — p={pv:.3g}"
    if 'star_pdc_py_imputed' in t:
        return f"Prior-year adherence assoc. change: coef={coef:.3f} — p={pv:.3g}"
    return ""

combined['interpretation'] = combined.apply(interpret_row, axis=1)

# Show table
pd.set_option('display.max_rows', 200)
print(combined[['model','term','coef','std_err','t_or_z','pval','interpretation']])

# -----------------------
# 9) Notes to user printed out
# -----------------------
notes = """
Notes & Recommendations:
1. We used prescriber-level 'presc_target_share' (target_members / total_members) to identify potential spillover: control members who visited prescribers
   with high presc_target_share are plausibly exposed to prescriber-level treatment.
2. The key coefficients to inspect:
   - is_target_member: direct randomized effect on member assigned to target
   - presc_target_share: association of prescriber-level share with member outcomes (spillover)
   - target_x_presc_share: whether the direct effect changes when prescriber has more targeted members (interaction)
3. We used clustered standard errors at prescriber (npi_id) to account for correlation of members within prescribers.
4. For new members we imputed star_pdc_py=0 and added a missingness flag. If you prefer other imputation (mean, median), change the imputation logic.
5. If Logit experiences perfect separation or convergence issues, consider penalized logistic (e.g., sklearn's LogisticRegression with C and l2) — but that will not give correct cluster-robust SE easily.
6. Interpret logit coefficients via marginal effects or predicted probabilities (we plotted predicted probabilities across presc_target_share).
"""
print(notes)

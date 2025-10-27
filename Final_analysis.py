import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------
# 1. Load and preprocess
# ------------------------------------
# df = pd.read_csv("your_data.csv")

df = df.copy()

# Ensure clean column names
df.columns = df.columns.str.strip()

# --- Binary outcome ---
df['EDS_conv'] = df['EDS_conversion_flag_150d'].astype(int)

# --- Target assignment ---
df['is_target_member'] = (
    df['target_or_control'].astype(str).str.lower().map({
        'target': 1, 't': 1, '1': 1,
        'control': 0, 'c': 0, '0': 0
    }).fillna(0)
).astype(int)

# --- Prescriber target share ---
df['total_members'] = pd.to_numeric(df['total_members'], errors='coerce')
df['target_members'] = pd.to_numeric(df['target_members'], errors='coerce')

df['presc_target_share'] = df['target_members'] / df['total_members']

if df['presc_target_share'].isnull().all():
    grp = df.groupby('npi_id')['is_target_member'].mean().rename('presc_target_share')
    df = df.merge(grp, on='npi_id', how='left')

df['presc_any_target'] = (df['presc_target_share'] > 0).astype(int)

# --- Member status ---
df['is_new_member'] = (df['member_status'].astype(str).str.lower() == 'new').astype(int)

# --- Prior adherence ---
df['star_pdc_py'] = pd.to_numeric(df['star_pdc_py'], errors='coerce')
df['star_pdc_py_missing'] = df['star_pdc_py'].isnull().astype(int)
df['star_pdc_py_imputed'] = df['star_pdc_py'].fillna(0.0)

# --- Interaction term ---
df['target_x_presc_share'] = df['is_target_member'] * df['presc_target_share']

# --- Drug class dummies ---
df['star_drug_cls'] = df['star_drug_cls'].astype(str).str.upper().str.strip()
drug_dummies = pd.get_dummies(df['star_drug_cls'], prefix='cls', dtype=int)
if 'cls_STATIN' in drug_dummies.columns:
    drug_dummies = drug_dummies.drop(columns=['cls_STATIN'])  # baseline

df = pd.concat([df, drug_dummies], axis=1)

# ------------------------------------
# 2. Modeling dataset
# ------------------------------------
model_df = df[[
    'npi_id', 'EDS_conv', 'is_target_member', 'presc_target_share', 'target_x_presc_share',
    'star_pdc_py_imputed', 'star_pdc_py_missing', 'is_new_member'
] + list(drug_dummies.columns)].dropna(subset=['npi_id', 'EDS_conv'])

# Force numeric
for col in model_df.columns:
    if col not in ['npi_id']:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

# Drop any remaining non-numeric or all-NaN columns
non_numeric = model_df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Dropping non-numeric columns:", non_numeric)
    model_df = model_df.drop(columns=non_numeric)

# Replace inf and NaN with 0 (safe default)
model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# ------------------------------------
# 3. VIF check (optional)
# ------------------------------------
X_for_vif = sm.add_constant(model_df.drop(columns=['npi_id', 'EDS_conv']))
vif_df = pd.DataFrame({
    'variable': X_for_vif.columns,
    'VIF': [variance_inflation_factor(X_for_vif.values, i)
            for i in range(X_for_vif.shape[1])]
})
print("VIF summary:\n", vif_df)

# ------------------------------------
# 4. Fit OLS (Linear Probability Model)
# ------------------------------------
X = sm.add_constant(model_df.drop(columns=['npi_id', 'EDS_conv']), has_constant='add').astype(float)
y = model_df['EDS_conv'].astype(float)

ols_model = sm.OLS(y, X)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': model_df['npi_id']})
print("\nOLS Results Summary:")
print(ols_res.summary())

# ------------------------------------
# 5. Fit Logistic Regression (Logit)
# ------------------------------------
X_logit = X.copy()
try:
    logit_model = sm.Logit(y, X_logit)
    logit_res = logit_model.fit(disp=False, maxiter=200)
    logit_clus = logit_res.get_robustcov_results(cov_type='cluster', groups=model_df['npi_id'])
except Exception as e:
    print("⚠️ Logit convergence issue, trying regularized fit:", e)
    logit_res = sm.Logit(y, X_logit).fit_regularized(alpha=0.1, disp=False)
    logit_clus = logit_res.get_robustcov_results(cov_type='HC1')

print("\nLogit Results Summary:")
print(logit_clus.summary())

# ------------------------------------
# 6. Marginal effects plot
# ------------------------------------
if 'presc_target_share' in X.columns:
    grid = np.linspace(0, 1, 50)
    base = X.mean()
    preds = []
    for t in [0, 1]:
        for s in grid:
            row = base.copy()
            row['is_target_member'] = t
            row['presc_target_share'] = s
            if 'target_x_presc_share' in row.index:
                row['target_x_presc_share'] = t * s
            pred_prob = 1 / (1 + np.exp(-np.dot(row, logit_res.params)))
            preds.append({'is_target_member': t, 'presc_target_share': s, 'pred_prob': pred_prob})
    pred_df = pd.DataFrame(preds)
    plt.figure(figsize=(8, 5))
    for t, label in zip([0, 1], ['Control', 'Target']):
        sub = pred_df[pred_df['is_target_member'] == t]
        plt.plot(sub['presc_target_share'], sub['pred_prob'], label=label)
    plt.xlabel("Prescriber Target Share")
    plt.ylabel("Predicted EDS Conversion Probability")
    plt.title("Marginal Effect of Prescriber Target Share")
    plt.legend()
    plt.grid(True)
    plt.show()

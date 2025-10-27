# Requirements:
# pip install pandas numpy statsmodels scikit-learn matplotlib patsy

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------
# 1) INPUT: your dataframe
# -----------------------
# df = pd.read_csv("your_data.csv")

# Expected columns:
# npi_id, target_or_control, target_members, control_members, total_members,
# EDS_conversion_flag_150d, star_pdc_py, member_status, star_drug_cls (DIAB/ACE_ARB/STATIN)

df = df.copy()

# -----------------------
# 2) Feature engineering
# -----------------------

# Binary outcome
df['EDS_conv'] = df['EDS_conversion_flag_150d'].astype(int)

# Target vs Control
df['is_target_member'] = df['target_or_control'].astype(str).str.lower().map({
    'target': 1, 't': 1, '1': 1,
    'control': 0, 'c': 0, '0': 0
}).fillna(0)

# Prescriber-level target share
df['total_members'] = df['total_members'].replace(0, np.nan)
df['presc_target_share'] = df['target_members'] / df['total_members']

if df['presc_target_share'].isnull().all():
    grp = df.groupby('npi_id')['is_target_member'].mean().rename('presc_target_share')
    df = df.merge(grp, on='npi_id', how='left')

df['presc_any_target'] = (df['presc_target_share'] > 0).astype(int)

# Member status dummy
df['is_new_member'] = (df['member_status'].astype(str).str.lower() == 'new').astype(int)

# Handle star_pdc_py missing values (0 for new, plus missingness flag)
df['star_pdc_py_missing'] = df['star_pdc_py'].isnull().astype(int)
df['star_pdc_py_imputed'] = df['star_pdc_py'].fillna(0.0)

# Interaction term for spillover
df['target_x_presc_share'] = df['is_target_member'] * df['presc_target_share']

# Drug class dummy variables
drug_dummies = pd.get_dummies(df['star_drug_cls'], prefix='cls')
# Drop one baseline to avoid dummy trap (STATIN as baseline)
if 'cls_STATIN' in drug_dummies.columns:
    drug_dummies = drug_dummies.drop(columns=['cls_STATIN'])
df = pd.concat([df, drug_dummies], axis=1)

# -----------------------
# 3) Model setup
# -----------------------
model_df = df[[
    'npi_id', 'EDS_conv', 'is_target_member', 'presc_target_share', 'target_x_presc_share',
    'star_pdc_py_imputed', 'star_pdc_py_missing', 'is_new_member'
] + list(drug_dummies.columns)].dropna(subset=['EDS_conv', 'npi_id'])

# -----------------------
# 4) Multicollinearity check
# -----------------------
X_for_vif = model_df.drop(columns=['npi_id', 'EDS_conv']).fillna(0.0)
X_for_vif = sm.add_constant(X_for_vif)
vif_df = pd.DataFrame({
    'variable': X_for_vif.columns,
    'VIF': [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])]
})
print("VIF summary:\n", vif_df)

# Drop high-VIF vars if any (optional)
drop_vars = vif_df.loc[vif_df['VIF'] > 10, 'variable']
if len(drop_vars) > 0:
    print("Dropping high-VIF variables:", drop_vars.tolist())
    model_df = model_df.drop(columns=[v for v in drop_vars if v != 'const'], errors='ignore')

# -----------------------
# 5) Fit OLS (Linear Probability Model)
# -----------------------
X = sm.add_constant(model_df.drop(columns=['npi_id', 'EDS_conv']), has_constant='add')
y = model_df['EDS_conv']

ols_model = sm.OLS(y, X)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': model_df['npi_id']})
print(ols_res.summary())

# -----------------------
# 6) Fit Logistic Regression (Logit)
# -----------------------
logit_model = sm.Logit(y, X)
try:
    logit_res = logit_model.fit(disp=False, maxiter=200)
except Exception as e:
    print("Logit convergence issue:", e)
    logit_res = logit_model.fit_regularized(alpha=0.1, disp=False)

# Clustered SE for Logit
try:
    logit_clus = logit_res.get_robustcov_results(cov_type='cluster', groups=model_df['npi_id'])
except:
    logit_clus = logit_res.get_robustcov_results(cov_type='HC1')

print(logit_clus.summary())

# -----------------------
# 7) Marginal effects plot for presc_target_share
# -----------------------
if 'presc_target_share' in X.columns:
    grid = np.linspace(0, 1, 50)
    base = X.drop(columns=['const']).mean()
    preds = []
    for t in [0, 1]:
        for s in grid:
            row = base.copy()
            row['presc_target_share'] = s
            row['is_target_member'] = t
            if 'target_x_presc_share' in row.index:
                row['target_x_presc_share'] = t * s
            row_df = sm.add_constant(pd.DataFrame([row]), has_constant='add')
            pred = 1 / (1 + np.exp(-np.dot(row_df, logit_res.params)))
            preds.append({'is_target_member': t, 'presc_target_share': s, 'pred_prob': pred[0, 0]})
    pred_df = pd.DataFrame(preds)

    plt.figure(figsize=(8, 5))
    for t, label in zip([0, 1], ['Control member', 'Target member']):
        sub = pred_df[pred_df['is_target_member'] == t]
        plt.plot(sub['presc_target_share'], sub['pred_prob'], label=label)
    plt.xlabel('Prescriber target share')
    plt.ylabel('Predicted EDS conversion probability')
    plt.title('Marginal effects by prescriber target share')
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------
# 8) Combine results for interpretation
# -----------------------
def summarize_results(res_obj, model_label):
    return pd.DataFrame({
        'term': res_obj.params.index,
        'coef': res_obj.params.values,
        'std_err': res_obj.bse.values,
        't_or_z': res_obj.tvalues if hasattr(res_obj, 'tvalues') else res_obj.zvalues,
        'pval': res_obj.pvalues,
        'model': model_label
    })

summary = pd.concat([
    summarize_results(ols_res, 'OLS_LPM_clustered'),
    summarize_results(logit_clus, 'Logit_clustered')
])

def interpret(term, coef, pval):
    if 'is_target_member' in term:
        return f"{'Higher' if coef > 0 else 'Lower'} EDS conversion for target vs control members ({coef:.3f}, p={pval:.3g})"
    elif 'presc_target_share' == term:
        return f"Evidence of spillover: prescribers with more targeted patients have {'higher' if coef > 0 else 'lower'} EDS ({coef:.3f}, p={pval:.3g})"
    elif 'target_x_presc_share' in term:
        return f"Interaction: treatment effect depends on prescriber exposure ({coef:.3f}, p={pval:.3g})"
    elif term.startswith('cls_'):
        drug = term.split('_')[1]
        return f"Compared to STATIN (ref), {drug} class members show {'higher' if coef > 0 else 'lower'} EDS conversion ({coef:.3f}, p={pval:.3g})"
    return ""

summary['interpretation'] = summary.apply(lambda r: interpret(r['term'], r['coef'], r['pval']), axis=1)
print(summary[['model', 'term', 'coef', 'std_err', 'pval', 'interpretation']])

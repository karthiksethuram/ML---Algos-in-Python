# ==========================================
# EDS Conversion Spillover Regression + Visualization
# ==========================================

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1️⃣ Input Dataset
# ==========================================
# Expected columns:
# npi_id, test_or_control, target_members, control_members, total_members,
# EDS_conversion_flag_150d, member_status, star_pdc_py
# Optional: 'condition' (DIAB / ACE_ARB / STATIN)

# Example structure (for testing — comment out when using real data)
# df = pd.DataFrame({
#     'npi_id': np.random.randint(100, 200, 5000),
#     'test_or_control': np.random.binomial(1, 0.5, 5000),
#     'target_members': np.random.randint(1, 20, 5000),
#     'control_members': np.random.randint(1, 20, 5000),
#     'total_members': np.random.randint(5, 30, 5000),
#     'EDS_conversion_flag_150d': np.random.binomial(1, 0.4, 5000),
#     'member_status': np.random.choice(['new','returning'], 5000),
#     'star_pdc_py': np.random.choice([np.nan, 0.6, 0.7, 0.8, 0.9], 5000),
#     'condition': np.random.choice(['DIAB','ACE_ARB','STATIN'], 5000)
# })

# ==========================================
# 2️⃣ Data Preparation
# ==========================================
# Compute prescriber exposure
df['prescriber_target_share'] = df['target_members'] / df['total_members']
df['prescriber_target_share'] = df['prescriber_target_share'].clip(0, 1)

# Handle missing prior PDC and returning/new
df['star_pdc_py_filled'] = df['star_pdc_py'].fillna(0)
df['is_returning'] = (df['member_status'] == 'returning').astype(int)

# ==========================================
# 3️⃣ Model Specification
# ==========================================
condition_term = " + C(condition)" if "condition" in df.columns else ""

formula = f"""
EDS_conversion_flag_150d ~ test_or_control
                         + prescriber_target_share
                         + test_or_control:prescriber_target_share
                         + star_pdc_py_filled * is_returning
                         + is_returning
                         {condition_term}
"""

# ==========================================
# 4️⃣ OLS Regression (interpretable % point effects)
# ==========================================
ols_model = smf.ols(formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['npi_id']}
)
print("\nOLS Results (Clustered by Prescriber NPI):")
print(ols_model.summary())

# ==========================================
# 5️⃣ Logistic Regression (binary probability model)
# ==========================================
try:
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    logit_cluster = logit_model.get_robustcov_results(
        cov_type='cluster', groups=df['npi_id']
    )
    print("\nLogistic Results (Clustered by Prescriber NPI):")
    print(logit_cluster.summary())
except Exception as e:
    print("\n[Logit model skipped — likely due to convergence or non-binary outcome]")
    logit_cluster = None

# ==========================================
# 6️⃣ Summary Table (OLS + Logit)
# ==========================================
summary_data = {
    'Effect': ['Direct Campaign Effect', 'Spillover Effect', 'Amplification (Interaction)'],
    'Variable': ['test_or_control', 'prescriber_target_share', 'test_or_control:prescriber_target_share'],
    'Beta (OLS)': [
        ols_model.params.get('test_or_control', np.nan),
        ols_model.params.get('prescriber_target_share', np.nan),
        ols_model.params.get('test_or_control:prescriber_target_share', np.nan)
    ],
    'p-value (OLS)': [
        ols_model.pvalues.get('test_or_control', np.nan),
        ols_model.pvalues.get('prescriber_target_share', np.nan),
        ols_model.pvalues.get('test_or_control:prescriber_target_share', np.nan)
    ]
}

if logit_cluster is not None:
    summary_data['Beta (Logit)'] = [
        logit_cluster.params.get('test_or_control', np.nan),
        logit_cluster.params.get('prescriber_target_share', np.nan),
        logit_cluster.params.get('test_or_control:prescriber_target_share', np.nan)
    ]
    summary_data['p-value (Logit)'] = [
        logit_cluster.pvalues.get('test_or_control', np.nan),
        logit_cluster.pvalues.get('prescriber_target_share', np.nan),
        logit_cluster.pvalues.get('test_or_control:prescriber_target_share', np.nan)
    ]

summary_df = pd.DataFrame(summary_data)

# Add qualitative interpretation
def interpret(row):
    sig = "Significant" if row['p-value (OLS)'] < 0.05 else "Not significant"
    if row['Effect'] == 'Direct Campaign Effect':
        return f"Direct impact of member being targeted ({sig})."
    elif row['Effect'] == 'Spillover Effect':
        return f"Indirect effect of prescriber exposure on all members ({sig})."
    else:
        return f"Amplification: stronger campaign impact under high prescriber exposure ({sig})."

summary_df['Interpretation'] = summary_df.apply(interpret, axis=1)
summary_df = summary_df.round(4)

print("\n============================")
print("📊 SUMMARY TABLE")
print("============================")
print(summary_df)

# ==========================================
# 7️⃣ Marginal Effects Plot (Visualization)
# ==========================================
# Compute predicted EDS_conversion probabilities from OLS (easier to interpret)
pred_df = df.copy()
pred_df['pred_ols'] = ols_model.predict(pred_df)

# Create smoothed plot
plt.figure(figsize=(8,5))
sns.regplot(
    data=pred_df,
    x='prescriber_target_share',
    y='pred_ols',
    logistic=False,
    lowess=True,
    scatter_kws={'alpha':0.1},
    line_kws={'lw':2},
    label='_nolegend_'
)
sns.lineplot(
    data=pred_df,
    x='prescriber_target_share',
    y='pred_ols',
    hue='test_or_control',
    palette=['gray','blue']
)

plt.title("Marginal Effect of Prescriber Exposure on EDS Conversion")
plt.xlabel("Prescriber Target Share")
plt.ylabel("Predicted EDS Conversion (OLS)")
plt.legend(title="Member Group", labels=["Control","Target"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

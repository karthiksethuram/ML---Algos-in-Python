# ==========================================
# SPILLOVER REGRESSION TEMPLATE
# EDS_conversion ~ Target + Prescriber Exposure + Interaction
# ==========================================

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

# ===============================
# 1. INPUT: Your dataset
# ===============================
# df must include:
#   EDS_conversion          → Binary (0/1) or continuous
#   target_flag             → 1 = Target, 0 = Control
#   prescriber_id           → Prescriber identifier
#   age, gender, risk_score → Control variables (optional)

# Example structure (comment this out when using real data)
# df = pd.DataFrame({
#     'EDS_conversion': np.random.binomial(1, 0.4, 5000),
#     'target_flag': np.random.binomial(1, 0.5, 5000),
#     'prescriber_id': np.random.randint(100, 200, 5000),
#     'age': np.random.randint(45, 85, 5000),
#     'gender': np.random.choice(['M','F'], 5000),
#     'risk_score': np.random.normal(1.0, 0.3, 5000)
# })

# ===============================
# 2. Compute Prescriber Target Exposure
# ===============================
prescriber_summary = (
    df.groupby('prescriber_id')['target_flag']
      .agg(['count', 'sum'])
      .rename(columns={'count':'total_members','sum':'targeted_members'})
)
prescriber_summary['prescriber_target_share'] = (
    prescriber_summary['targeted_members'] / prescriber_summary['total_members']
)

df = df.merge(
    prescriber_summary[['prescriber_target_share']], 
    on='prescriber_id', how='left'
)
df['prescriber_target_share'] = df['prescriber_target_share'].clip(0,1)

# Optional: Encode gender if categorical
if df['gender'].dtype == 'object':
    df['gender'] = df['gender'].map({'M':1, 'F':0})

# ===============================
# 3. Define Model Formula
# ===============================
formula = """
EDS_conversion ~ target_flag 
               + prescriber_target_share
               + target_flag:prescriber_target_share
               + age + gender + risk_score
"""

# ===============================
# 4. OLS Model (interpretable in percentage points)
# ===============================
ols_model = smf.ols(formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prescriber_id']}
)
print("\nOLS Results (Clustered by Prescriber):")
print(ols_model.summary())

# ===============================
# 5. Logistic Model (for binary outcome)
# ===============================
try:
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    logit_cluster = logit_model.get_robustcov_results(
        cov_type='cluster', groups=df['prescriber_id']
    )
    print("\nLogistic Regression (Clustered SEs):")
    print(logit_cluster.summary())
except Exception as e:
    print("\n[Logit model skipped — likely due to convergence issues or non-binary outcome]")
    logit_cluster = None

# ===============================
# 6. Build Automatic Summary Table
# ===============================
summary_data = {
    'Effect': ['Direct Campaign Effect', 'Spillover Effect', 'Amplification (Interaction)'],
    'Variable': ['target_flag', 'prescriber_target_share', 'target_flag:prescriber_target_share'],
    'Beta (OLS)': [
        ols_model.params.get('target_flag', np.nan),
        ols_model.params.get('prescriber_target_share', np.nan),
        ols_model.params.get('target_flag:prescriber_target_share', np.nan)
    ],
    'p-value (OLS)': [
        ols_model.pvalues.get('target_flag', np.nan),
        ols_model.pvalues.get('prescriber_target_share', np.nan),
        ols_model.pvalues.get('target_flag:prescriber_target_share', np.nan)
    ]
}

if logit_cluster is not None:
    summary_data['Beta (Logit)'] = [
        logit_cluster.params.get('target_flag', np.nan),
        logit_cluster.params.get('prescriber_target_share', np.nan),
        logit_cluster.params.get('target_flag:prescriber_target_share', np.nan)
    ]
    summary_data['p-value (Logit)'] = [
        logit_cluster.pvalues.get('target_flag', np.nan),
        logit_cluster.pvalues.get('prescriber_target_share', np.nan),
        logit_cluster.pvalues.get('target_flag:prescriber_target_share', np.nan)
    ]

summary_df = pd.DataFrame(summary_data)

# Add qualitative interpretation
def interpret(row):
    if row['p-value (OLS)'] < 0.05:
        sig = "Significant"
    else:
        sig = "Not significant"
    if row['Effect'] == 'Direct Campaign Effect':
        return f"Direct impact of being targeted ({sig})."
    elif row['Effect'] == 'Spillover Effect':
        return f"Effect of being under a prescriber with targeted members ({sig})."
    else:
        return f"Amplification of campaign impact under high prescriber exposure ({sig})."

summary_df['Interpretation'] = summary_df.apply(interpret, axis=1)

# Round numbers for presentation
summary_df = summary_df.round(4)

print("\n============================")
print("📊 SUMMARY TABLE")
print("============================")
print(summary_df)

# Optional: export to Excel or CSV
# summary_df.to_csv("spillover_summary.csv", index=False)

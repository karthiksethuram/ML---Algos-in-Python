# --- 1. Imports ---
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Data prep ---
df = df.copy()

# Create binary and imputed variables
df['is_target_member'] = (df['target_or_control'] == 'Target').astype(int)
df['is_new_member'] = (df['member_status'] == 'New').astype(int)
df['star_pdc_py_imputed'] = df['star_pdc_py'].fillna(df['star_pdc_py'].mean())

# Polynomial (quadratic) term
df['presc_target_share_sq'] = df['presc_target_share'] ** 2

# --- 3. Logistic regression (non-linear specification) ---
formula_poly = """
EDS_conversion_flag_150d ~ is_target_member
    + presc_target_share
    + presc_target_share_sq
    + is_target_member:presc_target_share
    + is_target_member:presc_target_share_sq
    + star_pdc_py_imputed
    + is_new_member
    + C(star_drug_cls)
"""

logit_poly = smf.logit(formula=formula_poly, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['npi_id']}
)
print(logit_poly.summary())

# --- 4. Compute Average Marginal Effects (AMEs) ---
mfx_poly = logit_poly.get_margeff(at='overall', method='dydx')
mfx_summary = mfx_poly.summary_frame()
print("\n=== Average Marginal Effects (AMEs) ===")
print(mfx_summary)

# --- 5. Create prediction grid for visualization ---
grid = pd.DataFrame({
    'presc_target_share': np.linspace(0, 1, 100)
})
grid['presc_target_share_sq'] = grid['presc_target_share'] ** 2

# Set representative values for other predictors
grid['is_new_member'] = 0
grid['star_pdc_py_imputed'] = df['star_pdc_py_imputed'].mean()
grid['star_drug_cls'] = 'STATIN'   # pick one drug class for visualization

# Compute predicted probabilities for Target and Control members
for grp, label in zip([1, 0], ['Target', 'Control']):
    grid['is_target_member'] = grp
    grid['pred_prob_' + label] = logit_poly.predict(grid)

# --- 6. Visualization of predicted probabilities ---
plt.figure(figsize=(8,5))
sns.lineplot(x='presc_target_share', y='pred_prob_Target', data=grid, label='Target Members', linewidth=2)
sns.lineplot(x='presc_target_share', y='pred_prob_Control', data=grid, label='Control Members', linewidth=2)
plt.title('Predicted EDS Conversion Probability vs Prescriber Target Share')
plt.xlabel('Prescriber Target Share')
plt.ylabel('Predicted Probability of EDS Conversion')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# --- 7. Visualization of marginal effect (optional) ---
plt.figure(figsize=(8,5))
sns.lineplot(x='presc_target_share', y=grid['pred_prob_Target'] - grid['pred_prob_Control'])
plt.title('Estimated Spillover Gap (Target - Control)')
plt.xlabel('Prescriber Target Share')
plt.ylabel('Difference in Predicted Conversion Probability')
plt.axhline(0, color='gray', linestyle='--')
plt.grid(alpha=0.3)
plt.show()

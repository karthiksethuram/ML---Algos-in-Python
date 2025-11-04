import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ==========================================================
# 1️⃣ Load Data (replace with your actual dataset)
# ==========================================================
# df = pd.read_csv("your_data.csv")
# Columns expected: npi_id, is_target_member, presc_target_share, is_new_member, EDS_conversion_flag_150d

# Example check:
# df.head()

# ==========================================================
# 2️⃣ Fit GEE model clustered on prescriber (npi_id)
# ==========================================================
model = smf.gee(
    "EDS_conversion_flag_150d ~ is_target_member + presc_target_share + "
    "is_target_member:presc_target_share + is_new_member",
    groups="npi_id",
    data=df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable()
)
result = model.fit()
print(result.summary())

# ==========================================================
# 3️⃣ Interpret coefficients
# ==========================================================
coef = result.params
pvals = result.pvalues

print("\n--- INTERPRETATION ---")

# Intercept
print(f"Intercept ({coef['Intercept']:.2f}, p={pvals['Intercept']:.3f}):")
print("→ Baseline log-odds of conversion for a control member visiting a prescriber "
      "with presc_target_share = 0 and is_new_member = 0.\n")

# is_target_member
print(f"is_target_member ({coef['is_target_member']:.2f}, p={pvals['is_target_member']:.3f}):")
print("→ Targeted members have higher odds of converting than control members "
      "when their prescriber has zero exposure.\n")

# presc_target_share
print(f"presc_target_share ({coef['presc_target_share']:.2f}, p={pvals['presc_target_share']:.3f}):")
print("→ For control members, higher prescriber exposure (target share) "
      "is linked to higher conversion odds — suggesting possible spillover.\n")

# interaction term
print(f"is_target_member:presc_target_share ({coef['is_target_member:presc_target_share']:.2f}, "
      f"p={pvals['is_target_member:presc_target_share']:.3f}):")
print("→ Negative interaction: the lift from direct targeting decreases as prescriber exposure increases.\n"
      "Possible reasons:\n"
      "- Diminishing returns: highly engaged prescribers already influence their patients.\n"
      "- Multicollinearity between is_target_member and presc_target_share.\n"
      "- Nonlinear (saturation) effects not captured by linear model.\n")

# is_new_member
print(f"is_new_member ({coef['is_new_member']:.2f}, p={pvals['is_new_member']:.3f}):")
print("→ Positive effect: newly onboarded members have higher conversion odds, "
      "perhaps due to greater initial engagement.\n")

print("👉 Summary: Both direct targeting and prescriber exposure help conversions, "
      "but their combination shows diminishing returns (less-than-additive effect).")

# ==========================================================
# 4️⃣ Marginal Effects Plot (Predicted Probabilities)
# ==========================================================

# Create prediction grid
share_seq = np.linspace(0, 1, 100)
plot_df = pd.DataFrame({
    "presc_target_share": np.tile(share_seq, 2),
    "is_target_member": np.repeat([0, 1], len(share_seq)),
    "is_new_member": 0  # fix new member status for clarity
})

# Predict log-odds and probability
plot_df["predicted_logit"] = result.predict(plot_df)
plot_df["predicted_prob"] = 1 / (1 + np.exp(-plot_df["predicted_logit"]))

# Plot marginal effects
plt.figure(figsize=(8, 6))
for target_flag, label, color in zip([0, 1], ["Control Member", "Targeted Member"], ["blue", "red"]):
    subset = plot_df[plot_df["is_target_member"] == target_flag]
    plt.plot(subset["presc_target_share"], subset["predicted_prob"], label=label, color=color, linewidth=2)

plt.xlabel("Prescriber Target Share", fontsize=12)
plt.ylabel("Predicted Probability of EDS Conversion (150 days)", fontsize=12)
plt.title("Marginal Effects of Member Targeting and Prescriber Exposure", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ---------------------------
# 0. Load your data (replace)
# ---------------------------
# df = pd.read_csv("your_data.csv")
# expected cols: npi_id, is_target_member, presc_target_share, is_new_member, EDS_conversion_flag_150d

# ---------------------------
# 1. Fit GEE
# ---------------------------
formula = "EDS_conversion_flag_150d ~ is_target_member + presc_target_share + is_target_member:presc_target_share + is_new_member"
model = smf.gee(formula, groups="npi_id", data=df,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
result = model.fit()
print(result.summary())

# ---------------------------
# 2. Build prediction grid
# ---------------------------
share_seq = np.linspace(0, 1, 200)  # finer resolution for smooth CI bands
pred_rows = []
for t in [0, 1]:
    for s in share_seq:
        pred_rows.append({
            "is_target_member": t,
            "presc_target_share": s,
            "is_new_member": 0  # fix for plotting; change if you want different strata
        })
plot_df = pd.DataFrame(pred_rows)

# ---------------------------
# 3. Create design matrix matching model param order
# ---------------------------
# Get ordered parameter names from fitted result
param_names = result.params.index.tolist()

# Build X design matrix columns in same order
# Expecting names like: 'Intercept', 'is_target_member', 'presc_target_share',
# 'is_target_member:presc_target_share', 'is_new_member'
X = pd.DataFrame(index=plot_df.index, columns=param_names, dtype=float)

# Fill columns
# Intercept
if 'Intercept' in X.columns:
    X['Intercept'] = 1.0
# main terms
X['is_target_member'] = plot_df['is_target_member'].astype(float)
X['presc_target_share'] = plot_df['presc_target_share'].astype(float)
# interaction (name matches result.params index)
inter_name = None
for nm in param_names:
    if ':' in nm and 'is_target_member' in nm and 'presc_target_share' in nm:
        inter_name = nm
        break
if inter_name is None:
    # fallback - try the literal name used earlier
    inter_name = 'is_target_member:presc_target_share'
if inter_name not in X.columns:
    X[inter_name] = X['is_target_member'] * X['presc_target_share']
else:
    X[inter_name] = X['is_target_member'] * X['presc_target_share']

# is_new_member
X['is_new_member'] = plot_df['is_new_member'].astype(float)

# Fill any remaining params (if any) with zeros to avoid shape mismatch
for col in X.columns:
    X[col] = X[col].fillna(0.0)

# ---------------------------
# 4. Linear predictor, SE, and 95% CI
# ---------------------------
params = result.params.values.reshape(-1, 1)                # (p,1)
covb = result.cov_params()                                  # (p,p)
X_mat = X.values                                            # (n,p)

# linear predictor
lp = X_mat.dot(params).flatten()                            # (n,)

# se for each prediction: sqrt(diag(X * covb * X'))
# compute vectorized: se2 = sum over i,j of X[:,i]*covb[i,j]*X[:,j]
se2 = np.einsum('ij,jk,ik->i', X_mat, covb.values, X_mat)   # (n,)
se = np.sqrt(np.maximum(se2, 0))                            # guard numerical negatives

# 95% CI on linear predictor
z = 1.96
lp_low = lp - z * se
lp_high = lp + z * se

# transform to probability space using logistic
def logistic(x): return 1.0 / (1.0 + np.exp(-x))
prob = logistic(lp)
prob_low = logistic(lp_low)
prob_high = logistic(lp_high)

plot_df['pred_prob'] = prob
plot_df['prob_low'] = prob_low
plot_df['prob_high'] = prob_high
plot_df['presc_target_share'] = plot_df['presc_target_share'].astype(float)
plot_df['is_target_member'] = plot_df['is_target_member'].astype(int)

# ---------------------------
# 5. Plot with confidence bands
# ---------------------------
plt.figure(figsize=(9,6))

for t, label, color in zip([0,1], ['Control Member', 'Targeted Member'], ['C0','C3']):
    subset = plot_df[plot_df['is_target_member']==t]
    x = subset['presc_target_share'].values
    y = subset['pred_prob'].values
    ylow = subset['prob_low'].values
    yhigh = subset['prob_high'].values

    plt.plot(x, y, label=label, color=color, linewidth=2)
    plt.fill_between(x, ylow, yhigh, color=color, alpha=0.2)

plt.xlabel("Prescriber Target Share", fontsize=12)
plt.ylabel("Predicted Probability of EDS Conversion (150 days)", fontsize=12)
plt.title("Predicted Probability vs Prescriber Target Share\n(with 95% Confidence Bands)", fontsize=13)
plt.legend()
plt.grid(alpha=0.25)
plt.ylim(0,1)
plt.show()

# ---------------------------
# 6. Robust predicted deltas at key exposures
#    (use nearest grid index instead of exact float equality)
# ---------------------------
key_exposures = [0.0, 0.25, 0.5, 0.75, 1.0]
unique_shares = np.array(sorted(plot_df['presc_target_share'].unique()))
print("\nPredicted target lift (Target - Control) at selected prescriber exposure levels:")
for x in key_exposures:
    idx = (np.abs(unique_shares - x)).argmin()   # nearest grid point index
    share_val = unique_shares[idx]
    # locate control and target rows for that share
    p_control = plot_df[(plot_df['presc_target_share']==share_val) & (plot_df['is_target_member']==0)]['pred_prob'].values[0]
    p_target  = plot_df[(plot_df['presc_target_share']==share_val) & (plot_df['is_target_member']==1)]['pred_prob'].values[0]
    delta = (p_target - p_control) * 100  # percentage points
    print(f"  At prescriber exposure ≈ {share_val:.3f}: Targeted lift = {delta:.2f} percentage points")

# ---------------------------
# 7. (Optional) Save plot to file
# ---------------------------
# plt.savefig("gee_marginal_effects_with_ci.png", dpi=200, bbox_inches='tight')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1️⃣ Load data (replace this with your actual)
# =====================================
# df = pd.read_csv("your_member_level_data.csv")
# Expected columns: npi_id, is_target_member, presc_target_share, is_new_member, EDS_conversion_flag_150d

# =====================================
# 2️⃣ Aggregate to prescriber level
# =====================================

# Compute prescriber-level averages
prescriber_df = (
    df.groupby("npi_id")
    .agg(
        total_members=("EDS_conversion_flag_150d", "count"),
        conversion_rate=("EDS_conversion_flag_150d", "mean"),
        presc_target_share=("presc_target_share", "mean"),
        pct_target_members=("is_target_member", "mean"),
        pct_new_members=("is_new_member", "mean"),
    )
    .reset_index()
)

# =====================================
# 3️⃣ Regression: ConversionRate_j ~ presc_target_share_j + controls
# =====================================
X = prescriber_df[["presc_target_share", "pct_target_members", "pct_new_members"]]
X = sm.add_constant(X)
y = prescriber_df["conversion_rate"]

model = sm.OLS(y, X).fit(cov_type='HC3')  # robust SEs
print(model.summary())

# =====================================
# 4️⃣ Quartile bucketing by exposure
# =====================================
prescriber_df["exposure_quartile"] = pd.qcut(prescriber_df["presc_target_share"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

# Compute mean conversion rates by quartile
quartile_summary = (
    prescriber_df.groupby("exposure_quartile")
    .agg(
        mean_conversion_rate=("conversion_rate", "mean"),
        mean_target_share=("presc_target_share", "mean"),
        n_prescribers=("npi_id", "count"),
    )
    .reset_index()
)

print("\nQuartile Summary:")
print(quartile_summary)

# =====================================
# 5️⃣ Visualization: Conversion rate by exposure quartile
# =====================================
plt.figure(figsize=(8,6))
sns.barplot(
    data=quartile_summary,
    x="exposure_quartile",
    y="mean_conversion_rate",
    palette="coolwarm",
)
plt.title("Prescriber Conversion Rate by Target Exposure Quartile", fontsize=14)
plt.xlabel("Prescriber Target Exposure Quartile", fontsize=12)
plt.ylabel("Mean Conversion Rate (150 days)", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

# =====================================
# 6️⃣ (Optional) Trend line view — scatter + regression fit
# =====================================
plt.figure(figsize=(8,6))
sns.regplot(
    data=prescriber_df,
    x="presc_target_share",
    y="conversion_rate",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title("Prescriber Conversion Rate vs Prescriber Target Exposure", fontsize=14)
plt.xlabel("Prescriber Target Share", fontsize=12)
plt.ylabel("Conversion Rate (150 days)", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

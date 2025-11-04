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

# ==========================================================
# 5️⃣ Optional: Print predicted delta at key exposure levels
# ==========================================================
for x in [0, 0.25, 0.5, 0.75, 1.0]:
    control = plot_df.query(f"presc_target_share=={x} and is_target_member==0")["predicted_prob"].values[0]
    target = plot_df.query(f"presc_target_share=={x} and is_target_member==1")["predicted_prob"].values[0]
    print(f"At prescriber exposure {x:.2f}: Targeted lift = {(target - control)*100:.2f}% points")

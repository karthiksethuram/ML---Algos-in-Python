from scipy import stats

# Ensure Statin_Filled and MEMBER_STATUS exist in matched data
assert 'Statin_Filled' in df_matched.columns, "Statin_Filled variable not found in df_matched"
assert 'MEMBER_STATUS' in df_matched.columns, "MEMBER_STATUS variable not found in df_matched"

print("\n=== 💊 Outcome Analysis: Statin_Filled ===")

# ---- 1️⃣ Overall difference ----
treated = df_matched[df_matched['treatment'] == 1]
control = df_matched[df_matched['treatment'] == 0]

t_stat, p_val = stats.ttest_ind(treated['Statin_Filled'], control['Statin_Filled'], equal_var=False)

mean_treated = treated['Statin_Filled'].mean()
mean_control = control['Statin_Filled'].mean()
delta = mean_treated - mean_control

print(f"\nOverall:")
print(f"Mean (Target group): {mean_treated:.3f}")
print(f"Mean (Control group): {mean_control:.3f}")
print(f"Δ (Difference): {delta:.3f}")
print(f"t = {t_stat:.3f}, p = {p_val:.4f}")

# ---- 2️⃣ By MEMBER_STATUS (New vs Returning) ----
print("\n=== 🧩 Subgroup Analysis by MEMBER_STATUS ===")

for status in df_matched['MEMBER_STATUS'].unique():
    subset = df_matched[df_matched['MEMBER_STATUS'] == status]
    treated_sub = subset[subset['treatment'] == 1]
    control_sub = subset[subset['treatment'] == 0]

    if treated_sub.shape[0] > 1 and control_sub.shape[0] > 1:
        t_stat, p_val = stats.ttest_ind(
            treated_sub['Statin_Filled'],
            control_sub['Statin_Filled'],
            equal_var=False
        )

        mean_treated = treated_sub['Statin_Filled'].mean()
        mean_control = control_sub['Statin_Filled'].mean()
        delta = mean_treated - mean_control

        print(f"\nMember Status: {status}")
        print(f"Mean (Target): {mean_treated:.3f}")
        print(f"Mean (Control): {mean_control:.3f}")
        print(f"Δ (Difference): {delta:.3f}")
        print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        print(f"\nMember Status: {status} — insufficient data for both groups.")

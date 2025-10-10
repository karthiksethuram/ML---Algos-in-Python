from scipy import stats

# --- 1. Compute group-wise fill rates ---
group_stats = (
    df_matched.groupby('group')['statin_filled']
    .agg(['mean', 'count', 'std'])
    .rename(columns={'mean': 'fill_rate', 'count': 'n', 'std': 'std_dev'})
)

print("Group-wise Statin Fill Rates:")
print(group_stats)
print("\n")

# --- 2. Run Welch’s t-test (unequal variances) ---
target = df_matched[df_matched['group'] == 1]['statin_filled']
control = df_matched[df_matched['group'] == 0]['statin_filled']

t_stat, p_val = stats.ttest_ind(target, control, equal_var=False)

# --- 3. Print results ---
print("Welch’s t-test results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")

# --- 4. Interpretation ---
alpha = 0.05
if p_val < alpha:
    print("\n✅ The difference in statin fill rates between groups is statistically significant (p < 0.05).")
else:
    print("\n❌ No statistically significant difference in statin fill rates between groups (p ≥ 0.05).")

Nice — you’ve already run the right family of approaches. Below I’ll explain why each model was chosen, what it estimates / assumes, why one would group by npi_id vs eph_id, what cov_type='clustered' does, and how these choices change interpretation vs plain OLS / logit. I’ll finish with clear practical guidance on which results to trust for each question you care about (spillover, diminishing returns, overall effect) and a few diagnostics to run.

1) Short summary (TL;DR)

GEE grouped on npi_id → population-average (marginal) estimates while accounting for within-prescriber correlation. Good when prescriber-level clustering / spillover is the main dependence you worry about.

GEE grouped on eph_id → population-average estimates accounting for repeated rows per member (multiple experiment rows per member). Good when correlation within a member (e.g., member targeted for multiple drug classes) is dominant.

PanelOLS (prescriber fixed effects) with cov_type='clustered' → within-prescriber causal contrasts (controls prescriber baseline), with cluster-robust standard errors that allow residuals to be correlated within prescribers. This is a within-prescriber estimate (removes prescriber confounding).

Difference vs OLS/Logit: FE and GEE are ways of accounting for correlated data / unobserved cluster confounding. Plain OLS/logit assumes independent observations and will give biased SEs (and biased coefficients if prescriber confounding exists).

2) Why you picked these models (rationale)
GEE (Generalized Estimating Equations)

Why chosen: GEE is designed to estimate marginal (population-average) effects for correlated outcome data (common in cluster/longitudinal data). You have two natural correlation sources: members sharing prescribers (clusters) and members appearing multiple times (multiple conditions). GEE lets you specify which cluster structure to account for.

What it estimates: The average change in probability of conversion associated with predictors (e.g., is_target_member) across the population, marginalizing over cluster effects.

Key assumption: Correct specification of the mean model (link & predictors). The working correlation (exchangeable, independent, etc.) affects efficiency but not consistency of coefficient estimates if mean model is correct.

Why group by npi_id: If the dominant dependence is that members of the same prescriber are correlated (e.g., prescriber practice patterns, EHR behavior), group by prescriber so GEE accounts for that within-prescriber correlation. This is essential to get correct standard errors and valid inference about spillover.

Why group by eph_id: If the same member (eph_id) appears multiple times (multiple conditions), residuals for that same member will be correlated across rows. Grouping by eph_id ensures within-member correlation is modeled. Use this when member-level repeated measures are primary.

PanelOLS with prescriber Fixed Effects (EntityEffects)

Why chosen: FE removes all time-invariant prescriber-level confounders by demeaning (or adding prescriber dummies). Since prescribers differ in unobserved ways (patient mix, outreach style, office workflows) that plausibly affect conversion, FE isolates within-prescriber comparisons — comparing targeted vs non-targeted members who see the same prescriber.

What it estimates: The effect of targeting and its interaction with prescriber exposure within prescribers (i.e., holding prescriber constant). Good for causal claims when you trust randomization at member level but worry prescriber-level confounding.

When it fails: If a prescriber is “pure” (all treated or all control), FE removes that prescriber’s treatment indicator — you can’t estimate within-prescriber effects there (we saw that earlier).

3) What cov_type='clustered' does (PanelOLS / statsmodels)

cov_type='clustered' instructs the model to compute cluster-robust (sandwich) standard errors that allow residuals to be arbitrarily correlated within the specified cluster (e.g., npi_id) while still being independent across clusters.

Effect: Coefficient estimates are unchanged, but standard errors (and p-values / CIs) are adjusted to account for within-cluster dependence. This is crucial when you have many observations per prescriber and residual dependence is present.

Caveats:

If number of clusters is small (< ~50), cluster-robust SEs can be downward biased; use caution and consider cluster bootstrap or wild cluster bootstrap.

Cluster-robust SEs do not fix bias in coefficients caused by omitted cluster-level confounders — that’s what FE (or random effects with covariates) attempts to address.

4) How these differ from plain linear / logistic regression
Plain OLS or logit (single-level)

Assumes independence of observations.

If independence violated (e.g., members clustered by prescriber) → SEs are too small (Type I error inflation) and coefficient estimates can be biased if cluster-level confounding exists.

No explicit modeling of cluster correlation or unobserved cluster heterogeneity.

GEE (clustered)

Robust SEs for clusters built-in via working correlation; estimates marginal effects. Coefficients consistent even if working correlation misspecified.

Population-averaged interpretation: e.g., “on average, X% change in probability across population.”

Doesn’t model random intercepts — it treats cluster correlation as nuisance.

PanelOLS with FE

Removes prescriber-level fixed heterogeneity (controls for unobserved time-invariant prescriber traits). That changes coefficient interpretation to within-prescriber effects.

Estimates conditional effects (conditional on prescriber).

Paired with cluster-robust SEs it also addresses residual within-cluster correlation.

Mixed Effects (random effects)

Model prescriber variation as random draws from a distribution. Random intercepts let you estimate both within- and between-cluster effects and provide subject-specific predictions.

Interpretation: Subject-specific (conditional) effects, unless you transform to marginal via integration.

Requires stronger assumptions (random effects uncorrelated with regressors unless you model correlation). If that assumption fails, random effects coefficients can be biased.

5) Practical guidance: which to use for each question

Goal → Recommended primary model (and why)

Do we have spillover (controls benefit from high prescriber exposure)?
→ Use GEE grouped by npi_id (Binomial family). This gives a robust population-average estimate of the association between presc_target_share and control conversion probability, accounting for within-prescriber correlation. Also show prescriber-level aggregated tests (prescriber means, permutations).

Is there diminishing returns in treatment effect with exposure?
→ Use PanelOLS with prescriber FE (within-transformation) on full data and a GEE with interaction term. FE shows whether the interaction remains after controlling prescriber baseline; GEE shows population-average curve. If both agree (negative interaction), stronger evidence.

Overall effect of outreach (average)?
→ GEE (marginal) gives the average effect; PanelOLS FE gives the within-prescriber effect (use both).

Robustness & transparency for leadership
→ Provide: (a) bin-wise simple two-sample tests, (b) Prescriber-aggregated comparisons (weighted), (c) GEE (npi clustering), (d) FE within-prescriber results. If all point same direction → convincing.

6) Specific notes comparing group by npi_id vs group by eph_id in your context

Group by npi_id (prescriber) is most natural when:

You expect correlation / interference across members because they share the same prescriber behavior/EHR exposure.

Primary nuisance is prescriber-level unobserved heterogeneity or spillover (controls benefiting because prescriber was targeted).

Group by eph_id (member/experiment unit) is natural when:

The same member has multiple observations (different drug classes) and you expect within-member correlation (e.g., an engaged member more likely to convert across drug classes).

Which to prefer: If both dependencies exist (members nested within prescribers and members have multiple rows), ideally model both: e.g., mixed model with random intercepts for prescribers and for members (crossed random effects), or run GEE with prescriber clustering and include member-level fixed effects or vice-versa. Practically, start with the cluster most relevant to your hypothesis: npi_id for spillover.

7) Diagnostics & robustness checks to run (short checklist)

Compare coefficient sign & significance across models: GEE (npi), GEE (eph), FE (PanelOLS), MixedLM. If direction consistent → confidence grows.

Marginal effect plots (with CIs) for the interaction: do both models show diminishing returns or negative dips? Check CI width at extremes.

Prescriber-aggregated checks: collapse to prescriber mean conversion vs prescriber-level share; run weighted regressions and permutation tests (transparent).

ICC: compute intra-class correlation to quantify how much variance is at prescriber vs member level:

ICC_prescriber = Var_prescriber / (Var_prescriber + Var_residual) — estimate via random intercept model.

Check cluster sizes: if many tiny clusters (1 obs), cluster-robust SEs may be unstable. Use wild cluster bootstrap if clusters small in count.

Collinearity / VIF: especially for interaction terms and share variables.

Sample sizes by share bin: extremes may be noisy; interpret cautiously.

8) Short practical recommendations for your leadership deliverable

Present the GEE (npi clustered) results as the main population-average evidence for spillover and average effect. It accounts for prescriber correlation and is easy to explain: “on average, after accounting for prescriber clustering…”

Present PanelOLS FE results as a robustness / causal check that controls for prescriber-specific baselines: “holding prescriber constant…”

Show a prescriber-aggregated, weighted comparison for the pure prescriber analysis (transparent and simple).

Show marginal effect curves from both GEE and FE with CIs—leadership will understand “effect shrinks with exposure” or “controls rise with exposure”.

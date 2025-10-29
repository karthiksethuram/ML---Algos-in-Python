# --- Install packages (if not already) ---
# !pip install rpy2
# In R: install.packages("interference")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Import R packages
ro.r('library(interference)')

# Pass your pandas DataFrame to R
r_df = pandas2ri.py2rpy(df)

# Define the R command
ro.r.assign("r_df", r_df)
ro.r('''
results <- interference(
  formula = EDS_conversion_flag_150d ~ star_pdc_py_imputed + is_new_member + factor(star_drug_cls),
  treat = is_target_member,
  group = npi_id,
  allocation = "observed",
  model_method = "logit",
  effect = "both",
  data = r_df
)
print(summary(results))
''')

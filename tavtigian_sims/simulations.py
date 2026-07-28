#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys, os
sys.path.insert(0, os.path.abspath(".."))   # adds assay_calibration/ to path
sys.path.insert(0, os.path.abspath("../src"))  # adds assay_calibration/src/ for evidence_thresholds

from tavtigian_sims import prior_grid
import importlib
import tavtigian_sims.compare
importlib.reload(tavtigian_sims.compare)
from tavtigian_sims.compare import run_all_methods, plot_three_way_comparison, \
    plot_boundary_posteriors_three_way, plot_additivity_experiment, plot_combined_error_comparison, plot_additivity_dilemma, \
    plot_slope_geometry, plot_combination_paths


# In[3]:


priors = prior_grid("paper")
t, pw, pw_add, lpa, cont = run_all_methods(priors, n_jobs=64)


# In[4]:


cont


# In[6]:


plot_three_way_comparison(t, pw, cont_df=cont, log_scale=False)#, codes="all")


# In[8]:


plot_boundary_posteriors_three_way(t, pw, log_scale=False)  # 4 methods


# In[7]:


# fig, axes = plot_additivity_experiment(
#     priors=(0.05, 0.10, 0.25, 0.50),   # which priors to show in the slope panels (row 0)
#     tav_df=t,                            # reuses precomputed C* — avoids re-running grid search
#     pw_add_df=pw_add
# )


# In[20]:


# plot_combined_error_comparison(priors, tav_df=t, pw_add_df=pw_add, log_scale=False)


# In[9]:


# from tavtigian_sims import prior_grid
# from tavtigian_sims.compare import (
#     run_all_methods,
#     plot_additivity_dilemma,
#     plot_slope_geometry,
# )

# priors = prior_grid("paper")
# t, pw, pw_add, cont = run_all_methods(priors, n_jobs=10)

# fig, _ = plot_additivity_dilemma(priors, tav_df=t)
# fig.savefig('figures/additivity_dilemma.pdf', bbox_inches='tight')

# fig, _ = plot_slope_geometry(tav_df=t)
# fig.savefig('figures/slope_geometry.pdf', bbox_inches='tight')


# In[12]:


fig, _ = plot_additivity_dilemma(priors, tav_df=t)#, lpa_df=lpa)
# fig.savefig('figures/additivity_dilemma.pdf', bbox_inches='tight')

fig, _ = plot_slope_geometry(tav_df=t)#, lpa_df=lpa)
# fig.savefig('figures/slope_geometry.pdf', bbox_inches='tight')


# In[10]:


fig, axes = plot_combination_paths(priors, tav_df=t)


# In[ ]:





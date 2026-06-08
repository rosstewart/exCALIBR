#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, json, pickle, gzip, logging, warnings
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from joblib import Parallel, delayed

logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
from src.assay_calibration.fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from src.assay_calibration.fit_utils.two_sample import density_utils  # noqa: E402
from src.assay_calibration.fit_utils.point_ranges import (enforce_monotonicity_point_ranges, prior_equation_2c, prior_invalid, get_fit_prior, extend_points_to_xlims, get_bootstrap_score_ranges, remove_insufficient_bootstrap_converage_points, check_thresholds_reached)  # noqa: E402
from src.assay_calibration.data_utils.dataset import Scoreset  # noqa: E402
from src.assay_calibration.fit_utils.utils import serialize_dict  # noqa: E402
from src.assay_calibration.plot_utils.utils import plot_scoreset, plot_scoreset_compare_point_assignments, plot_scoreset_example_publication
import pandas as pd
import importlib
import src.assay_calibration.data_utils.dataset
importlib.reload(src.assay_calibration.data_utils.dataset)
from src.assay_calibration.data_utils.dataset import Scoreset


# In[157]:


df_auth = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_analysis.csv.gz")


# In[158]:


with open("/data/ross/assay_calibration/dataframe/new_names_dict.pkl","rb") as f:
    new_names_dict = pickle.load(f)
old_names_dict = {v: k for k,v in new_names_dict.items()}


# In[159]:


# df_auth = pd.read_csv("/data/ross/assay_calibration/author_comparison/Fxn_Pejaver_calibrations_w_notes_ppdf111225_120125_w_standardized_class.csv.gz")
# df_final = df_auth

df_auth['Dataset'] = df_auth['Dataset'].map(old_names_dict)
df_auth = df_auth[~df_auth.Dataset.isin(["TP53_Fayer_2021_meta","SFPQ_unpublished","F9_Popp_2025_model","TSC2_rapgap_unpublished","TSC2_tuberin_unpublished"])]
df_final = df_auth
df_clingen = df_auth
df_clingen_unique = df_clingen.Dataset.unique()

# df_final = pd.read_csv('/data/ross/assay_calibration/Dan_fxn_calibrations_new_2025_2018_clust_calib_2025_12_27.csv.gz')
# df_auth = df_final


# In[14]:


df_std = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')
df_std = df_std[~df_std.Dataset.isin(["TP53_Fayer_2021_meta","SFPQ_unpublished","F9_Popp_2025_model","TSC2_rapgap_unpublished","TSC2_tuberin_unpublished"])]


# In[160]:


# df_clingen = pd.read_csv('/data/ross/assay_calibration/Dan_fxn_calibrations_new_2025_2018_clust_calib_2025_12_27.csv.gz')
# df_clingen_unique = df_clingen.Dataset.unique()


# In[161]:


# for c in df_auth.columns:
#     if "ClinGen" in c:
#         print('df_auth',c)

# for c in df_std.columns:
#     if "ClinGen" in c:
#         print('df_std',c)
        
# for c in df_clingen.columns:
#     if "ClinGen" in c:
#         print('df_clingen',c)
#         # print(f"class [{c}] unique vals: {df_auth[c].unique()}")


# In[162]:


reported_list = ['BRCA1_Findlay_2018', 'BRCA2_Hu_2024', 'VHL_Buckley_2024',
       'JAG1_Gilbert_2024', 'BARD1_unpublished', 'PALB2_unpublished',
       # 'SFPQ_unpublished',
       'RAD51D_unpublished', 'CTCF_unpublished',
       'BAP1_Waters_2024', 'DDX3X_Radford_2023_cLFC_day15',
       'RHO_Wan_2019', 'RAD51C_Olvera-León_2024_z_score_D4_D14',
       'FKRP_Ma_2024', 'LARGE1_Ma_2024',
       'CARD11_Meitlis_2020_Ibrutinib_no_introns',
       'CARD11_Meitlis_2020_DMSO_no_introns',
       'ASPA_Grønbæk-Thygesen_2024_abundance',
       'ASPA_Grønbæk-Thygesen_2024_toxicity',
       'BRCA1_Adamovich_2022_Cisplatin', 'BRCA1_Adamovich_2022_HDR',
       'CHEK2_Gebbia_2024', 'CRX_Shepherdson_2024', #'F9_Popp_2025_model',
       'G6PD_unpublished', 'GCK_Gersing_2023_complementation',
       'GCK_Gersing_2024_abundance','KCNE1_Muhammad_2024_absence_of_WT',
       'KCNE1_Muhammad_2024_potassium_flux',
       'KCNE1_Muhammad_2024_presence_of_WT', 
       'KCNH2_Jiang_2022',
       'KCNH2_Kozek_Glazer_2020', 'KCNH2_O_Neill_2024_surface_expression',
       'MSH2_Scott_2022', 'NDUFAF6_Sung_2024', 'OTC_Lo_2023','PTEN_Matreyek_2018',
       'PTEN_Mighell_2018', 'SCN5A_Glazer_2020',
       'SCN5A_Ma_2024_current_density', 'SGCB_Li_2023',
       # 'TP53_Fortuno_2021_Kato_meta',
       # 'TSC2_rapgap_unpublished', 'TSC2_tuberin_unpublished',
       'TSC2_combined_unpublished',
       'KCNQ4_Zheng_2022_current_homozygous',
       'KCNQ4_Zheng_2022_v12_homozygous']


# In[ ]:


df_auth[df_auth["Dataset"].isin(reported_list)]["StandardizedClass"].unique()


# In[164]:


# calibrated_datasets = [
#     'ASPA_Grønbæk-Thygesen_2024_abundance',
#     'ASPA_Grønbæk-Thygesen_2024_toxicity',
#     'BAP1_Waters_2024',
#     'BARD1_unpublished',
#     'BRCA1_Adamovich_2022_Cisplatin',
#     'BRCA1_Adamovich_2022_HDR',
#     'BRCA1_Findlay_2018',
#     'BRCA2_Hu_2024',
#     'BRCA2_Sahu_2023_exon13_Cisplatin',
#     'BRCA2_Sahu_2023_exon13_Olaparib',
#     'BRCA2_Sahu_2023_exon13_SGE',
#     'BRCA2_Sahu_2023_exon13_global_score',
#     'BRCA2_Sahu_2025_HDR',
#     'BRCA2_unpublished',
#     'CALM1_CALM2_CALM3_Weile_2017',
#     'CARD11_Meitlis_2020_DMSO_no_introns',
#     'CARD11_Meitlis_2020_Ibrutinib_no_introns',
#     'CBS_Sun_2020_high_B6',
#     'CBS_Sun_2020_low_B6',
#     'CHK2_Gebbia_2024',
#     'CRX_Shepherdson_2024',
#     'CTCF_unpublished',
#     'DDX3X_Radford_2023_cLFC_day15',
#     'F9_Popp_2025_carboxy_F9_specific',
#     'F9_Popp_2025_carboxy_gla_motif',
#     'F9_Popp_2025_heavy_chain',
#     'F9_Popp_2025_light_chain',
#     'F9_Popp_2025_strep_2',
#     'FKRP_Ma_2024',
#     'G6PD_unpublished',
#     'GCK_Gersing_2023_complementation',
#     'GCK_Gersing_2024_abundance',
#     'HMBS_van_Loggerenberg_2023_combined',
#     'HMBS_van_Loggerenberg_2023_erythroid',
#     'HMBS_van_Loggerenberg_2023_ubquitous',
#     'JAG1_Gilbert_2024',
#     'KCNE1_Muhammad_2024_absence_of_WT',
#     'KCNE1_Muhammad_2024_potassium_flux',
#     'KCNE1_Muhammad_2024_presence_of_WT',
#     'KCNH2_Jiang_2022',
#     'KCNH2_O_Neill_2024_surface_expression',
#     'KCNQ4_Zheng_2022_current_homozygous',
#     'KCNQ4_Zheng_2022_v12_homozygous',
#     'LARGE1_Ma_2024',
#     'MSH2_Jia_2021',
#     'NDUFAF6_Sung_2024',
#     'OTC_Lo_2023',
#     'PALB2_unpublished',
#     'PAX6_McDonnell_2024_BLX_geneticin',
#     'PAX6_McDonnell_2024_BLX_no_geneticin',
#     'PAX6_McDonnell_2024_LE9_geneticin',
#     'PAX6_McDonnell_2024_LE9_no_geneticin',
#     'PTEN_Matreyek_2018',
#     'PTEN_Mighell_2018',
#     'RAD51C_Olvera-León_2024_z_score_D4_D14',
#     'RAD51D_unpublished',
#     'RHO_Wan_2019',
#     'SCN5A_Glazer_2020',
#     'SCN5A_Ma_2024_current_density',
#     'TP53_Boettcher_2019',
#     'TP53_Fortuno_2021_Kato_meta',
#     'TP53_Giacomelli_2018_combined_score',
#     'TP53_Giacomelli_2018_p53WT_Nutlin3',
#     'TP53_Giacomelli_2018_p53null_Nutlin3',
#     'TP53_Giacomelli_2018_p53null_etoposide',
#     'TP53_Kato_2003_AIP1nWT',
#     'TP53_Kato_2003_BAXnWT',
#     'TP53_Kato_2003_GADD45nWT',
#     'TP53_Kato_2003_MDM2nWT',
#     'TP53_Kato_2003_NOXAnWT',
#     'TP53_Kato_2003_P53R2nWT',
#     'TP53_Kato_2003_WAF1nWT',
#     'TP53_Kato_2003_h1433snWT',
#     'TPK1_Weile_2017',
#     'TSC2_rapgap_unpublished',
#     'TSC2_tuberin_unpublished',
#     'VHL_Buckley_2024',
#     'XRCC2_unpublished',
# ]


# In[165]:


# with open('/data/ross/assay_calibration/new_dataset_configs.json','wt') as f:
#     config_data = {
#      'ASPA_Grønbæk-Thygesen_2024_abundance': ('3c', 'avg'),
#      'ASPA_Grønbæk-Thygesen_2024_toxicity': ('3c', 'avg'),
#      'BAP1_Waters_2024': ('3c', 'avg'),
#      'BAP1_Waters_2024_clinvar_2018': ('3c', 'avg'),
#      'BARD1_unpublished': ('3c', 'avg'),
#      'BRCA1_Adamovich_2022_Cisplatin': ('2c', 'avg'),
#      'BRCA1_Adamovich_2022_Cisplatin_clinvar_2018': ('2c', 'avg'),
#      'BRCA1_Adamovich_2022_HDR': ('2c', 'avg'),
#      'BRCA1_Adamovich_2022_HDR_clinvar_2018': ('2c', 'avg'),
#      'BRCA1_Findlay_2018': ('2c', 'avg'),
#      'BRCA1_Findlay_2018_clinvar_2018': ('2c', 'avg'),
#      'BRCA2_Hu_2024': ('2c', 'avg'),
#      'BRCA2_Hu_2024_clinvar_2018': ('2c', 'avg'),
#      'BRCA2_Sahu_2023_exon13_Cisplatin': ('2c', 'benign'),
#      'BRCA2_Sahu_2023_exon13_Cisplatin_clinvar_2018': ('2c', 'benign'),
#      'BRCA2_Sahu_2023_exon13_Olaparib': ('3c', 'avg'),
#      'BRCA2_Sahu_2023_exon13_Olaparib_clinvar_2018': ('3c', 'avg'),
#      'BRCA2_Sahu_2023_exon13_SGE': ('3c', 'benign'), # KEEP OLD
#      'BRCA2_Sahu_2023_exon13_SGE_clinvar_2018': ('3c', 'avg'), # KEEP OLD
#      'BRCA2_Sahu_2023_exon13_global_score': ('2c', 'benign'),
#      'BRCA2_Sahu_2023_exon13_global_score_clinvar_2018': ('2c', 'benign'),
#      'BRCA2_Sahu_2025_HDR': ('2c', 'avg'),
#      'BRCA2_Sahu_2025_HDR_clinvar_2018': ('2c', 'avg'),
#      'BRCA2_unpublished': ('3c', 'avg'),
#      'BRCA2_unpublished_clinvar_2018': ('3c', 'avg'),
#      'CALM1_CALM2_CALM3_Weile_2017': ('2c', 'avg'),
#      'CARD11_Meitlis_2020_DMSO_no_introns': ('2c', 'avg'),
#      'CARD11_Meitlis_2020_Ibrutinib_no_introns': ('3c', 'avg'),
#      'CBS_Sun_2020_high_B6': ('3c', 'avg'), # KEEP OLD
#      'CBS_Sun_2020_low_B6': ('2c', 'avg'), 
#      'CHEK2_Gebbia_2024': ('2c', 'avg'),
#      'CRX_Shepherdson_2024': ('2c', 'avg'),
#      'CTCF_unpublished': ('2c', 'avg'),
#      'DDX3X_Radford_2023_cLFC_day15': ('2c', 'avg'),
#      'DDX3X_Radford_2023_cLFC_day15_clinvar_2018': ('2c', 'avg'),
#      'F9_Popp_2025_carboxy_F9_specific': ('3c', 'avg'), # KEEP OLD
#      'F9_Popp_2025_carboxy_gla_motif': ('3c', 'avg'), # KEEP OLD
#      'F9_Popp_2025_heavy_chain': ('3c', 'benign'), # KEEP OLD
#      'F9_Popp_2025_light_chain': ('3c', 'benign'), # KEEP OLD
#      'F9_Popp_2025_strep_2': ('3c', 'avg'), # KEEP OLD
#      'FKRP_Ma_2024': ('3c', 'avg'),
#      'G6PD_unpublished': ('3c', 'avg'),
#      'GCK_Gersing_2023_complementation': ('3c', 'avg'),
#      'GCK_Gersing_2024_abundance': ('2c', 'avg'), # KEEP OLD
#      'HMBS_van_Loggerenberg_2023_combined': ('2c', 'avg'),
#      'HMBS_van_Loggerenberg_2023_erythroid': ('3c', 'avg'),
#      'HMBS_van_Loggerenberg_2023_ubquitous': ('3c', 'avg'),
#      'JAG1_Gilbert_2024': ('3c', 'avg'),
#      'KCNE1_Muhammad_2024_absence_of_WT': ('3c', 'avg'), # KEEP OLD
#      'KCNE1_Muhammad_2024_potassium_flux': ('3c', 'avg'),
#      'KCNE1_Muhammad_2024_presence_of_WT': ('3c', 'avg'),
#      'KCNH2_Jiang_2022': ('3c', 'avg'),
#      'KCNH2_Kozek_Glazer_2020': ('3c', 'avg'),
#      'KCNH2_O_Neill_2024_surface_expression': ('3c', 'avg'), # RERUN
#      'KCNQ4_Zheng_2022_current_homozygous': ('2c', 'avg'),
#      'KCNQ4_Zheng_2022_v12_homozygous': ('2c', 'avg'),
#      'LARGE1_Ma_2024': ('3c', 'avg'),
#      'MSH2_Jia_2021': ('2c', 'avg'), # variance in 3c benign sample captures uncertainty of IR better
#      'MSH2_Jia_2021_clinvar_2018': ('2c', 'avg'),
#      'NDUFAF6_Sung_2024': ('3c', 'avg'),
#      'OTC_Lo_2023': ('3c', 'avg'), # KEEP OLD
#      'PALB2_unpublished': ('3c', 'avg'),
#      'PAX6_McDonnell_2024_BLX_geneticin': ('3c', 'avg'),
#      'PAX6_McDonnell_2024_BLX_no_geneticin': ('3c', 'avg'),
#      'PAX6_McDonnell_2024_LE9_geneticin': ('2c', 'avg'), # KEEP OLD
#      'PAX6_McDonnell_2024_LE9_no_geneticin': ('2c', 'avg'), # KEEP OLD
#      'PTEN_Matreyek_2018': ('3c', 'avg'),
#      'PTEN_Matreyek_2018_clinvar_2018': ('3c', 'avg'),
#      'PTEN_Mighell_2018': ('3c', 'avg'),
#      'PTEN_Mighell_2018_clinvar_2018': ('3c', 'avg'),
#      'RAD51C_Olvera-León_2024_z_score_D4_D14': ('3c', 'avg'),
#      'RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018': ('3c', 'avg'),
#      'RAD51D_unpublished': ('3c', 'avg'),
#      'RHO_Wan_2019': ('2c', 'avg'),
#      'SCN5A_Glazer_2020': ('3c', 'avg'),
#      'SCN5A_Ma_2024_current_density': ('2c', 'avg'),
#      'SFPQ_unpublished': ('2c', 'avg'),
#      'SGCB_Li_2023': ('2c', 'avg'),
#      'TARDBP_Bolognesi_Faure_2019': ('2c', 'avg'),
#      'TP53_Boettcher_2019': ('2c', 'avg'),
#      'TP53_Boettcher_2019_clinvar_2018': ('2c', 'avg'),
#      'TP53_Fayer_2021_meta': ('3c', 'avg'),
#      'TP53_Fortuno_2021_Kato_meta': ('2c', 'avg'), # KEEP OLD
#      'TP53_Fortuno_2021_Kato_meta_clinvar_2018': ('2c', 'avg'), # KEEP OLD
#      'TP53_Giacomelli_2018_combined_score': ('3c', 'avg'),
#      'TP53_Giacomelli_2018_combined_score_clinvar_2018': ('3c', 'avg'),
#      'TP53_Giacomelli_2018_p53WT_Nutlin3': ('2c', 'benign'),
#      'TP53_Giacomelli_2018_p53WT_Nutlin3_clinvar_2018': ('2c', 'benign'),
#      'TP53_Giacomelli_2018_p53null_Nutlin3': ('2c', 'avg'),
#      'TP53_Giacomelli_2018_p53null_Nutlin3_clinvar_2018': ('2c', 'avg'),
#      'TP53_Giacomelli_2018_p53null_etoposide': ('3c', 'avg'),
#      'TP53_Giacomelli_2018_p53null_etoposide_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_AIP1nWT': ('3c', 'avg'),
#      'TP53_Kato_2003_AIP1nWT_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_BAXnWT': ('3c', 'avg'), # KEEP OLD
#      'TP53_Kato_2003_BAXnWT_clinvar_2018': ('3c', 'avg'), # KEEP OLD
#      'TP53_Kato_2003_GADD45nWT': ('3c', 'avg'),
#      'TP53_Kato_2003_GADD45nWT_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_MDM2nWT': ('3c', 'avg'), # KEEP OLD
#      'TP53_Kato_2003_MDM2nWT_clinvar_2018': ('3c', 'avg'), # KEEP OLD
#      'TP53_Kato_2003_NOXAnWT': ('3c', 'avg'),
#      'TP53_Kato_2003_NOXAnWT_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_P53R2nWT': ('3c', 'avg'),
#      'TP53_Kato_2003_P53R2nWT_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_WAF1nWT': ('3c', 'avg'),
#      'TP53_Kato_2003_WAF1nWT_clinvar_2018': ('3c', 'avg'),
#      'TP53_Kato_2003_h1433snWT': ('3c', 'avg'),
#      'TP53_Kato_2003_h1433snWT_clinvar_2018': ('3c', 'avg'),
#      'TPK1_Weile_2017': ('2c', 'avg'),
#      'TSC2_rapgap_unpublished': ('3c', 'avg'),
#      'TSC2_tuberin_unpublished': ('3c', 'avg'), # KEEP OLD
#      'VHL_Buckley_2024': ('3c', 'avg'),
#      'VHL_Buckley_2024_clinvar_2018': ('3c', 'avg'),
#      'XRCC2_unpublished': ('2c', 'avg'),
#     }
#     config_data = {k: list(v) for k, v in config_data.items()}
#     json.dump(config_data,f,indent=2)


# In[166]:


# with open('/data/ross/assay_calibration/dataset_configs.json','rt') as f:
#     dataset_configs = json.load(f)
# with open('/data/ross/assay_calibration/auto_dataset_config.json','rt') as f:
#     auto_dataset_config = json.load(f)
with open('/data/ross/assay_calibration/new_dataset_configs.json','rt') as f:
    new_dataset_configs = json.load(f)


# In[167]:


keep_old_list = set()
with open('/data/ross/assay_calibration/keep_old_datasets.txt','r') as f:
    for line in f:
        keep_old_list.add(line.strip())


# In[168]:


internal_dataset_aliases = {
    # "SFPQ_unpublished": None,
    # "RAD51C_Olvera-León_2024_z_score_D4_D14": "RAD51C_Olvera-León_2024_z_score_D4_D14",
    # "CHEK2_Gebbia_2024": "CHK2_Gebbia_2024",
    "F9_Popp_2025_model": None,
    # "SGCB_Li_2023": None,
    # "TARDBP_Bolognesi_Faure_2019": None,
    # "TP53_Fayer_2021_meta": None,
    "MSH2_Scott_2022": "MSH2_Jia_2021",
    # "PTEN_Matreyek_2018": "PTEN_Matreyek_2018_filtered_nonsense"
}


# In[169]:


# SFPQ_unpublished: only gnomAD and synonymous
# RAD51C_Olvera-León_2024_z_score_D4_D14: naming issue?
# CHEK2_Gebbia_2024: naming issue
# F9_Popp_2025_model: invalid
# KCNH2_Kozek_Glazer_2020: fixed
# SGCB_Li_2023: no benign nor synonymous
# TARDBP_Bolognesi_Faure_2019: no benign nor synonymous
# TP53_Fayer_2021_meta: looks like a predictor
# MSH2_Scott_2022: same thing as msh2 jia but with better calibrations


# In[170]:


# problem_values = ["NOT SPECIFIED", "INDETERMINATE"]

# df_sub = df_auth[df_auth["Dataset"].isin(reported_list)]

# # Boolean mask for bad / ambiguous classes
# mask = (
#     df_sub["StandardizedClass"].isin(problem_values)
#     | df_sub["StandardizedClass"].isna()
# )

# # Count occurrences per dataset
# counts = df_sub[mask].groupby("Dataset").size()

# # Datasets with more than one bad entry
# datasets_with_multiple = counts[counts > 1]

# print("Number of datasets with >1 ambiguous StandardizedClass:", len(datasets_with_multiple))
# print("Datasets:", datasets_with_multiple.index.tolist())


# In[171]:


# df_auth[df_auth["Dataset"].isin(reported_list)][df_auth["StandardizedClass"].isna()]["Dataset"].unique()


# In[172]:


# import importlib
# import src.assay_calibration.data_utils.dataset
# importlib.reload(src.assay_calibration.data_utils.dataset)
# from src.assay_calibration.data_utils.dataset import Scoreset

# scoreset = Scoreset(df_auth[df_auth["Dataset"] == "VHL_Buckley_2024"])
# scoreset
# # scoreset.scores
# # scoreset.auth_labels


# In[173]:


# len(scoreset.scores), len(scoreset.sample_assignments), len(scoreset.auth_labels)


# In[174]:


# import matplotlib.pyplot as plt
# import numpy as np


# mask = scoreset.sample_assignments[:, :2].any(axis=1)

# scores = np.array(scoreset.scores[mask])
# labels = np.array(scoreset.auth_labels[mask])

# unique_labels = np.unique(labels)

# plt.figure(figsize=(8, 5))

# for label in unique_labels:
#     mask = labels == label
#     plt.hist(scores[mask], bins=30, alpha=0.6, label=str(label))

# plt.xlabel("Score")
# plt.ylabel("Count")
# plt.title("Histogram of Scores by Auth Label")
# plt.legend()
# plt.tight_layout()
# plt.show()


# In[ ]:





# In[175]:


# dataset_configs = {
#     "ASPA_Grønbæk-Thygesen_2024_abundance": ("3c", "avg"),
#     "ASPA_Grønbæk-Thygesen_2024_toxicity": ("3c", "avg"),
#     "BARD1_unpublished": ("2c", "avg"),
#     "CALM1_CALM2_CALM3_Weile_2017": ("3c", "avg"),
#     "CARD11_Meitlis_2020_DMSO_no_introns": ("3c", "avg"),
#     "CARD11_Meitlis_2020_Ibrutinib_no_introns": ("3c", "avg"),
#     "CBS_Sun_2020_high_B6": ("3c", "avg"),
#     "CBS_Sun_2020_low_B6": ("2c", "avg"),
#     "CHK2_Gebbia_2024": ("2c", "benign"),
#     "CHEK2_Gebbia_2024": ("2c", "benign"),
#     "CRX_Shepherdson_2024": ("2c", "avg"),
#     "CTCF_unpublished": ("2c", "avg"),
#     "F9_Popp_2025_carboxy_F9_specific": ("3c", "avg"),
#     "F9_Popp_2025_carboxy_gla_motif": ("3c", "avg"),
#     "F9_Popp_2025_heavy_chain": ("3c", "benign"),
#     "F9_Popp_2025_light_chain": ("3c", "benign"),
#     "F9_Popp_2025_strep_2": ("3c", "avg"),
#     "FKRP_Ma_2024": ("3c", "avg"),
#     "G6PD_unpublished": ("2c", "avg"),
#     "GCK_Gersing_2023_complementation": ("3c", "avg"),
#     "GCK_Gersing_2024_abundance": ("2c", "avg"),
#     "HMBS_van_Loggerenberg_2023_combined": ("3c", "avg"),
#     "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
#     "HMBS_van_Loggerenberg_2023_ubquitous": ("2c", "avg"),
#     "JAG1_Gilbert_2024": ("3c", "avg"),
#     "KCNE1_Muhammad_2024_absence_of_WT": ("3c", "avg"),
#     "KCNE1_Muhammad_2024_potassium_flux": ("2c", "avg"),
#     "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
#     "KCNH2_Jiang_2022": ("2c", "avg"),
#     "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"),
#     "KCNQ4_Zheng_2022_current_homozygous": ("3c", "avg"),
#     "KCNQ4_Zheng_2022_v12_homozygous": ("2c", "avg"),
#     "LARGE1_Ma_2024": ("3c", "avg"),
#     "MSH2_Jia_2021": ("2c", "avg"),
#     "NDUFAF6_Sung_2024": ("2c", "avg"),
#     "OTC_Lo_2023": ("3c", "avg"),
#     "PALB2_unpublished": ("2c", "avg"),
#     "PAX6_McDonnell_2024_BLX_geneticin": ("3c", "avg"),
#     "PAX6_McDonnell_2024_BLX_no_geneticin": ("3c", "avg"),
#     "PAX6_McDonnell_2024_LE9_geneticin": ("2c", "avg"),
#     "PAX6_McDonnell_2024_LE9_no_geneticin": ("2c", "avg"),
#     "RAD51D_unpublished": ("3c", "avg"),
#     "RHO_Wan_2019": ("2c", "avg"),
#     "SCN5A_Glazer_2020": ("3c", "avg"),
#     "SCN5A_Ma_2024_current_density": ("2c", "avg"),
#     "TP53_Boettcher_2019": ("2c", "avg"),
#     "TP53_Fortuno_2021_Kato_meta": ("2c", "avg"),
#     "TP53_Giacomelli_2018_combined_score": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53WT_Nutlin3": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53null_Nutlin3": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53null_etoposide": ("2c", "avg"),
#     "TP53_Kato_2003_AIP1nWT": ("3c", "avg"),
#     "TP53_Kato_2003_BAXnWT": ("3c", "avg"),
#     "TP53_Kato_2003_GADD45nWT": ("2c", "avg"),
#     "TP53_Kato_2003_MDM2nWT": ("3c", "avg"),
#     "TP53_Kato_2003_NOXAnWT": ("3c", "avg"),
#     "TP53_Kato_2003_P53R2nWT": ("2c", "avg"),
#     "TP53_Kato_2003_WAF1nWT": ("3c", "avg"),
#     "TP53_Kato_2003_h1433snWT": ("3c", "avg"),
#     "TPK1_Weile_2017": ("2c", "avg"),
#     "TSC2_rapgap_unpublished": ("3c", "avg"),
#     "TSC2_tuberin_unpublished": ("3c", "avg"),
#     "XRCC2_unpublished": ("2c", "avg"),
#     "BAP1_Waters_2024": ("3c", "avg"),
#     "BRCA1_Adamovich_2022_Cisplatin": ("2c", "avg"),
#     "BRCA1_Adamovich_2022_HDR": ("2c", "avg"),
#     "BRCA1_Findlay_2018": ("2c", "avg"),
#     "BRCA2_Hu_2024": ("2c", "avg"),
#     "BRCA2_Sahu_2023_exon13_Cisplatin": ("3c", "benign"),
#     "BRCA2_Sahu_2023_exon13_Olaparib": ("2c", "avg"),
#     "BRCA2_Sahu_2023_exon13_SGE": ("3c", "benign"),
#     "BRCA2_Sahu_2023_exon13_global_score": ("2c", "avg"),
#     "BRCA2_Sahu_2025_HDR": ("2c", "avg"),
#     "BRCA2_unpublished": ("3c", "avg"),
#     "DDX3X_Radford_2023_cLFC_day15": ("2c", "benign"),
#     # "PTEN_Matreyek_2018": ("2c", "avg"), # replace with filtered nonsense
#     "PTEN_Mighell_2018": ("2c", "avg"),
#     "RAD51C_Olvera-León_2024_z_score_D4_D14": ("2c", "avg"),
#     "VHL_Buckley_2024": ("2c", "benign"),
#     "BAP1_Waters_2024_clinvar_2018": ("3c", "avg"),
#     "BRCA1_Adamovich_2022_Cisplatin_clinvar_2018": ("2c", "avg"),
#     "BRCA1_Adamovich_2022_HDR_clinvar_2018": ("2c", "avg"),
#     "BRCA1_Findlay_2018_clinvar_2018": ("2c", "avg"),
#     "BRCA2_Hu_2024_clinvar_2018": ("2c", "avg"),
#     "BRCA2_Sahu_2023_exon13_Cisplatin_clinvar_2018": ("3c", "avg"),
#     "BRCA2_Sahu_2023_exon13_Olaparib_clinvar_2018": ("2c", "avg"),
#     "BRCA2_Sahu_2023_exon13_SGE_clinvar_2018": ("3c", "avg"),
#     "BRCA2_Sahu_2023_exon13_global_score_clinvar_2018": ("3c", "avg"),
#     "BRCA2_Sahu_2025_HDR_clinvar_2018": ("2c", "avg"),
#     "BRCA2_unpublished_clinvar_2018": ("2c", "avg"),
#     "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
#     # "PTEN_Matreyek_2018_clinvar_2018": ("2c", "avg"), # replace with filtered nonsense
#     "PTEN_Mighell_2018_clinvar_2018": ("2c", "avg"),
#     "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("2c", "avg"),
#     "VHL_Buckley_2024_clinvar_2018": ("2c", "benign"),
#     "KCNH2_Kozek_Glazer_2020": ("3c", "avg"),
#     "TP53_Fayer_2021_meta": ("2c", "avg"),
#     "TARDBP_Bolognesi_Faure_2019": ("3c", "avg"),
#     "SGCB_Li_2023": ("2c", "avg"),
#     "SFPQ_unpublished": ("2c", "avg")
# }

# dataset_relax_configs = {
#     "BARD1_unpublished": ("2c", "avg"),
#     "DDX3X_Radford_2023_cLFC_day15": ("2c", "avg"),
#     "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
#     # "FKRP_Ma_2024": ("3c", "avg"),
#     # "G6PD_unpublished": ("3c", "avg"),
#     "HMBS_van_Loggerenberg_2023_combined": ("2c", "avg"),
#     "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
#     "HMBS_van_Loggerenberg_2023_ubquitous": ("3c", "avg"),
#     "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
#     # "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"), # KEEP CONSTRAINED 3c avg # only compute lr+ when densities are high enough, extend
#     # "LARGE1_Ma_2024": ("3c", "avg"), # KEEP CONSTRAINED
#     "PTEN_Matreyek_2018_filtered_nonsense": ("2c", "avg"),
#     "PTEN_Matreyek_2018_filtered_nonsense_clinvar_2018": ("2c", "avg"),
#     "RAD51C_Olvera-León_2024_z_score_D4_D14": ("3c", "avg"), # ENFORCE 10e-3 on normalized densities 0-1 
#     "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("3c", "avg"),
#     "RAD51D_unpublished": ("2c", "avg"),
#     # "TSC2_Calhoun_cliPE_unpublished": ("2c", "avg"),
#     # "TSC2_Calhoun_immuneSGE_unpublished": ("2c", "avg"),
#     "TSC2_tuberin_unpublished": ("3c", "avg"),
#     "VHL_Buckley_2024": ("3c", "avg"),
#     "VHL_Buckley_2024_clinvar_2018": ("3c", "avg"),
#     # "XRCC2_unpublished": ("2c", "avg"), 
    
#     # new 11_22_25 rerun datasets (all relax)
#     "CALM1_CALM2_CALM3_Weile_2017": ("3c", "avg"),
#     "G6PD_unpublished": ("3c", "avg"),
#     "MSH2_Jia_2021_clinvar_2018": ("2c", "avg"),
#     "TP53_Boettcher_2019_clinvar_2018": ("2c", "avg"),
#     "TP53_Fortuno_2021_Kato_meta_clinvar_2018": ("2c", "avg"),
#     "TP53_Giacomelli_2018_combined_score_clinvar_2018": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53WT_Nutlin3_clinvar_2018": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53null_Nutlin3_clinvar_2018": ("2c", "avg"),
#     "TP53_Giacomelli_2018_p53null_etoposide_clinvar_2018": ("2c", "avg"),
#     "TP53_Kato_2003_AIP1nWT_clinvar_2018": ("3c", "avg"),
#     "TP53_Kato_2003_BAXnWT_clinvar_2018": ("3c", "avg"),
#     "TP53_Kato_2003_GADD45nWT_clinvar_2018": ("2c", "avg"),
#     "TP53_Kato_2003_MDM2nWT_clinvar_2018": ("3c", "avg"),
#     "TP53_Kato_2003_NOXAnWT_clinvar_2018": ("3c", "avg"),
#     "TP53_Kato_2003_P53R2nWT_clinvar_2018": ("2c", "avg"),
#     "TP53_Kato_2003_WAF1nWT_clinvar_2018": ("3c", "avg"),
#     "TP53_Kato_2003_h1433snWT_clinvar_2018": ("3c", "avg"),
#     "TPK1_Weile_2017": ("2c", "avg"),
#     "XRCC2_unpublished": ("2c", "avg"),
#     "LARGE1_Ma_2024": ("3c", "avg"),
#     "FKRP_Ma_2024": ("3c", "avg"),
# }


# In[176]:


from src.assay_calibration.plot_utils.utils import load_dataset_for_plot, import_dataset_configurations

def load_scoreset(df, dataset, clinvar_release="2025", for_fit=False):
    return Scoreset(df[df["Dataset"] == dataset.replace("_clinvar_2018","")], clinvar_release=clinvar_release, synonymous_exclusive=for_fit, filter_nonsense=for_fit and dataset.startswith("PTEN_Matreyek"))

_, dataset_relax_configs, new_dataset_configs, keep_old_list = import_dataset_configurations()


# In[177]:


import re
def make_variant_id(v):
    return f"{v.Gene}_{v.Chrom}_{v.hgvs_c}"

def remove_prefix_with_var_id(s):
    """Remove everything up to and including '_varXXXX_' pattern."""
    return re.sub(r'^.*?_var\d+_', '', s)
    
with open("/data/ross/assay_calibration/variant_to_oob_points_full.pkl","rb") as f:
    variant_to_oob_points = pickle.load(f)

for dataset in variant_to_oob_points:
    if variant_to_oob_points[dataset] is not None:
        variant_to_oob_points[dataset] = {remove_prefix_with_var_id(key): val for key,val in variant_to_oob_points[dataset].items()}


# In[178]:


datasets_auth_temp = df_final.Dataset.unique()
for dataset in reported_list:
    if dataset not in datasets_auth_temp:
        print(dataset)
    if dataset not in variant_to_oob_points or variant_to_oob_points[dataset] is None:
        print(dataset)


# In[179]:


# import importlib
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import src.assay_calibration.data_utils.dataset
# importlib.reload(src.assay_calibration.data_utils.dataset)
# from src.assay_calibration.data_utils.dataset import Scoreset

# def flatten_point_ranges(point_ranges):
#     """Flatten 2D arrays in point_ranges to 1D, asserting only one or zero arrays."""
#     flattened = {}
#     for key, ranges in point_ranges.items():
#         if len(ranges) == 0:
#             flattened[key] = []
#         elif len(ranges) == 1:
#             assert len(ranges[0]) == 2, f"Expected 2 values in range for key {key}, got {ranges[0]}"
#             flattened[key] = ranges[0]
#         else:
#             raise AssertionError(f"Expected 0 or 1 range for key {key}, got {len(ranges)}")
#     return flattened

# def assign_points(assay_score, point_ranges):
#     """Assign points based on which range the assay_score falls into."""
#     if assay_score is None or pd.isna(assay_score):
#         return None
    
#     matched_points = []
#     for point_str, range_vals in point_ranges.items():
#         if len(range_vals) == 2:
#             low, high = range_vals
#             if low <= assay_score <= high:
#                 matched_points.append(int(point_str))
    
#     assert len(matched_points) <= 1, f"Score {assay_score} matched multiple ranges: {matched_points}, point ranges: {point_ranges}"
#     return matched_points[0] if matched_points else 0

# def normalize_author_label(lbl):
#     """Normalize author labels to 'Normal', 'Abnormal', 'IR'"""
#     if lbl is None or pd.isna(lbl):
#         return 'IR'
#     lbl = str(lbl).strip().lower()
#     if lbl == 'normal':
#         return 'Normal'
#     if lbl == 'abnormal':
#         return 'Abnormal'
#     return 'IR'   # anything else


# def plot_three_way_comparison(dataset, scoreset, calibration_f, variant_to_oob_points, 
#                                figsize=(14, 12)):
#     """
#     Compare In-Bag vs OOB vs Author against ClinVar ground truth.
#     Vertically stacked plots for threshold comparison.
#     P/LP in red, B/LB in blue, on the same axis.
#     """
    
#     print("\n" + "="*80)
#     print(f"DATASET: {dataset}")
#     print("="*80)
    
#     # Load calibration
#     with open(calibration_f, 'r') as f:
#         calibration_data = json.load(f)
    
#     if calibration_data["point_ranges"] is None:
#         raise ValueError(f"Dataset uncalibratable")
    
#     point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
#     print(f"\nPoint Ranges: {point_ranges}")
    
#     # Get P/LP and B/LB variants (ClinVar ground truth)
#     plp_mask = scoreset.sample_assignments[:, 0]
#     blb_mask = scoreset.sample_assignments[:, 1]
    
#     plp_scores = scoreset.scores[plp_mask]
#     blb_scores = scoreset.scores[blb_mask]
#     plp_auth = scoreset.auth_labels[plp_mask]
#     blb_auth = scoreset.auth_labels[blb_mask]
#     plp_indices = np.where(plp_mask)[0]
#     blb_indices = np.where(blb_mask)[0]
    
#     print(f"\nTotal variants: {len(scoreset.scores)}")
#     print(f"P/LP variants: {len(plp_scores)}")
#     print(f"B/LB variants: {len(blb_scores)}")
#     print(f"Score range: [{scoreset.scores.min():.3f}, {scoreset.scores.max():.3f}]")
    
#     # BUILD CORRECT INDEX MAPPING
#     variants_by_id = scoreset.get_variants_by_id()
#     kept_idx_to_variant_id = {}
#     kept_idx = 0
    
#     for all_idx, (variant_id, variants) in enumerate(variants_by_id.items()):
#         if scoreset._keep_mask[all_idx]:
#             kept_idx_to_variant_id[kept_idx] = make_variant_id(variants[0])
#             kept_idx += 1
    
#     # Normalize author labels
#     plp_auth_norm = np.array([normalize_author_label(l) for l in plp_auth])
#     blb_auth_norm = np.array([normalize_author_label(l) for l in blb_auth])
    
#     # Classify each method for P/LP variants
#     plp_inbag, plp_oob, plp_author = [], [], []
#     plp_oob_matched_count = 0

    
#     for idx, score in zip(plp_indices, plp_scores):
#         # In-bag
#         points_inbag = assign_points(score, point_ranges)
#         plp_inbag.append('Abnormal' if points_inbag > 0 else ('Normal' if points_inbag < 0 else 'IR'))
        
#         # OOB - USE CORRECT MAPPING
#         variant_id = kept_idx_to_variant_id[idx]  # Fixed!
#         if variant_id in variant_to_oob_points:
#             points_oob = variant_to_oob_points[variant_id]['points']
#             plp_oob.append('Abnormal' if points_oob > 0 else ('Normal' if points_oob < 0 else 'IR'))
#             plp_oob_matched_count += 1
#         else:
#             plp_oob.append('IR')  # No OOB data
        
#         # Author
#         plp_author.append(plp_auth_norm[len(plp_author)])
    
#     # Classify each method for B/LB variants
#     blb_inbag, blb_oob, blb_author = [], [], []
#     blb_oob_matched_count = 0
    
#     for idx, score in zip(blb_indices, blb_scores):
#         # In-bag
#         points_inbag = assign_points(score, point_ranges)
#         blb_inbag.append('Abnormal' if points_inbag > 0 else ('Normal' if points_inbag < 0 else 'IR'))
        
#         # OOB - USE VARIANT ID LOOKUP
#         variant_id = kept_idx_to_variant_id[idx]
#         if variant_id in variant_to_oob_points:
#             points_oob = variant_to_oob_points[variant_id]['points']
#             blb_oob.append('Abnormal' if points_oob > 0 else ('Normal' if points_oob < 0 else 'IR'))
#             blb_oob_matched_count += 1
#         else:
#             blb_oob.append('IR')  # No OOB data
        
#         # Author
#         blb_author.append(blb_auth_norm[len(blb_author)])
    
#     print(f"\nOOB Matching:")
#     print(f"  P/LP: {plp_oob_matched_count}/{len(plp_scores)} ({100*plp_oob_matched_count/len(plp_scores):.1f}%)")
#     print(f"  B/LB: {blb_oob_matched_count}/{len(blb_scores)} ({100*blb_oob_matched_count/len(blb_scores):.1f}%)")
    
#     # Convert to arrays
#     plp_inbag, plp_oob, plp_author = np.array(plp_inbag), np.array(plp_oob), np.array(plp_author)
#     blb_inbag, blb_oob, blb_author = np.array(blb_inbag), np.array(blb_oob), np.array(blb_author)
    
#     # Print classification breakdown
#     print("\n" + "-"*80)
#     print("CLASSIFICATION BREAKDOWN (P/LP variants - should be Abnormal)")
#     print("-"*80)
#     methods = ['In-Bag', 'OOB', 'Author']
#     plp_results = [plp_inbag, plp_oob, plp_author]
    
#     for method, results in zip(methods, plp_results):
#         correct = sum(results == 'Abnormal')
#         wrong = sum(results == 'Normal')
#         ir = sum(results == 'IR')
#         total = len(results)
#         precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0
#         recall = correct / total if total > 0 else 0
        
#         print(f"\n{method}:")
#         print(f"  Correct (Abnormal): {correct}/{total} ({100*correct/total:.1f}%)")
#         print(f"  Wrong (Normal):     {wrong}/{total} ({100*wrong/total:.1f}%)")
#         print(f"  IR:                 {ir}/{total} ({100*ir/total:.1f}%)")
#         print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
    
#     print("\n" + "-"*80)
#     print("CLASSIFICATION BREAKDOWN (B/LB variants - should be Normal)")
#     print("-"*80)
#     blb_results = [blb_inbag, blb_oob, blb_author]
    
#     for method, results in zip(methods, blb_results):
#         correct = sum(results == 'Normal')
#         wrong = sum(results == 'Abnormal')
#         ir = sum(results == 'IR')
#         total = len(results)
#         precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0
#         recall = correct / total if total > 0 else 0
        
#         print(f"\n{method}:")
#         print(f"  Correct (Normal):   {correct}/{total} ({100*correct/total:.1f}%)")
#         print(f"  Wrong (Abnormal):   {wrong}/{total} ({100*wrong/total:.1f}%)")
#         print(f"  IR:                 {ir}/{total} ({100*ir/total:.1f}%)")
#         print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")

    
#     # Create figure with vertical stacking
#     fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
#     # Colors for ClinVar groups
#     plp_color = '#E74C3C'  # Red for P/LP
#     blb_color = '#3498DB'  # Blue for B/LB
    
#     # Marker styles for correctness
#     markers = {
#         'correct': 'o',    # circle
#         'wrong': 'x',      # X
#         'IR': 's'          # square
#     }
    
#     y_pos = 0  # All points on same y-axis position
    
#     for ax, method, plp_res, blb_res in zip(axes, methods, plp_results, blb_results):
        
#         # P/LP variants (should be Abnormal = correct)
#         plp_correct = plp_res == 'Abnormal'
#         plp_wrong = plp_res == 'Normal'
#         plp_ir = plp_res == 'IR'
        
#         # Plot P/LP with different markers for correctness
#         if np.any(plp_correct):
#             ax.scatter(plp_scores[plp_correct], np.ones(sum(plp_correct)) * y_pos, 
#                       c=plp_color, s=80, alpha=0.7, marker=markers['correct'], 
#                       edgecolors='black', linewidths=0.5,
#                       label=f'P/LP Correct ({sum(plp_correct)})')
#         if np.any(plp_wrong):
#             ax.scatter(plp_scores[plp_wrong], np.ones(sum(plp_wrong)) * y_pos, 
#                       c=plp_color, s=80, alpha=0.7, marker=markers['wrong'],
#                       edgecolors='black', linewidths=0.5,
#                       label=f'P/LP Wrong ({sum(plp_wrong)})')
#         if np.any(plp_ir):
#             ax.scatter(plp_scores[plp_ir], np.ones(sum(plp_ir)) * y_pos, 
#                       c=plp_color, s=80, alpha=0.4, marker=markers['IR'],
#                       edgecolors='black', linewidths=0.5,
#                       label=f'P/LP IR ({sum(plp_ir)})')
        
#         # B/LB variants (should be Normal = correct)
#         blb_correct = blb_res == 'Normal'
#         blb_wrong = blb_res == 'Abnormal'
#         blb_ir = blb_res == 'IR'
        
#         # Plot B/LB with different markers for correctness
#         if np.any(blb_correct):
#             ax.scatter(blb_scores[blb_correct], np.ones(sum(blb_correct)) * y_pos, 
#                       c=blb_color, s=80, alpha=0.7, marker=markers['correct'],
#                       edgecolors='black', linewidths=0.5,
#                       label=f'B/LB Correct ({sum(blb_correct)})')
#         if np.any(blb_wrong):
#             ax.scatter(blb_scores[blb_wrong], np.ones(sum(blb_wrong)) * y_pos, 
#                       c=blb_color, s=80, alpha=0.7, marker=markers['wrong'],
#                       edgecolors='black', linewidths=0.5,
#                       label=f'B/LB Wrong ({sum(blb_wrong)})')
#         if np.any(blb_ir):
#             ax.scatter(blb_scores[blb_ir], np.ones(sum(blb_ir)) * y_pos, 
#                       c=blb_color, s=80, alpha=0.4, marker=markers['IR'],
#                       edgecolors='black', linewidths=0.5,
#                       label=f'B/LB IR ({sum(blb_ir)})')
        
#         # Calculate stats
#         plp_precision = sum(plp_correct) / (sum(plp_correct) + sum(plp_wrong)) if (sum(plp_correct) + sum(plp_wrong)) > 0 else 0
#         plp_recall = sum(plp_correct) / len(plp_res) if len(plp_res) > 0 else 0
#         blb_precision = sum(blb_correct) / (sum(blb_correct) + sum(blb_wrong)) if (sum(blb_correct) + sum(blb_wrong)) > 0 else 0
#         blb_recall = sum(blb_correct) / len(blb_res) if len(blb_res) > 0 else 0
        
#         # Format subplot
#         ax.set_yticks([y_pos])
#         ax.set_yticklabels([''])
#         ax.set_ylabel(f'{method}', fontsize=12, fontweight='bold', rotation=0, 
#                      ha='right', va='center')
#         ax.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.9)
#         ax.grid(True, alpha=0.3, axis='x')
#         ax.set_ylim(-0.2, 0.2)
        
#         # Add stats as text
#         stats_text = f'P/LP: Prec={plp_precision:.2f}, Rec={plp_recall:.2f} | B/LB: Prec={blb_precision:.2f}, Rec={blb_recall:.2f}'
#         ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
#                fontsize=9, va='top', ha='right',
#                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     # Only show x-label on bottom plot
#     axes[-1].set_xlabel('Score', fontsize=12)
    
#     # Add overall title
#     plt.suptitle(f'{dataset}: Three-Way Comparison vs ClinVar Ground Truth\n(Red=P/LP, Blue=B/LB; Circle=Correct, X=Wrong, Square=IR)', 
#                 fontsize=13, fontweight='bold', y=0.995)
    
#     plt.tight_layout()
    
#     return fig

# # dataset = "BRCA1_Findlay_2018"
# # mode = "point_assignment_replicate_run"
# # points_save_dir = f'/data/ross/assay_calibration/{mode}/{dataset}'
# # scoreset = Scoreset(df_final[df_final["Dataset"] == dataset])

    
# # with open("/data/ross/assay_calibration/variant_to_oob_points.pkl","rb") as f:
# #     variant_to_oob_points = pickle.load(f)

# # fig = plot_inbag_vs_oob_comparison(
# #     dataset, scoreset, indv_summary, 
# #     variant_to_oob_points[dataset]
# # )
# # plt.show()


# In[180]:


# with open("/data/ross/assay_calibration/variant_to_oob_points_lib.pkl","rb") as f:
#     variant_to_oob_points = pickle.load(f)

# print(len(variant_to_oob_points))

# for dataset in variant_to_oob_points:
#     if variant_to_oob_points[dataset] is None:
#         print(dataset)


# In[181]:


import importlib
import src.assay_calibration.plot_utils.utils
import src.assay_calibration.data_utils.dataset
importlib.reload(src.assay_calibration.plot_utils.utils)
importlib.reload(src.assay_calibration.data_utils.dataset)
from src.assay_calibration.plot_utils.utils import calculate_confusion_mat, plot_confusion_mat, flatten_point_ranges, assign_points
from src.assay_calibration.data_utils.dataset import Scoreset
import re


plot_save_dir = '/data/ross/assay_calibration/author_confusion_mat'


def process_dataset(dataset, config, plot_save_dir, dataset_std_name, relax=False, keep_old=False):

    # scoreset_auth_labels = Scoreset(df_auth[df_auth["Dataset"] == dataset_std_name])
    # var_to_auth = {make_variant_id(v): v.auth_label for v in scoreset_auth_labels.variants}

    point_values = [1,2,3,4,5,6,7,8]

    relax = True
    mode = "point_assignment_replicate_run"
    if keep_old:
        if dataset in dataset_relax_configs:
            relax = True
            mode = "point_assignment_semifinal_rerun"
        else:
            relax = False
            mode = "point_assignment_comparison"
            if dataset.endswith('_clinvar_2018'):
                mode = "point_assignment_clinvar_2018"

    # print(dataset, mode)

    use_median_prior, use_2c_equation = True, False
    n_c, benign_method = config

    # use 2018 fits but plot confusion matrix on all P/LP & B/LB
    points_2018 = False
    genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
    for gene in genes_2018:
        if gene in dataset:
            dataset = dataset+"_clinvar_2018"
            points_2018 = True
            break
    points_save_dir = f'/data/ross/assay_calibration/{mode}/{dataset}'
    
    os.makedirs(plot_save_dir, exist_ok=True)

    experiment_code = f'{dataset}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
    
    pkl_filepath = f'{points_save_dir}/{experiment_code}.pkl'

    if not os.path.exists(pkl_filepath):
        pkl_filepath = pkl_filepath.replace("_avg","")
        benign_method = "benign"
        assert os.path.exists(pkl_filepath), f"{pkl_filepath} does not exist"

    log_f = pkl_filepath.replace('.pkl','.log')
    scoreset_flipped = False
    with open(log_f,'r') as f:
        for line in f:
            if line.strip() == "scoreset_flipped: True": 
                scoreset_flipped = True
    
    with open(pkl_filepath,'rb') as f:
        scoreset_from_fit, indv_summary, fits, score_range, _, n_c = pickle.load(f)


    if "MSH2" in dataset:
        dataset = dataset.replace("Scott_2022","Jia_2021")

    scoreset = load_scoreset(df_final, dataset, clinvar_release="2025" if not points_2018 else "2018", for_fit=False) # get all variants

    
    # scoreset = Scoreset(df_final[df_final["Dataset"] == dataset.replace("_clinvar_2018","")], 
    #                     clinvar_release="2018" if points_2018 else "2025",
    #                     synonymous_exclusive=False) # THIS WILL KEEP SYNONYMOUS BENIGNS AS BENIGN AS WELL
    
    
    sample_names = [s[1] for s in scoreset.samples]
    if not ("Pathogenic/Likely Pathogenic" in sample_names and "Benign/Likely Benign" in sample_names):
        return (None, None), (None, None)

    assert sample_names[0] == "Pathogenic/Likely Pathogenic" and sample_names[1] == "Benign/Likely Benign"

    assert "Abnormal" in scoreset.auth_labels or "Normal" in scoreset.auth_labels, f"{dataset} does not contain valid author labels - Raw: {df_final[df_final['Dataset'] == dataset_std_name]['auth_reported_func_class'].unique()}, Loaded: {np.unique(scoreset.auth_labels)}"

    # print(scoreset.auth_labels[:5])
    # print([v.ID for v in scoreset.variants[:5]])
    # print(list(var_to_auth.keys())[:5])
    
    n_samples = len([s for s in scoreset.samples])

    calibration_f = f"/data/ross/assay_calibration/calibrations_12_25_25/{dataset.replace('Scott_2022','Jia_2021')}.json"

    
    to_return_inbag = calculate_confusion_mat(dataset, scoreset, calibration_f, verbose=False)#indv_summary, fits, score_range, 
                                    # f'({benign_method})', n_c, n_samples, relax=relax, flipped=scoreset_flipped, verbose=True)
    to_return_oob = calculate_confusion_mat_oob(dataset, scoreset, variant_to_oob_points[dataset.replace('Scott_2022','Jia_2021')], calibration_f, fits, score_range, 
                                    f'({benign_method})', n_c, n_samples, relax=relax, flipped=scoreset_flipped, debug=False, verbose=False)

    if to_return_inbag[0] is None:
        print(f"{dataset} in bag result is none")
    
    # fig = plot_three_way_comparison(
    #     dataset, scoreset, calibration_f, 
    #     variant_to_oob_points[dataset]
    # )
    # plt.savefig(f'/data/ross/assay_calibration/oob_comparisons/{dataset}_oob_comparison.png')
    # plt.show()
    
    # print('auth labels match:',len(scoreset.scores) == len(scoreset.auth_labels))
    assert len(scoreset.scores) == len(scoreset.auth_labels), f"{dataset} does not match auth labels"
    # print(dataset,"done")
    return to_return_inbag, to_return_oob



def calculate_confusion_mat_oob(dataset, scoreset, variant_to_oob_points, calibration_f,
                                 fits, score_range, config, n_c, n_samples, 
                                 relax=False, flipped=False, debug=False, verbose=True):
    
    if pd.isna(scoreset.auth_labels).all():
        print(f"{dataset}: all auth labels are nan")
        return None, None
    
    # Build variant_id list for KEPT variants only
    variants_by_id = scoreset.get_variants_by_id()
    kept_variant_ids = []
    
    for idx, (variant_id, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[idx]:
            kept_variant_ids.append(make_variant_id(variants[0]))
    
    assert len(kept_variant_ids) == len(scoreset.scores), \
        f"Mismatch: {len(kept_variant_ids)} kept IDs vs {len(scoreset.scores)} scores"

    
    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]
    
    plp_auth_labels = scoreset.auth_labels[plp_mask]
    blb_auth_labels = scoreset.auth_labels[blb_mask]
    
    blb_in_benign, blb_in_ir, blb_in_path = 0, 0, 0
    plp_in_benign, plp_in_ir, plp_in_path = 0, 0, 0
    
    matched_count = 0

    
    for idx in range(len(scoreset.scores)):
        # Now idx aligns with kept_variant_ids
        variant_id = kept_variant_ids[idx]
        
        if variant_id in variant_to_oob_points:
            oob_data = variant_to_oob_points[variant_id]
            points = oob_data['points']
            matched_count += 1
        else:
            points = 0  # No OOB evidence, fall back to standard (i call "in-bag" but are still OOB) thresholds since it was not used in fitting
            with open(calibration_f, 'r') as f:
                calibration_data = json.load(f)
        
            if calibration_data["point_ranges"] is None:
                raise ValueError(f"  ERROR: Dataset was uncalibratable {calibration_f}")
                return None, None
            
            point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
            points = assign_points(scoreset.scores[idx], point_ranges)
        
        # Classify
        if points <= -1:
            category = 'benign'
        elif points >= 1:
            category = 'path'
        else:
            category = 'ir'
        
        # Count
        if blb_mask[idx]:
            if category == 'benign':
                blb_in_benign += 1
            elif category == 'path':
                blb_in_path += 1
            else:
                blb_in_ir += 1
        
        if plp_mask[idx]:
            if category == 'benign':
                plp_in_benign += 1
            elif category == 'path':
                plp_in_path += 1
            else:
                plp_in_ir += 1
    
    danz = pd.DataFrame({
        'BLB': [blb_in_benign, blb_in_ir, blb_in_path],
        'PLP': [plp_in_benign, plp_in_ir, plp_in_path]
    }).T
    
    indeterminate_codes = ["NOT SPECIFIED", "INDETERMINATE", "IGNORE"]
    
    blb_norm = (np.char.upper(blb_auth_labels) == "NORMAL").sum()
    blb_ir = (np.isin(np.char.upper(blb_auth_labels), indeterminate_codes) | pd.isna(blb_auth_labels)).sum()
    blb_abnorm = (np.char.upper(blb_auth_labels) == "ABNORMAL").sum()
    
    plp_norm = (np.char.upper(plp_auth_labels) == "NORMAL").sum()
    plp_ir = (np.isin(np.char.upper(plp_auth_labels), indeterminate_codes) | pd.isna(plp_auth_labels)).sum()
    plp_abnorm = (np.char.upper(plp_auth_labels) == "ABNORMAL").sum()
    
    auth = pd.DataFrame({
        'BLB': [blb_norm, blb_ir, blb_abnorm],
        'PLP': [plp_norm, plp_ir, plp_abnorm]
    }).T
    
    ind = ['Normal', 'IR', 'Abnormal']
    danz.columns = ind
    auth.columns = ind
    
    if verbose:
        print(f"\n{dataset}")
        print(f"  Matched OOB: {matched_count}/{len(scoreset.scores)} ({100*matched_count/len(scoreset.scores):.1f}%)")
        print(f"  P/LP total: {plp_mask.sum()}, B/LB total: {blb_mask.sum()}")
        print('danz\n', danz)
        print('auth\n', auth)
    
    return danz, auth


from concurrent.futures import ProcessPoolExecutor, as_completed
import src.assay_calibration.plot_utils.utils
importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import calculate_confusion_mat, plot_confusion_mat
import src.assay_calibration.data_utils.dataset
importlib.reload(src.assay_calibration.data_utils.dataset)
from src.assay_calibration.data_utils.dataset import Scoreset

def _process_single_dataset(
    dataset,
    reported_list,
    internal_dataset_aliases,
    dataset_configs,
    plot_save_dir,
    keep_old,
):
    dataset_std_name = dataset

    # alias handling
    if dataset in internal_dataset_aliases:
        dataset = internal_dataset_aliases[dataset]
        if dataset is None:
            return None

    if dataset not in dataset_configs:
        print(f"\n{dataset} not in dataset_configs\n")
        return None

    config = dataset_configs[dataset]

    try:
        (danz, auth), (danz_oob, auth_oob) = process_dataset(
            dataset, config, plot_save_dir, dataset_std_name
        )
    except AssertionError as e:
        print(f"\n{dataset}: {e}\n")
        return None

    if danz is None or auth is None:# or danz_oob is None or auth_oob is None:
        print(f"{dataset} danz or auth is None")
        return None

    return dataset, danz, auth, danz_oob, auth_oob


def process_datasets_parallel(
    reported_list,
    internal_dataset_aliases,
    dataset_configs,
    plot_save_dir,
    keep_old_list,
    max_workers=None,
):
    datasets, danzs, auths, danzs_oob, auths_oob = [],[],[],[],[]

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                _process_single_dataset,
                dataset,
                reported_list,
                internal_dataset_aliases,
                dataset_configs,
                plot_save_dir,
                keep_old=dataset in keep_old_list
            )
            for dataset in reported_list
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            dataset, danz, auth, danz_oob, auth_oob = result
            datasets.append(dataset)
            danzs.append(danz)
            auths.append(auth)
            danzs_oob.append(danz_oob)
            auths_oob.append(auth_oob)

    return datasets, danzs, auths, danzs_oob, auths_oob

datasets_new_configs, danzs_new_configs, auths_new_configs, danzs_new_configs_oob, auths_new_configs_oob = process_datasets_parallel(
    reported_list,
    internal_dataset_aliases,
    new_dataset_configs,
    plot_save_dir,
    keep_old_list,
    max_workers=len(reported_list),  # or None for default
)


# In[182]:


datasets_for_coverage = [d.replace("MSH2_Jia_2021","MSH2_Scott_2022") for d in datasets_new_configs]


# In[183]:


def _process_single_dataset(
    dataset,
    reported_list,
    internal_dataset_aliases,
    dataset_configs,
    plot_save_dir,
    keep_old,
):
    dataset_std_name = dataset

    # alias handling
    if dataset in internal_dataset_aliases:
        dataset = internal_dataset_aliases[dataset]
        if dataset is None:
            return None

    if dataset not in dataset_configs:
        print(f"\n{dataset} not in dataset_configs\n")
        return None

    config = dataset_configs[dataset]

    try:
        (danz, auth), (danz_oob, auth_oob) = process_dataset(
            dataset, config, plot_save_dir, dataset_std_name
        )
    except AssertionError as e:
        print(f"\n{dataset}: {e}\n")
        return None

    if danz is None or auth is None:
        print(f"{dataset} danz or auth is None")
        return None

    return dataset, danz, auth, danz_oob, auth_oob


def process_datasets_parallel_with_coverage(
    reported_list,
    internal_dataset_aliases,
    dataset_configs,
    plot_save_dir,
    keep_old_list,
    df_final,
    variant_to_oob_points,
    make_variant_id,
    max_workers=None,
):
    """
    Process datasets in parallel and compute coverage statistics.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import numpy as np
    
    datasets, danzs, auths, danzs_oob, auths_oob = [], [], [], [], []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                _process_single_dataset,
                dataset,
                reported_list,
                internal_dataset_aliases,
                dataset_configs,
                plot_save_dir,
                keep_old=dataset in keep_old_list
            )
            for dataset in reported_list
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            dataset, danz, auth, danz_oob, auth_oob = result
            datasets.append(dataset)
            danzs.append(danz)
            auths.append(auth)
            danzs_oob.append(danz_oob)
            auths_oob.append(auth_oob)

    # Now compute coverage statistics
    print(f"\n{'='*80}")
    print("COMPUTING COVERAGE STATISTICS")
    print('='*80)
    
    # Track assignments for coverage calculation
    all_danz_assignments = []
    all_author_assignments = []
    all_is_vus = []
    all_is_snv = []
    
    # Separate tracking for confusion matrix variants (kept + has P/LP or B/LB)
    confusion_matrix_danz = []
    confusion_matrix_author = []
    
    # Process each dataset to extract variant-level data
    for dataset_std_name in tqdm(datasets, desc="Processing datasets for coverage"):
        # Handle aliases
        dataset = dataset_std_name
        if dataset in internal_dataset_aliases:
            dataset_alias = internal_dataset_aliases[dataset]
            if dataset_alias is None:
                continue
            dataset = dataset_alias
        
        # Handle 2018 datasets
        genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
        use_2018 = any(gene in dataset for gene in genes_2018)
        oob_dataset_key = dataset_std_name
        if use_2018 and not dataset_std_name.endswith('_clinvar_2018'):
            oob_dataset_key = dataset_std_name + "_clinvar_2018"
            dataset = dataset + "_clinvar_2018"
        
        # Get OOB points for this dataset
        variant_to_oob = variant_to_oob_points.get(oob_dataset_key, {})
        if variant_to_oob is None:
            variant_to_oob = {}
        
        # Load calibration
        calibration_f = f"/data/ross/assay_calibration/calibrations_12_25_25/{dataset.replace('Scott_2022','Jia_2021')}.json"
        if not os.path.exists(calibration_f):
            print(f"  {dataset_std_name}: Calibration file not found")
            continue
        
        with open(calibration_f, 'r') as f:
            calibration_data = json.load(f)
        
        if calibration_data["point_ranges"] is None:
            print(f"  {dataset_std_name}: No point_ranges in calibration")
            continue
        
        point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
        
        # Load scoreset
        clinvar_rel = "2018" if use_2018 else "2025"
        
        try:
            scoreset = load_scoreset(df_final, dataset_std_name, clinvar_release=clinvar_rel, for_fit=False)
            # scoreset = Scoreset(
            #     df_final[df_final["Dataset"] == dataset_std_name],
            #     clinvar_release=clinvar_rel,
            #     synonymous_exclusive=False
            # )
        except Exception as e:
            print(f"  {dataset_std_name}: Scoreset error - {e}")
            continue
        
        if pd.isna(scoreset.auth_labels).all():
            print(f"  {dataset_std_name}: All auth labels are NaN")
            continue
        
        # Get kept variants using same logic as calculate_confusion_mat_oob
        variants_by_id = scoreset.get_variants_by_id()
        kept_variant_ids = []
        
        for idx, (variant_id, variants) in enumerate(variants_by_id.items()):
            if scoreset._keep_mask[idx]:
                kept_variant_ids.append(make_variant_id(variants[0]))
        
        # Verify alignment
        if len(kept_variant_ids) != len(scoreset.scores):
            print(f"  {dataset_std_name}: Skipping - kept variant mismatch ({len(kept_variant_ids)} != {len(scoreset.scores)})")
            continue
        
        # Check if dataset has both P/LP and B/LB
        if scoreset.sample_assignments.shape[1] < 2:
            print(f"  {dataset_std_name}: Skipping - insufficient sample types (shape={scoreset.sample_assignments.shape})")
            continue
        
        # Get masks for kept variants only (same as confusion matrix code)
        plp_mask = scoreset.sample_assignments[:, 0]
        blb_mask = scoreset.sample_assignments[:, 1]
        
        n_kept = len(scoreset.scores)
        n_plp = plp_mask.sum()
        n_blb = blb_mask.sum()
        
        print(f"  {dataset_std_name}: kept={n_kept}, P/LP={n_plp}, B/LB={n_blb}")
        
        # FIRST: Process kept variants for P/LP + B/LB coverage (matches confusion matrix exactly)
        added_count = 0
        for idx in range(len(scoreset.scores)):
            variant_id = kept_variant_ids[idx]
            
            # Get DanZ assignment (OOB if available, else in-bag)
            if variant_id in variant_to_oob:
                oob_data = variant_to_oob[variant_id]
                danz_point = oob_data['points']
            else:
                # continue
                score = scoreset.scores[idx]
                danz_point = assign_points(score, point_ranges)
            
            # Get author assignment from scoreset
            auth_label = scoreset.auth_labels[idx]
            if pd.isna(auth_label):
                author_num = 1
            else:
                label_upper = str(auth_label).upper()
                if label_upper == "NORMAL":
                    author_num = 0
                elif label_upper == "ABNORMAL":
                    author_num = 2
                else:
                    author_num = 1
            
            # Track for P/LP + B/LB coverage (using masks like confusion matrix)
            if plp_mask[idx] or blb_mask[idx]:
                confusion_matrix_danz.append(danz_point)
                confusion_matrix_author.append(author_num)
                added_count += 1
        
        print(f"    -> Actually added {added_count} to confusion_matrix")
        
        # SECOND: Process ALL variants for VUS and SNV coverage
        for idx, (variant_id_key, variants) in enumerate(variants_by_id.items()):
            variant = variants[0]
            variant_id = make_variant_id(variant)
            
            # Get DanZ assignment (OOB if available, else in-bag)
            if variant_id in variant_to_oob:
                danz_point = variant_to_oob[variant_id]['points']
            else:
                # continue
                score = variant.auth_reported_score
                danz_point = assign_points(score, point_ranges)
            
            # Get author assignment
            auth_label = variant.auth_label
            if pd.isna(auth_label):
                author_num = 1
            else:
                label_upper = str(auth_label).upper()
                if label_upper == "NORMAL":
                    author_num = 0
                elif label_upper == "ABNORMAL":
                    author_num = 2
                else:
                    author_num = 1
            
            # Track for VUS and SNV only
            is_vus = bool(variant.is_vus)
            is_snv = bool(variant.is_snv)
            
            all_danz_assignments.append(danz_point)
            all_author_assignments.append(author_num)
            all_is_vus.append(is_vus)
            all_is_snv.append(is_snv)
    
    # Convert to arrays
    all_danz_assignments = np.array(all_danz_assignments)
    all_author_assignments = np.array(all_author_assignments)
    all_is_vus = np.array(all_is_vus)
    all_is_snv = np.array(all_is_snv)
    
    confusion_matrix_danz = np.array(confusion_matrix_danz)
    confusion_matrix_author = np.array(confusion_matrix_author)
    
    print(f"\nDEBUG: confusion_matrix_danz length = {len(confusion_matrix_danz)}")
    print(f"DEBUG: confusion_matrix_author length = {len(confusion_matrix_author)}")
    
    # Calculate coverage
    def calculate_coverage(danz_arr, author_arr, mask=None):
        """Calculate coverage for DanZ and Author given a boolean mask (or all if None)"""
        if mask is not None:
            danz_arr = danz_arr[mask]
            author_arr = author_arr[mask]
        
        if len(danz_arr) == 0:
            return None, None
        
        danz_coverage = 100 * (danz_arr != 0).sum() / len(danz_arr)
        author_coverage = 100 * (author_arr != 1).sum() / len(author_arr)
        return danz_coverage, author_coverage
    
    coverage_stats = {}
    
    # Coverage for P/LP + B/LB controls (use confusion matrix variants - kept with P/LP or B/LB)
    danz_cov, auth_cov = calculate_coverage(confusion_matrix_danz, confusion_matrix_author)
    coverage_stats['controls'] = {
        'n': len(confusion_matrix_danz),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Coverage for VUS (use all variants)
    vus_mask = all_is_vus
    danz_cov, auth_cov = calculate_coverage(all_danz_assignments, all_author_assignments, vus_mask)
    coverage_stats['vus'] = {
        'n': vus_mask.sum(),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Coverage for SNVs (use all variants)
    snv_mask = all_is_snv
    danz_cov, auth_cov = calculate_coverage(all_danz_assignments, all_author_assignments, snv_mask)
    coverage_stats['snvs'] = {
        'n': snv_mask.sum(),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Print coverage statistics
    print(f"\n{'='*80}")
    print("COVERAGE STATISTICS (% Non-Indeterminate)")
    print('='*80)
    print(f"Total variants processed: {len(all_danz_assignments):,}")
    
    # Always print controls, even if 0
    stats = coverage_stats['controls']
    print(f"\nP/LP + B/LB Controls (n={stats['n']:,}):")
    if stats['n'] > 0:
        print(f"  DanZ Coverage:   {stats['danz_coverage']:.1f}%")
        print(f"  Author Coverage: {stats['author_coverage']:.1f}%")
    else:
        print(f"  No P/LP or B/LB variants found!")
    
    # Print VUS and SNV
    for group_name, key in [('VUS Variants', 'vus'), ('SNVs', 'snvs')]:
        stats = coverage_stats[key]
        if stats['n'] > 0:
            print(f"\n{group_name} (n={stats['n']:,}):")
            print(f"  DanZ Coverage:   {stats['danz_coverage']:.1f}%")
            print(f"  Author Coverage: {stats['author_coverage']:.1f}%")
    
    print('='*80)
    
    return datasets, danzs, auths, danzs_oob, auths_oob

from tqdm import tqdm
# Usage:
datasets_new_configs_test, danzs_new_configs_test, auths_new_configs_test, danzs_new_configs_oob_test, auths_new_configs_oob_test = \
    process_datasets_parallel_with_coverage(
        datasets_for_coverage,
        internal_dataset_aliases,
        new_dataset_configs,
        plot_save_dir,
        keep_old_list,
        df_final,
        variant_to_oob_points,
        make_variant_id,
        max_workers=len(reported_list),
    )


# _ = \
#     process_datasets_parallel_with_coverage(
#         ["TP53_Fayer_2021_meta"],
#         internal_dataset_aliases,
#         new_dataset_configs,
#         plot_save_dir,
#         keep_old_list,
#         df_final,
#         variant_to_oob_points,
#         make_variant_id,
#         max_workers=1,
#     )


# In[184]:


variant_to_oob_points.keys()


# In[185]:


len(np.unique(np.array([d.split('_')[0] for d in datasets_new_configs])))


# In[186]:


# importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import plot_individual_confusion_matrices_with_metrics, print_aggregate_performance


# # 1. Plot individual confusion matrices with metrics
# fig, metrics_df = plot_individual_confusion_matrices_with_metrics(
#     datasets_new_configs[1:2],
#     danzs_new_configs[1:2],
#     auths_new_configs[1:2],
#     new_dataset_configs
# )

# # plt.savefig('/data/ross/assay_calibration/flagship/individual_confusion_matrices_with_metrics.png', 
# #             dpi=300, bbox_inches='tight')

# # 2. Save individual metrics
# # metrics_df.to_csv('/data/ross/assay_calibration/flagship/individual_metrics.csv', index=False)

# # 3. Print and get aggregate statistics
danz_agg, auth_agg, individual_df = print_aggregate_performance(
    danzs_new_configs_oob,
    auths_new_configs_oob,
    datasets_new_configs
)

# # 4. Save aggregate metrics
# # with open('/data/ross/assay_calibration/flagship/aggregate_metrics.json', 'w') as f:
# #     json.dump({
# #         'danz_aggregate': danz_agg,
# #         'auth_aggregate': auth_agg
# #     }, f, indent=2)

# # print("\nIndividual metrics saved to: individual_metrics.csv")
# # print("Aggregate metrics saved to: aggregate_metrics.json")


# In[187]:


for dataset in reported_list:
    if dataset not in datasets_new_configs:
        print(dataset)


# In[188]:


for dataset in datasets_new_configs:
    if "TP53" in dataset or "F9" in dataset:
        print(dataset)


# In[189]:


len(set(datasets_new_configs)), len(set(reported_list))


# In[190]:


len(danzs_new_configs_oob), len(auths_new_configs_oob)


# In[191]:


with open("/data/ross/assay_calibration/flagship/excalibr_confusion_mat.pkl","wb") as f:
    pickle.dump(danzs_new_configs_oob, f)
with open("/data/ross/assay_calibration/flagship/auth_confusion_mat.pkl","wb") as f:
    pickle.dump(auths_new_configs_oob, f)
with open("/data/ross/assay_calibration/flagship/datasets_confusion_mat.pkl","wb") as f:
    pickle.dump(datasets_new_configs, f)


# In[192]:


with open("/data/ross/assay_calibration/flagship/excalibr_confusion_mat.pkl","rb") as f:
    danzs_new_configs_oob = pickle.load(f)
with open("/data/ross/assay_calibration/flagship/auth_confusion_mat.pkl","rb") as f:
    auths_new_configs_oob = pickle.load(f)
with open("/data/ross/assay_calibration/flagship/datasets_confusion_mat.pkl","rb") as f:
    datasets_new_configs = pickle.load(f)


# In[209]:


# semifull


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import plot_aggregate_confusion_matrices

companion_paper = True
fig, danz_agg_metrics, auth_agg_metrics = plot_aggregate_confusion_matrices(
    danzs_new_configs_oob,
    auths_new_configs_oob,
    datasets_new_configs,
    letters=companion_paper
)

plt.savefig(f'/data/ross/assay_calibration/flagship/aggregate_confusion_matrices_oob{"" if companion_paper else "_noletters"}.png', 
            dpi=300, bbox_inches='tight')
plt.show()


# In[194]:


def fmt_int(x):
    return f"{x:,}"

def fmt_pct(x):
    return f"{100*x:.1f}\\%"

def fmt3(x):
    return f"{x:.3f}"

def fmt1(x):
    return f"{x:.1f}"


def latex_performance_table(danz, auth):
    total = danz["total"]  # same for both by design

    rows = [
        ("Total variants",
         fmt_int(total),
         fmt_int(total)),

        ("Determinate assignments",
         f'{fmt_int(danz["determinate"])} ({fmt_pct(danz["coverage"])})',
         f'{fmt_int(auth["determinate"])} ({fmt_pct(auth["coverage"])})'),

        ("Indeterminate assignments",
         f'{fmt_int(danz["uncertain"])} ({fmt_pct(1-danz["coverage"])})',
         f'{fmt_int(auth["uncertain"])} ({fmt_pct(1-auth["coverage"])})'),

        ("\\midrule", "", ""),

        ("Accuracy",
         fmt3(danz["accuracy"]),
         fmt3(auth["accuracy"])),

        ("Sensitivity",
         fmt3(danz["sensitivity"]),
         fmt3(auth["sensitivity"])),

        ("Specificity",
         fmt3(danz["specificity"]),
         fmt3(auth["specificity"])),

        ("MCC",
         fmt3(danz["mcc"]),
         fmt3(auth["mcc"])),

        ("\\midrule", "", ""),

        ("LR$^+$ (pathogenic vs. benign)",
         fmt1(danz["lr_plus_standard"]),
         fmt1(auth["lr_plus_standard"])),

        ("LR$^+$ (pathogenic vs. rest)",
         fmt1(danz["lr_plus_pathogenic"]),
         fmt1(auth["lr_plus_pathogenic"])),

        ("LR$^+$ (benign vs. rest)",
         fmt1(danz["lr_plus_benign"]),
         fmt1(auth["lr_plus_benign"])),

        ("DOR (pathogenic vs. benign)",
         fmt1(danz["dor_standard"]),
         fmt1(auth["dor_standard"])),

        ("DOR (pathogenic vs. rest)",
         fmt1(danz["dor_pathogenic"]),
         fmt1(auth["dor_pathogenic"])),

        ("DOR (benign vs. rest)",
         fmt1(danz["dor_benign"]),
         fmt1(auth["dor_benign"])),
    ]

    print(r"""\begin{table}[!tb]
\centering
\caption{Performance comparison of out-of-bag \excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants. Pathogenic and benign refer to the direction of evidence assigned.}
\label{tab:author_performance}
\begin{tabular}{lcc}
\toprule
Metric & Calibrated Evidence & Author Annotations \\
\midrule""")

    for r in rows:
        if r[0] == "\\midrule":
            print(r"\midrule")
        else:
            print(f"{r[0]} & {r[1]} & {r[2]} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

latex_performance_table(danz_agg_metrics, auth_agg_metrics)


# In[195]:


# import pandas as pd
# import numpy as np

# def compute_vus_counts(
#     df,
#     datasets_new_configs,
#     group_cols=['Gene', 'Chrom', 'hg38_start', 'ref_allele', 'alt_allele'],
#     filter_nonsense=False
# ):
    
#     # Get genes from dataset names
#     genes_new_configs = [d.split('_')[0] for d in datasets_new_configs]
    
#     # ClinVar star filter
#     star_set = {
#         np.nan,
#         'no assertion criteria provided',
#         'no assertion provided',
#         'no interpretation for the single variant',
#         'no classification provided',
#         '-'
#     }
    
#     # Initial filter
#     df_filtered = df[
#         df['Gene'].isin(genes_new_configs) &
#         (df['clinvar_sig_2025'] == 'Uncertain significance') &
#         (~df['clinvar_star_2025'].isin(star_set))
#     ].copy()
    
#     # Remove flagged variants
#     df_filtered = df_filtered[df_filtered['Flag'] != "*"]
    
#     # Splicing filter
#     if 'splice_measure' in df_filtered.columns and df_filtered['splice_measure'].nunique() == 1:
#         detects_splice = df_filtered['splice_measure'].iloc[0] == "Yes"
#         if not detects_splice:
#             df_filtered = df_filtered[
#                 ~df_filtered['simplified_consequence'].str.lower().isin(
#                     ['splice region', 'splice_site_variant']
#                 )
#             ]
#             # SpliceAI threshold filter
#             for col in ['spliceAI_DS_AG', 'spliceAI_DS_AL', 'spliceAI_DS_DG', 'spliceAI_DS_DL']:
#                 if col in df_filtered.columns:
#                     df_filtered = df_filtered[df_filtered[col] < 0.2]
    
#     # Count unique variants per gene (drop duplicates)
#     unique_variants = df_filtered.drop_duplicates(subset=group_cols)
#     vus_counts_series = unique_variants.groupby('Gene').size()
#     gene_to_vus_count = vus_counts_series.to_dict()
    
#     # Count NaN auth_reported_score
#     num_nan_auth = df_filtered['auth_reported_score'].isna().sum() if 'auth_reported_score' in df_filtered.columns else 0
    
#     # Count variants per dataset
#     dataset_counts = df_filtered['Dataset'].value_counts() if 'Dataset' in df_filtered.columns else pd.Series()
    
#     return gene_to_vus_count, num_nan_auth, dataset_counts

# gene_to_vus_count, num_nan_auth, dataset_vus_counts = compute_vus_counts(
#     df_std,
#     datasets_new_configs,
# )


# In[196]:


# importlib.reload(src.assay_calibration.plot_utils.utils)
# from src.assay_calibration.plot_utils.utils import plot_gene_level_performance_comparison

# fig, dor_dataset_results = plot_gene_level_performance_comparison(
#     danzs_new_configs_oob,
#     auths_new_configs_oob,
#     datasets_new_configs,
#     gene_to_vus_count,  # Dictionary: dataset_name -> VUS count
#     metric='accuracy'  # or 'accuracy'
# )

# plt.savefig(f'/data/ross/assay_calibration/flagship/dor_comparison_scatter.png', 
#             dpi=300, bbox_inches='tight')
# plt.show()


# In[197]:


# importlib.reload(src.assay_calibration.plot_utils.utils)
# from src.assay_calibration.plot_utils.utils import plot_aggregate_confusion_diff_matrix

# fig, danz_agg_metrics, auth_agg_metrics = plot_aggregate_confusion_diff_matrix(
#     cache2,
#     cache3,
#     cache1
# )

# plt.savefig('/data/ross/assay_calibration/flagship/aggregate_confusion_diff_matrix_inbag.png', 
#             dpi=300, bbox_inches='tight')
# plt.show()


# In[198]:


# print([d for d in df_auth.columns if "clinvar" in d])


# In[199]:


# df_auth[df_auth["clinvar_sig_2025"] == "Uncertain significance"]


# In[200]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import load_all_variant_assignments_with_oob

all_danz_oob, all_author, all_clinvar, all_scores, dataset_info_df = load_all_variant_assignments_with_oob(
    dataset_names=datasets_new_configs,
    dataset_configs=new_dataset_configs,
    keep_old_list=keep_old_list,
    internal_dataset_aliases=internal_dataset_aliases,
    df_final=df_final,
    variant_to_oob_points_all=variant_to_oob_points,  # The loaded OOB dict
    make_variant_id=make_variant_id,
    verbose=True
)

# Save for later use
# with open('/data/ross/assay_calibration/flagship/all_variant_assignments.pkl', 'wb') as f:
#     pickle.dump({
#         'danz_assignments': all_danz,
#         'author_annotations': all_author,
#         'clinvar_classes': all_clinvar,
#         'scores': all_scores,
#         'dataset_info': dataset_info_df
#     }, f)


# In[201]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import plot_aggregate_evidence_distributions


# Now create the aggregate evidence distribution plots
fig_author, fig_clinvar = plot_aggregate_evidence_distributions(
    all_danz_oob,
    all_author,
    all_clinvar,
)

fig_author.savefig('/data/ross/assay_calibration/flagship/aggregate_evidence_by_author_oob.png', 
                   dpi=300, bbox_inches='tight')
# fig_clinvar.savefig('/data/ross/assay_calibration/flagship/aggregate_evidence_by_clinvar_oob.png', 
#                     dpi=300, bbox_inches='tight')
plt.close(fig_clinvar)

# Print summary statistics
print("\nDataset Info:")
print(dataset_info_df)
# dataset_info_df.to_csv('/data/ross/assay_calibration/flagship/dataset_variant_counts.csv', index=False)


# In[202]:


with open("/data/ross/assay_calibration/flagship/fig4_evidence.pkl","wb") as f:
    pickle.dump(all_danz_oob, f)
with open("/data/ross/assay_calibration/flagship/fig4_auth_labels.pkl","wb") as f:
    pickle.dump(all_author, f)
with open("/data/ross/assay_calibration/flagship/fig4_clinvar_labels.pkl","wb") as f:
    pickle.dump(all_clinvar, f)


# In[203]:


for dataset in datasets_new_configs:
    if "meta" in dataset.lower():
        print(dataset)


# In[204]:


new_dataset_configs


# In[205]:


genes_2018 = ['BRCA1', 'MSH2', 'PTEN', 'TP53']

full_nonredundant_dataset_list = set()
for dataset in new_dataset_configs:
    # if dataset.split('_')[0] in genes_2018 and "_clinvar_2018" not in dataset:
    #     continue
    # if dataset.split('_')[0] not in genes_2018 and "_clinvar_2018" in dataset:
    #     continue
    if "_clinvar_2018" in dataset: # doesn't expect clinvar 2018 tag
        continue

    if dataset == "SFPQ_unpublished" or dataset == "TP53_Fayer_2021_meta" or dataset == "TSC2_rapgap_unpublished" or dataset == "TSC2_tuberin_unpublished":
        continue

    full_nonredundant_dataset_list.add(dataset)

len(full_nonredundant_dataset_list)


# In[206]:


# with open("/data/ross/assay_calibration/variant_to_oob_points_full.pkl","rb") as f:
#     variant_to_oob_points = pickle.load(f)

# for dataset in variant_to_oob_points:
#     if variant_to_oob_points[dataset] is not None:
#         variant_to_oob_points[dataset] = {remove_prefix_with_var_id(key): val for key,val in variant_to_oob_points[dataset].items()}


# In[212]:


[d for d in full_nonredundant_dataset_list if d not in df_clingen_unique]


# In[213]:


'''
updated all datasets for clinvar not just author ones. can have duplicate variants across dataset, just note that in text
'''

importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import load_all_variant_assignments_with_oob

all_danz_oob_full, all_author_full, all_clinvar_full, all_scores_full, dataset_info_df_full = load_all_variant_assignments_with_oob(
    dataset_names=full_nonredundant_dataset_list,
    dataset_configs=new_dataset_configs,
    keep_old_list=keep_old_list,
    internal_dataset_aliases=internal_dataset_aliases,
    df_final=df_std,
    variant_to_oob_points_all=variant_to_oob_points,  # The loaded OOB dict
    make_variant_id=make_variant_id,
    verbose=True
)


# In[214]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import plot_aggregate_evidence_distributions


# Now create the aggregate evidence distribution plots
fig_author, fig_clinvar = plot_aggregate_evidence_distributions(
    all_danz_oob_full,
    all_author_full,
    all_clinvar_full,
)
plt.close(fig_author)

# do NOT save author one, it has artificial indeterminates
# fig_author.savefig('/data/ross/assay_calibration/flagship/aggregate_evidence_by_author_oob.png', 
#                    dpi=300, bbox_inches='tight')
fig_clinvar.savefig('/data/ross/assay_calibration/flagship/aggregate_evidence_by_clinvar_oob.png', 
                    dpi=300, bbox_inches='tight')

# Print summary statistics
print("\nDataset Info:")
print(dataset_info_df)
# dataset_info_df.to_csv('/data/ross/assay_calibration/flagship/dataset_variant_counts.csv', index=False)


# In[215]:


with open("/data/ross/assay_calibration/flagship/fig5_evidence.pkl","wb") as f:
    pickle.dump(all_danz_oob_full, f)
with open("/data/ross/assay_calibration/flagship/fig5_auth_labels.pkl","wb") as f:
    pickle.dump(all_author_full, f)
with open("/data/ross/assay_calibration/flagship/fig5_clinvar_labels.pkl","wb") as f:
    pickle.dump(all_clinvar_full, f)


# ### combined auth & clinvar distr plot

# In[389]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import plot_combined_evidence_distributions

fig = plot_combined_evidence_distributions(
    all_danz_oob, all_author,
    all_danz_oob_full, all_clinvar_full,
    save_path='/data/ross/assay_calibration/paper/combined_auth_clinvar_evidence_distr.png'
)

plt.show()


# In[486]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import compute_genewise_evidence_table

gwe_author_table, gwe_clinvar_table, gwe_latex_str = compute_genewise_evidence_table(
    all_danz_oob, all_author, dataset_info_df,
    all_danz_oob_full, all_clinvar_full, dataset_info_df_full,
)

print(gwe_latex_str)


# #### ns here are for clinvar 2025 but clinvar 2018 was used for oob evidence

# In[217]:


len(datasets_new_configs), len(set([d.split('_')[0] for d in datasets_new_configs]))


# In[218]:


# print("unique assay type: ",end='')
# print(*df_std['Assay Type'].unique(), sep=', ')

# print("unique model system: ",end='')
# print(*df_std.Model_system.unique(), sep=', ')


# print("unique GO term: ",end='')
# print(*[str(l).strip().split(' (')[0] for l in df_std['Molecular or Biological Process Investigated (GO term)'].unique()], sep=', ')


# print("unique phenotype: ",end='')
# print(*[str(l).strip().split(' (')[0] for l in df_std['Phenotype Measured ontology term'].unique()], sep=', ')


# In[219]:


df_std.columns


# In[220]:


# # df_std = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')

# # Build a mapping from Dataset -> (assay_type, model_system)
# lookup = (
#     df_std
#     .groupby("Dataset")
#     .first()[["Assay Type", "Model_system"]]
#     .rename(columns={
#         "Dataset": "dataset",
#         "Assay Type": "assay_type",
#         "Model_system": "model_system"
#     })
# )

# # Join into dataset_info_df_full
# dataset_info_df_full = dataset_info_df_full.merge(
#     lookup,
#     left_on="dataset",
#     right_index=True,
#     how="left"
# )


# In[221]:


# dataset_info_df_full.assay_type.unique()


# In[ ]:





# In[222]:


# model_map = {
#     'immortalized human cells': 'Immortalized Human cells',
#     'murine primary cells': 'Mouse cells',
#     'yeast': 'Yeast',
#     'other': 'Other',
#     'Not Applicable': 'N/A'
# }

# dataset_info_df_full['model_system'] = (
#     dataset_info_df_full['model_system'].map(model_map)
# )

# mapping = {
#     'Reporter': 'Reporter',
#     'Cell fitness': 'Cell Fitness',
#     'Cell Fitness': 'Cell Fitness',
#     'Direct Protein Function': 'Direct Protein Function',
#     'Not Applicable (trained predictor)': 'Not Applicable (trained predictor)'
# }

# dataset_info_df_full['assay_type'] = dataset_info_df_full['assay_type'].map(mapping)

# assay_method_map = pd.read_csv("/data/ross/assay_calibration/dataframe/var_effect_measurements_dataset.csv")
assay_method_map = pd.read_csv("/data/ross/assay_calibration/paper/excalibr_datasets_oldnames.csv")
assay_method_map.iloc[0]


# In[223]:


assay_method_map['disease'].value_counts(), assay_method_map['model_system'].value_counts(), assay_method_map['assay_type'].value_counts(), assay_method_map['vamp_sge'].value_counts()


# In[224]:


len(assay_method_map)


# In[225]:


# df_std[df_std.Dataset == "TSC2_combined_unpublished"].iloc[0]['Model_system']


# In[487]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_dataset_point_heatmap(dataset_info_df, all_danz_assignments, 
                               assay_method_map=None,
                               igvf_image_path=None,
                               figsize=(16, 16),
                               mark_zeros='white',
                               log_scale=False,
                               sort_by='assay_type'):
    """
    Create heatmap of evidence point distributions with letter-coded categorical markers.
    
    Parameters
    ----------
    dataset_info_df : pd.DataFrame
        DataFrame with columns: dataset, n_variants, gene
    all_danz_assignments : np.ndarray
        Array of DanZ point assignments
    assay_method_map : pd.DataFrame, optional
        DataFrame with columns: dataset, vamp_sge, model_system, disease, IGVF_produced
    igvf_image_path : str, optional
        Path to image file for IGVF marker (if None, uses star symbol)
    figsize : tuple
    mark_zeros : str
    log_scale : bool
    sort_by : str or list of str
        Column(s) to sort rows by. Options:
            'assay_type'    - VAMP-seq/SGE/Meta-analysis/Other  (default)
            'model_system'  - immortalized human cells / murine / yeast / etc.
            'disease'       - cancer / cardiovascular / rare disease / metabolic
            'gene'          - gene name alphabetical
            'dataset'       - dataset name alphabetical
            'igvf'          - IGVF-produced first
        Pass a list for multi-level sorting, e.g. ['disease', 'gene', 'dataset'].
        'dataset' is always appended as a tiebreaker if not already present.
    """
    
    all_points = list(range(-8, 9))
    negative_points = [p for p in all_points if p < 0]
    positive_points = [p for p in all_points if p > 0]
    
    # Letter code mappings
    VAMP_SGE_CODES = {
        'VAMP-seq': 'V',
        'SGE': 'S',
        'Meta-analysis': 'M',
        'other': 'O',
        'Other': 'O',
        None: 'O',
        np.nan: 'O'
    }
    
    MODEL_SYSTEM_CODES = {
        'immortalized human cells': 'H',
        'murine primary cells': 'M',
        'yeast': 'Y',
        'other': 'O',
        'not applicable': 'N',
        None: 'N',
        np.nan: 'N'
    }
    
    DISEASE_CODES = {
        'cancer': 'C',
        'cardiovascular': 'V',
        'cardio': 'V',
        'rare disease': 'R',
        'metabolic': 'M',
        None: 'O',
        np.nan: 'O'
    }
    
    # Color mappings
    VAMP_SGE_COLORS = {
        'V': '#2EB5AC',
        'S': '#E77F3A',
        'M': '#6C9EBF',
        'O': '#9E9E9E'
    }
    
    MODEL_SYSTEM_COLORS = {
        'H': '#E8927A',
        'M': '#D4A574',
        'Y': '#C9B87C',
        'N': '#BDBDBD',
        'O': '#9E9E9E'
    }
    
    DISEASE_COLORS = {
        'C': '#D45F5F',
        'V': '#5B8AC4',
        'R': '#C5A87A',
        'M': '#60B89F',
        'O': '#9E9E9E'
    }
    
    # Create metadata mappings
    vamp_sge_map = {}
    model_system_map = {}
    disease_map = {}
    igvf_map = {}
    gene_map = {}
    
    if assay_method_map is not None:
        for _, row in assay_method_map.iterrows():
            dataset = row['dataset']
            
            vamp_val = row.get('vamp_sge', None)
            model_val = row.get('model_system', None)
            
            if (pd.isna(vamp_val) or vamp_val == '' or vamp_val == 'not applicable') and \
               (pd.isna(model_val) or model_val == 'not applicable'):
                vamp_val = 'Meta-analysis'
            elif pd.isna(vamp_val) or vamp_val == '' or vamp_val == 'not applicable':
                vamp_val = 'Other'
            
            vamp_sge_map[dataset] = vamp_val
            model_system_map[dataset] = model_val if not pd.isna(model_val) else 'not applicable'
            disease_map[dataset] = row.get('disease', None)
            igvf_map[dataset] = row.get('IGVF_produced', False)
            gene_map[dataset] = row.get('gene', 'Other')
    
    # Build proportion matrix
    dataset_proportions = []
    dataset_names = []
    vamp_sge_vals = []
    model_system_vals = []
    disease_vals = []
    is_igvf = []
    genes = []
    
    vt_idx = 0
    for row_idx, row in dataset_info_df.iterrows():
        dataset = row['dataset']
        n_variants = row['n_variants']
        new_vt_idx = vt_idx + n_variants
        
        points, counts = np.unique(all_danz_assignments[vt_idx:new_vt_idx], return_counts=True)
        point_dict = dict(zip(points, counts))
        proportions = np.array([point_dict.get(p, 0) / n_variants for p in all_points])
        
        dataset_proportions.append(proportions)
        dataset_names.append(dataset)
        
        vamp_sge_vals.append(vamp_sge_map.get(dataset, 'Other'))
        model_system_vals.append(model_system_map.get(dataset, 'not applicable'))
        disease_vals.append(disease_map.get(dataset, None))
        is_igvf.append(igvf_map.get(dataset, False))
        genes.append(gene_map.get(dataset, row.get('gene', 'Other')))
        
        vt_idx = new_vt_idx
    
    proportion_df = pd.DataFrame(dataset_proportions, index=dataset_names, columns=all_points)
    
    # ── Sorting ──────────────────────────────────────────────────────
    vamp_sge_order = {'VAMP-seq': 0, 'SGE': 1, 'Meta-analysis': 2, 'other': 3, 'Other': 3}
    model_system_order = {
        'immortalized human cells': 0, 'murine primary cells': 1,
        'yeast': 2, 'other': 3, 'not applicable': 4,
    }
    disease_order = {'cancer': 0, 'cardiovascular': 1, 'cardio': 1, 'rare disease': 2, 'metabolic': 3}
    
    sort_df = pd.DataFrame({
        '_vamp_sge': vamp_sge_vals,
        '_vamp_sge_order': [vamp_sge_order.get(v, 3) for v in vamp_sge_vals],
        '_model_system': model_system_vals,
        '_model_system_order': [model_system_order.get(v, 4) for v in model_system_vals],
        '_disease': disease_vals,
        '_disease_order': [disease_order.get(v, 99) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 99 for v in disease_vals],
        '_gene': genes,
        '_igvf': is_igvf,
        '_igvf_order': [0 if ig else 1 for ig in is_igvf],
        '_dataset': dataset_names,
    })
    
    # Map sort_by names → sort_df column names
    SORT_KEY_MAP = {
        'assay_type':   '_vamp_sge_order',
        'vamp_sge':     '_vamp_sge_order',
        'model_system': '_model_system_order',
        'disease':      '_disease_order',
        'gene':         '_gene',
        'dataset':      '_dataset',
        'igvf':         '_igvf_order',
    }
    
    # Normalise to list
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    
    sort_cols = []
    for key in sort_by:
        col = SORT_KEY_MAP.get(key)
        if col is None:
            raise ValueError(
                f"Unknown sort_by key '{key}'. "
                f"Choose from: {list(SORT_KEY_MAP.keys())}"
            )
        if col not in sort_cols:
            sort_cols.append(col)
    
    # Always add dataset as final tiebreaker
    if '_dataset' not in sort_cols:
        sort_cols.append('_dataset')
    
    sort_df = sort_df.sort_values(sort_cols, ascending=True)
    proportion_df = proportion_df.reindex(sort_df['_dataset'].values)
    
    # ── Create figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.12, 0.50, 0.83])
    
    # Colormaps
    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", 
                                                  ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E'])
    red_cmap = LinearSegmentedColormap.from_list("red_gradient", 
                                                 ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744'])
    gray_cmap = LinearSegmentedColormap.from_list("gray_gradient", 
                                                  ['#F5F5F5', '#CCCCCC', '#999999', '#666666'])
    
    # Split data
    neg_idx = [all_points.index(p) for p in negative_points]
    zero_idx = [all_points.index(0)]
    pos_idx = [all_points.index(p) for p in positive_points]
    
    plot_data = proportion_df.values.copy()
    
    neg_data = np.full_like(plot_data, np.nan)
    neg_data[:, neg_idx] = plot_data[:, neg_idx]
    if mark_zeros == 'white':
        neg_data = np.ma.masked_where(neg_data == 0, neg_data)
    
    zero_data = np.full_like(plot_data, np.nan)
    zero_data[:, zero_idx] = plot_data[:, zero_idx]
    if mark_zeros == 'white':
        zero_data = np.ma.masked_where(zero_data == 0, zero_data)
    
    pos_data = np.full_like(plot_data, np.nan)
    pos_data[:, pos_idx] = plot_data[:, pos_idx]
    if mark_zeros == 'white':
        pos_data = np.ma.masked_where(pos_data == 0, pos_data)
    
    # Normalization
    if log_scale:
        non_zero_vals = plot_data[plot_data > 0]
        if len(non_zero_vals) > 0:
            vmin = max(non_zero_vals.min(), 1e-4)
            vmax = non_zero_vals.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None
        vmin, vmax = 0, 1
    
    # Plot heatmaps
    ax.imshow(neg_data, aspect='auto', cmap=blue_cmap, norm=norm, 
              vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)
    im_zero = ax.imshow(zero_data, aspect='auto', cmap=gray_cmap, norm=norm,
                        vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)
    ax.imshow(pos_data, aspect='auto', cmap=red_cmap, norm=norm,
              vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)
    
    # Smaller colorbar
    cbar = plt.colorbar(im_zero, ax=ax, shrink=0.4, pad=0.16, aspect=15)
    cbar_label = 'Proportion of variants' + (' (log scale)' if log_scale else '')
    cbar.set_label(cbar_label, fontsize=11, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    # Grid lines (soft horizontal lines)
    for i in range(proportion_df.shape[0] + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=1, alpha=0.5, zorder=10)
    
    # Cell borders
    y_offset = 0.55
    for i in range(proportion_df.shape[0]):
        for j in range(proportion_df.shape[1]):
            if proportion_df.iloc[i, j] != 0:
                ax.plot([j-0.5, j-0.5], [i-y_offset, i+0.5], color="#E0E0E0", lw=1)
                ax.plot([j-0.5, j+0.5], [i-y_offset, i-y_offset], color="#E0E0E0", lw=1)
                ax.plot([j+0.5, j+0.5], [i-y_offset, i+0.5], color="#E0E0E0", lw=1)
                ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color="#E0E0E0", lw=1)
    
    # Axis labels
    ax.set_xlabel('Evidence points', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Dataset', fontsize=14, fontweight='bold', labelpad=10)
    
    xlabels = [f"{p:+d}" if p != 0 else "0" for p in all_points]
    ax.set_xticks(np.arange(len(all_points)))
    ax.set_xticklabels(xlabels, rotation=0, fontsize=11, fontweight='500')
    
    # Load IGVF image if provided
    igvf_image = None
    if igvf_image_path is not None:
        try:
            igvf_image = plt.imread(igvf_image_path)
        except Exception as e:
            print(f"Could not load IGVF image: {e}")
            igvf_image = None
    
    # Y-axis labels with IGVF markers
    clean_names = []
    for i, dataset in enumerate(proportion_df.index):
        if igvf_map.get(dataset, False) and not dataset.startswith("HMBS") and not dataset.startswith("TP53"):
            if igvf_image is not None:
                imagebox = OffsetImage(igvf_image, zoom=0.04)
                ab = AnnotationBbox(imagebox, (-0.02, i+0.06), 
                                   xycoords=('axes fraction', 'data'),
                                   frameon=False, box_alignment=(1, 0.5))
                ax.add_artist(ab)
            clean_names.append(new_names_dict[dataset] + '      ')
        else:
            clean_names.append(new_names_dict[dataset])
    
    ax.set_yticks(np.arange(len(proportion_df)))
    ax.set_yticklabels(clean_names, rotation=0, fontsize=8)

    left, right = ax.get_xlim()
    ax.set_xlim(left - 0.3, right)
    
    # Direction labels
    zero_idx_pos = all_points.index(0)
    benign_center = zero_idx_pos / 2
    pathogenic_center = (zero_idx_pos + len(all_points)) / 2
    
    ax.text(benign_center, -0.8, '← Benign Evidence', ha='center', va='bottom', 
           fontsize=12, style='italic', fontweight='bold', color='#2166AC')
    ax.text(pathogenic_center, -0.8, 'Pathogenic Evidence →', ha='center', va='bottom',
           fontsize=12, style='italic', fontweight='bold', color='#B2182B')
    
    # ========== COLORED LETTER MARKER COLUMNS ==========
    
    marker_width = 0.05
    marker_spacing = 0.01
    marker_col1_x = 1.05
    marker_col2_x = marker_col1_x + marker_width + marker_spacing
    marker_col3_x = marker_col2_x + marker_width + marker_spacing
    
    n_rows = len(proportion_df)
    
    ax_pos = ax.get_position()
    header_y_fig = ax_pos.y0 - 0.004
    
    def axis_to_fig_x(x_axis):
        return ax_pos.x0 + (x_axis - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) * ax_pos.width - 0.022
    
    fig.text(axis_to_fig_x(marker_col1_x * (ax.get_xlim()[1] - ax.get_xlim()[0])), 
             header_y_fig, 'Assay Type',
             ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)
    
    fig.text(axis_to_fig_x(marker_col2_x * (ax.get_xlim()[1] - ax.get_xlim()[0])), 
             header_y_fig, 'Model System',
             ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)
    
    fig.text(axis_to_fig_x(marker_col3_x * (ax.get_xlim()[1] - ax.get_xlim()[0])), 
             header_y_fig, 'Disease',
             ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)
    
    # Draw markers for each dataset
    for idx, dataset in enumerate(proportion_df.index):
        y_pos = idx
        
        vamp_code = VAMP_SGE_CODES.get(sort_df.iloc[idx]['_vamp_sge'], 'O')
        model_code = MODEL_SYSTEM_CODES.get(sort_df.iloc[idx]['_model_system'], 'N')
        disease_code = DISEASE_CODES.get(sort_df.iloc[idx]['_disease'], 'O')
        
        # VAMP/SGE marker
        vamp_color = VAMP_SGE_COLORS[vamp_code]
        rect1 = Rectangle((marker_col1_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(),
                         facecolor=vamp_color, edgecolor='black', linewidth=0.5,
                         clip_on=False, zorder=19)
        ax.add_patch(rect1)
        ax.text(marker_col1_x, y_pos, vamp_code,
               transform=ax.get_yaxis_transform(), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)
        
        # Model system marker
        model_color = MODEL_SYSTEM_COLORS[model_code]
        rect2 = Rectangle((marker_col2_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(),
                         facecolor=model_color, edgecolor='black', linewidth=0.5,
                         clip_on=False, zorder=19)
        ax.add_patch(rect2)
        ax.text(marker_col2_x, y_pos, model_code,
               transform=ax.get_yaxis_transform(), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)
        
        # Disease marker
        disease_color = DISEASE_COLORS[disease_code]
        rect3 = Rectangle((marker_col3_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(),
                         facecolor=disease_color, edgecolor='black', linewidth=0.5,
                         clip_on=False, zorder=19)
        ax.add_patch(rect3)
        ax.text(marker_col3_x, y_pos, disease_code,
               transform=ax.get_yaxis_transform(), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)
    
    # ========== LEGENDS ==========
    
    if igvf_image is not None and igvf_image_path is not None:
        imagebox_legend = OffsetImage(igvf_image, zoom=0.07)
        ab_legend = AnnotationBbox(imagebox_legend, (1.005, 1.0007), 
                                   xycoords='axes fraction',
                                   frameon=False, box_alignment=(1, 1))
        ax.add_artist(ab_legend)
        ax.text(0.94, 0.995, 'IGVF dataset', transform=ax.transAxes,
               ha='right', va='top', fontsize=10)

        from matplotlib.patches import FancyBboxPatch

        box_x = 0.757
        box_y = 0.9805
        box_width = 0.265
        box_height = 0.019
        
        legend_box = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.005",
            edgecolor='#999999',
            facecolor='white',
            linewidth=1,
            alpha=0.9,
            zorder=1,
            clip_on=False
        )
        ax.add_patch(legend_box)
    
    else:
        igvf_legend = [Line2D([0], [0], marker='*', color='w', 
                             markerfacecolor='black', markersize=10, 
                             label='IGVF dataset', linestyle='None')]
        ax.legend(handles=igvf_legend, loc='upper right', fontsize=9, framealpha=0.9)
    
    # VAMP/SGE legend
    vamp_legend_text = [
        ('V', 'VAMP-seq', VAMP_SGE_COLORS['V']),
        ('S', 'SGE', VAMP_SGE_COLORS['S']),
        ('M', 'Meta-analysis', VAMP_SGE_COLORS['M']),
        ('O', 'Other', VAMP_SGE_COLORS['O'])
    ]

    LEGEND_X = 0.55
    fig.text(LEGEND_X, 0.80, 'Assay Type:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(vamp_legend_text):
        y_pos = 0.78 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015,
                        transform=fig.transFigure, facecolor=color, 
                        edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')
    
    # Model System legend
    model_legend_text = [
        ('H', 'Immortalized\nhuman cells', MODEL_SYSTEM_COLORS['H']),
        ('M', 'Murine\nprimary cells', MODEL_SYSTEM_COLORS['M']),
        ('Y', 'Yeast', MODEL_SYSTEM_COLORS['Y']),
        ('N', 'Not applicable', MODEL_SYSTEM_COLORS['N']),
        ('O', 'Other', MODEL_SYSTEM_COLORS['O'])
    ]
    
    fig.text(LEGEND_X, 0.60, 'Model System:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(model_legend_text):
        y_pos = 0.58 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015,
                        transform=fig.transFigure, facecolor=color, 
                        edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')
    
    # Disease legend
    disease_legend_text = [
        ('C', 'Cancer', DISEASE_COLORS['C']),
        ('V', 'Cardiovascular', DISEASE_COLORS['V']),
        ('R', 'Rare disease', DISEASE_COLORS['R']),
        ('M', 'Metabolic', DISEASE_COLORS['M'])
    ]
    
    fig.text(LEGEND_X, 0.40, 'Disease:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(disease_legend_text):
        y_pos = 0.38 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015,
                        transform=fig.transFigure, facecolor=color, 
                        edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig, ax, proportion_df, sort_df


# In[492]:


fig, ax, df, sort_df= plot_dataset_point_heatmap(
    dataset_info_df_full, 
    all_danz_oob_full,
    assay_method_map=assay_method_map,
    igvf_image_path='/data/ross/assay_calibration/flagship/igvf_logo.png',  # Provide your IGVF image path
    mark_zeros='white',
    log_scale=False,
    sort_by='model_system'
)


# In[493]:


fig.savefig(f'/data/ross/assay_calibration/paper/points_heatmap_sort_model_system.png',dpi=300,bbox_inches='tight')


# In[496]:


import numpy as np
import pandas as pd


def compute_assay_evidence_stats(dataset_info_df, all_danz_assignments, assay_method_map):
    """
    For each dataset, compute the proportion of variants reaching various
    evidence thresholds. Then summarize by assay type, model system, and disease.
    
    Parameters
    ----------
    dataset_info_df : pd.DataFrame
        DataFrame with columns: dataset, n_variants, n_plp, n_blb, n_gnomad,
        n_synonymous, n_vus, gene
    all_danz_assignments : np.ndarray
        Array of DanZ point assignments
    assay_method_map : pd.DataFrame
        DataFrame with columns: dataset, vamp_sge, model_system, disease, IGVF_produced
    
    Returns
    -------
    dataset_stats : pd.DataFrame
        Per-dataset stats with metadata columns.
    summary_dict : dict
        Keys are ('assay_type', 'model_system', 'disease'), values are
        DataFrames summarizing counts/percentages per group.
    """
    
    # Build metadata maps
    meta = {}
    for _, row in assay_method_map.iterrows():
        ds = row['dataset']
        vamp_val = row.get('vamp_sge', None)
        model_val = row.get('model_system', None)
        
        if (pd.isna(vamp_val) or vamp_val in ('', 'not applicable')) and \
           (pd.isna(model_val) or model_val == 'not applicable'):
            vamp_val = 'Meta-analysis'
        elif pd.isna(vamp_val) or vamp_val in ('', 'not applicable'):
            vamp_val = 'Other'
        
        meta[ds] = {
            'assay_type': vamp_val,
            'model_system': model_val if not pd.isna(model_val) else 'not applicable',
            'disease': row.get('disease', 'Other'),
            'gene': row.get('gene', ds.split('_')[0]),
        }
    
    # Control count columns from dataset_info_df
    control_cols = ['n_plp', 'n_blb', 'n_gnomad', 'n_synonymous']
    has_controls = all(c in dataset_info_df.columns for c in control_cols)
    optional_cols = ['n_vus']
    
    # Per-dataset evidence stats
    records = []
    vt_idx = 0
    for _, row in dataset_info_df.iterrows():
        ds = row['dataset']
        n = row['n_variants']
        pts = all_danz_assignments[vt_idx:vt_idx + n]
        vt_idx += n
        
        m = meta.get(ds, {})
        rec = {
            'dataset': ds,
            'gene': m.get('gene', ds.split('_')[0]),
            'assay_type': m.get('assay_type', 'Other'),
            'model_system': m.get('model_system', 'Other'),
            'disease': m.get('disease', 'Other'),
            'n_variants': n,
            # Direction proportions
            'pct_pathogenic': (pts > 0).sum() / n * 100,
            'pct_benign': (pts < 0).sum() / n * 100,
            'pct_indeterminate': (pts == 0).sum() / n * 100,
            # Threshold-specific proportions
            'pct_ge_plus1': (pts >= 1).sum() / n * 100,
            'pct_ge_plus2': (pts >= 2).sum() / n * 100,
            'pct_ge_plus4': (pts >= 4).sum() / n * 100,
            'pct_ge_plus8': (pts >= 8).sum() / n * 100,
            'pct_le_minus1': (pts <= -1).sum() / n * 100,
            'pct_le_minus2': (pts <= -2).sum() / n * 100,
            'pct_le_minus4': (pts <= -4).sum() / n * 100,
            'pct_le_minus8': (pts <= -8).sum() / n * 100,
            # Binary: does the dataset reach this threshold at all?
            'reaches_ge_plus1': (pts >= 1).any(),
            'reaches_ge_plus2': (pts >= 2).any(),
            'reaches_ge_plus4': (pts >= 4).any(),
            'reaches_ge_plus8': (pts >= 8).any(),
            'reaches_le_minus1': (pts <= -1).any(),
            'reaches_le_minus2': (pts <= -2).any(),
            'reaches_le_minus4': (pts <= -4).any(),
            'reaches_le_minus8': (pts <= -8).any(),
            # Non-zero = "any evidence"
            'pct_any_evidence': (pts != 0).sum() / n * 100,
            'reaches_any_evidence': (pts != 0).any(),
        }
        
        # Pull control counts directly from dataset_info_df
        if has_controls:
            for cc in control_cols:
                rec[cc] = int(row.get(cc, 0))
            rec['n_controls'] = sum(rec[cc] for cc in control_cols)
        for oc in optional_cols:
            if oc in dataset_info_df.columns:
                rec[oc] = int(row.get(oc, 0))
        
        records.append(rec)
    
    dataset_stats = pd.DataFrame(records)
    
    # Summarize by each stratification
    strat_cols = ['assay_type', 'model_system', 'disease']
    pct_cols = [c for c in dataset_stats.columns if c.startswith('pct_')]
    reaches_cols = [c for c in dataset_stats.columns if c.startswith('reaches_')]
    count_cols = [c for c in control_cols + ['n_controls', 'n_vus']
                  if c in dataset_stats.columns]
    
    summary_dict = {}
    for strat in strat_cols:
        rows = []
        for group, gdf in dataset_stats.groupby(strat):
            n_datasets = len(gdf)
            n_variants = gdf['n_variants'].sum()
            rec = {
                strat: group,
                'n_datasets': n_datasets,
                'n_variants': n_variants,
            }
            # Aggregate control counts
            for cc in count_cols:
                rec[f'{cc}_total'] = int(gdf[cc].sum())
                rec[f'{cc}_median'] = gdf[cc].median()
                rec[f'{cc}_mean'] = gdf[cc].mean()
            # Unweighted mean and median across datasets
            for sc in pct_cols:
                rec[f'{sc}_mean'] = gdf[sc].mean()
                rec[f'{sc}_median'] = gdf[sc].median()
            # Variant-weighted mean across datasets
            total_variants = gdf['n_variants'].sum()
            for sc in pct_cols:
                rec[f'{sc}_wmean'] = (gdf[sc] * gdf['n_variants']).sum() / total_variants if total_variants > 0 else 0.0
            # How many datasets have >50% of variants reaching each threshold
            for sc in pct_cols:
                rec[f'{sc}_n_above50'] = (gdf[sc] > 50).sum()
            # How many datasets reach the threshold at all (≥1 variant)
            for rc in reaches_cols:
                rec[f'{rc}_count'] = gdf[rc].sum()
            rows.append(rec)
        summary_dict[strat] = pd.DataFrame(rows).sort_values('n_datasets', ascending=False)
    
    return dataset_stats, summary_dict


def print_assay_evidence_report(dataset_stats, summary_dict):
    """Pretty-print the key findings."""
    
    has_controls = 'n_controls' in dataset_stats.columns
    
    print("=" * 80)
    print("ASSAY-LEVEL EVIDENCE STATISTICS")
    print("=" * 80)
    
    for strat, sdf in summary_dict.items():
        print(f"\n{'─' * 80}")
        print(f"Stratified by: {strat.upper()}")
        print(f"{'─' * 80}")
        
        for _, row in sdf.iterrows():
            group = row[strat]
            n_ds = int(row['n_datasets'])
            n_var = int(row['n_variants'])
            print(f"\n  {group} ({n_ds} datasets, {n_var:,} variants)")
            
            if has_controls:
                print(f"    Control variants (total across datasets):")
                print(f"      P/LP: {int(row['n_plp_total']):,}"
                      f"  |  B/LB: {int(row['n_blb_total']):,}"
                      f"  |  gnomAD: {int(row['n_gnomad_total']):,}"
                      f"  |  Syn: {int(row['n_synonymous_total']):,}")
                print(f"    Control variants (median per dataset):")
                print(f"      P/LP: {row['n_plp_median']:.0f}"
                      f"  |  B/LB: {row['n_blb_median']:.0f}"
                      f"  |  gnomAD: {row['n_gnomad_median']:.0f}"
                      f"  |  Syn: {row['n_synonymous_median']:.0f}")
                if 'n_vus_total' in row:
                    print(f"      VUS: {int(row['n_vus_total']):,} total"
                          f"  (median {row['n_vus_median']:.0f}/dataset)")
            
            print(f"    Direction (variant-weighted across datasets):")
            print(f"      Pathogenic (>0): {row['pct_pathogenic_wmean']:.1f}%")
            print(f"      Benign    (<0):  {row['pct_benign_wmean']:.1f}%")
            print(f"      Indet.    (=0):  {row['pct_indeterminate_wmean']:.1f}%")
            
            print(f"    Datasets REACHING threshold (≥1 variant):")
            print(f"      ≥+1: {int(row['reaches_ge_plus1_count'])}/{n_ds}"
                  f"  |  ≤-1: {int(row['reaches_le_minus1_count'])}/{n_ds}")
            print(f"      ≥+2: {int(row['reaches_ge_plus2_count'])}/{n_ds}"
                  f"  |  ≤-2: {int(row['reaches_le_minus2_count'])}/{n_ds}")
            print(f"      ≥+4: {int(row['reaches_ge_plus4_count'])}/{n_ds}"
                  f"  |  ≤-4: {int(row['reaches_le_minus4_count'])}/{n_ds}")
            print(f"      ≥+8: {int(row['reaches_ge_plus8_count'])}/{n_ds}"
                  f"  |  ≤-8: {int(row['reaches_le_minus8_count'])}/{n_ds}")
            
            print(f"    Variant-weighted % reaching threshold:")
            print(f"      ≥+2: {row['pct_ge_plus2_wmean']:.1f}%  |  ≤-2: {row['pct_le_minus2_wmean']:.1f}%")
            print(f"      ≥+4: {row['pct_ge_plus4_wmean']:.1f}%  |  ≤-4: {row['pct_le_minus4_wmean']:.1f}%")
            print(f"      ≥+8: {row['pct_ge_plus8_wmean']:.1f}%  |  ≤-8: {row['pct_le_minus8_wmean']:.1f}%")
            
            print(f"    Datasets with >50% variants reaching:")
            print(f"      ≥+2: {int(row['pct_ge_plus2_n_above50'])}/{n_ds}"
                  f"  |  ≤-2: {int(row['pct_le_minus2_n_above50'])}/{n_ds}")
            print(f"      Any evidence: {int(row['pct_any_evidence_n_above50'])}/{n_ds}")
    
    # Specific VAMP vs SGE comparison
    print(f"\n{'=' * 80}")
    print("VAMP-seq vs SGE COMPARISON")
    print("=" * 80)
    
    for assay in ['VAMP-seq', 'SGE']:
        sub = dataset_stats[dataset_stats['assay_type'] == assay]
        n_ds = len(sub)
        if n_ds == 0:
            continue
        print(f"\n  {assay} ({n_ds} datasets):")
        if has_controls:
            print(f"    Control variants (median per dataset):")
            print(f"      P/LP: {sub['n_plp'].median():.0f}"
                  f"  |  B/LB: {sub['n_blb'].median():.0f}"
                  f"  |  gnomAD: {sub['n_gnomad'].median():.0f}"
                  f"  |  Syn: {sub['n_synonymous'].median():.0f}")
        print(f"    Datasets reaching (≥1 variant):")
        print(f"      ≥+2: {sub['reaches_ge_plus2'].sum()}/{n_ds}"
              f"  |  ≤-2: {sub['reaches_le_minus2'].sum()}/{n_ds}")
        print(f"      ≥+4: {sub['reaches_ge_plus4'].sum()}/{n_ds}"
              f"  |  ≤-4: {sub['reaches_le_minus4'].sum()}/{n_ds}")
        print(f"      ≥+8: {sub['reaches_ge_plus8'].sum()}/{n_ds}"
              f"  |  ≤-8: {sub['reaches_le_minus8'].sum()}/{n_ds}")
        print(f"    Datasets where ≥50% variants reach:")
        print(f"      ≥+2: {(sub['pct_ge_plus2'] > 50).sum()}/{n_ds}"
              f"  |  ≤-2: {(sub['pct_le_minus2'] > 50).sum()}/{n_ds}")
        print(f"      Any evidence: {(sub['pct_any_evidence'] > 50).sum()}/{n_ds}")
        total_v = sub['n_variants'].sum()
        wpath = (sub['pct_pathogenic'] * sub['n_variants']).sum() / total_v
        wben = (sub['pct_benign'] * sub['n_variants']).sum() / total_v
        wind = (sub['pct_indeterminate'] * sub['n_variants']).sum() / total_v
        print(f"    Variant-weighted %: path={wpath:.1f}%"
              f"  ben={wben:.1f}%"
              f"  ind={wind:.1f}%"
              f"  ({total_v:,} variants)")


des_dataset_stats, des_summary_dict = compute_assay_evidence_stats(
    dataset_info_df_full, all_danz_oob_full, assay_method_map
)
print_assay_evidence_report(des_dataset_stats, des_summary_dict)

# Drill into specific patterns:
# dataset_stats[dataset_stats['model_system'] == 'yeast'].sort_values('pct_any_evidence')
# dataset_stats[dataset_stats['disease'] == 'cancer'].sort_values('pct_pathogenic', ascending=False)


# In[495]:


dataset_info_df_full.columns


# In[229]:


# df_hu = df_std[df_std.Dataset == "BRCA2_Hu_2024"]


# In[230]:


# importlib.reload(src.assay_calibration.data_utils.dataset)
# from src.assay_calibration.data_utils.dataset import Scoreset

# scoreset = Scoreset(df_hu, clinvar_release="2025", synonymous_exclusive=False, score_avg=True)
# plt.hist(scoreset.scores[scoreset.sample_assignments[:,0]], color='red', alpha=0.5)
# plt.hist(scoreset.scores[scoreset.sample_assignments[:,1]], color='blue', alpha=0.5)


# In[231]:


# import numpy as np

# df_hu = df_std[df_std.Dataset == "BRCA2_Hu_2024"]
# # --- Splice filter (same logic as your method) ---
# detects_splice = df_hu["splice_measure"].iloc[0] == "Yes"

# # if not detects_splice:
# #     df_hu = df_hu[
# #         df_hu["simplified_consequence"].str.lower() != "splice region"
# #     ]
# #     df_hu = df_hu[
# #         df_hu["simplified_consequence"].str.lower() != "splice_site_variant"
# #     ]
# #     df_hu = df_hu[
# #         (df_hu["spliceAI_DS_AG"] < 0.2) &
# #         (df_hu["spliceAI_DS_AL"] < 0.2) &
# #         (df_hu["spliceAI_DS_DG"] < 0.2) &
# #         (df_hu["spliceAI_DS_DL"] < 0.2)
# #     ]

# # --- ClinVar reliability filter ---
# zero_star_statuses = {
#     np.nan,
#     'no assertion criteria provided',
#     'no assertion provided',
#     'no interpretation for the single variant',
#     'no classification provided',
#     '-'
# }

# plp_set = {
#     "Pathogenic",
#     "Likely pathogenic",
#     "Pathogenic/Likely pathogenic",
# }

# mask = (
#     (df_hu["Flag"] != "*")
#     & df_hu["clinvar_sig_2025"].isin(plp_set)
#     # & ~df_hu["clinvar_star_2025"].isin(zero_star_statuses)
# )

# df_plp = df_hu.loc[mask].copy()


# blb_set = {
#     "Benign",
#     "Likely benign",
#     "Benign/Likely benign",
# }

# mask = (
#     (df_hu["Flag"] != "*")
#     & df_hu["clinvar_sig_2025"].isin(blb_set)
#     # & ~df_hu["clinvar_star_2025"].isin(zero_star_statuses)
# )

# df_blb = df_hu.loc[mask].copy()

# plt.hist(df_plp.auth_reported_score, color='red', alpha=0.5)
# plt.hist(df_blb.auth_reported_score, color='blue', alpha=0.5)


# In[232]:


# import numpy as np
# import matplotlib.pyplot as plt

# # Prepare the data
# df_hu = df_std[df_std.Dataset == "BRCA2_Hu_2024"]
# detects_splice = df_hu["splice_measure"].iloc[0] == "Yes"

# # Zero star statuses
# zero_star_statuses = {
#     np.nan,
#     'no assertion criteria provided',
#     'no assertion provided',
#     'no interpretation for the single variant',
#     'no classification provided',
#     '-'
# }

# plp_set = {
#     "Pathogenic",
#     "Likely pathogenic",
#     "Pathogenic/Likely pathogenic",
# }

# blb_set = {
#     "Benign",
#     "Likely benign",
#     "Benign/Likely benign",
# }

# # Define the 4 specific combinations
# combinations = [
#     {"cv_version": "2025", "filter_splice": True, "star_mode": "both", "label": "2025, Splice, ≥1 star"},
#     {"cv_version": "2025", "filter_splice": False, "star_mode": "both", "label": "2025, No Splice, ≥1 star"},
#     {"cv_version": "2025", "filter_splice": True, "star_mode": "neither", "label": "2025, Splice, ≥0 star"},
#     {"cv_version": "2018", "filter_splice": True, "star_mode": "both", "label": "2018, Splice, ≥1 star"},
# ]

# # Create grid: 1 row × 4 cols
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# fig.suptitle('BRCA2 Hu', fontsize=16, fontweight='bold')

# print("\n" + "="*80)
# print("SUMMARY STATISTICS")
# print("="*80)

# for idx, combo in enumerate(combinations):
#     ax = axes[idx]
#     cv_version = combo["cv_version"]
#     filter_splice = combo["filter_splice"]
#     star_mode = combo["star_mode"]
    
#     # Apply splice filtering
#     df_filtered = df_hu.copy()
#     if filter_splice and not detects_splice:
#         df_filtered = df_filtered[
#             df_filtered["simplified_consequence"].str.lower() != "splice region"
#         ]
#         df_filtered = df_filtered[
#             df_filtered["simplified_consequence"].str.lower() != "splice_site_variant"
#         ]
#         df_filtered = df_filtered[
#             (df_filtered["spliceAI_DS_AG"] < 0.2) &
#             (df_filtered["spliceAI_DS_AL"] < 0.2) &
#             (df_filtered["spliceAI_DS_DG"] < 0.2) &
#             (df_filtered["spliceAI_DS_DL"] < 0.2)
#         ]
    
#     # Select ClinVar columns
#     sig_col = f"clinvar_sig_{cv_version}"
#     star_col = f"clinvar_star_{cv_version}"
    
#     # Apply star enforcement for PLP
#     mask_plp = (
#         (df_filtered["Flag"] != "*") &
#         df_filtered[sig_col].isin(plp_set)
#     )
#     if star_mode == "both":
#         mask_plp &= ~df_filtered[star_col].isin(zero_star_statuses)
    
#     df_plp = df_filtered.loc[mask_plp].copy()
    
#     # Apply star enforcement for BLB
#     mask_blb = (
#         (df_filtered["Flag"] != "*") &
#         df_filtered[sig_col].isin(blb_set)
#     )
#     if star_mode == "both":
#         mask_blb &= ~df_filtered[star_col].isin(zero_star_statuses)
    
#     df_blb = df_filtered.loc[mask_blb].copy()
    
#     # Plot histograms
#     if len(df_plp) > 0:
#         ax.hist(df_plp.auth_reported_score, bins=20, color='red', 
#                alpha=0.5, label=f'PLP (n={len(df_plp)})', density=True)
#     if len(df_blb) > 0:
#         ax.hist(df_blb.auth_reported_score, bins=20, color='blue', 
#                alpha=0.5, label=f'BLB (n={len(df_blb)})', density=True)
    
#     # Styling
#     ax.set_xlabel('Score', fontsize=9)
#     if idx == 0:
#         ax.set_ylabel('Density', fontsize=9)
#     ax.legend(fontsize=8, loc='upper right')
#     ax.tick_params(labelsize=8)
#     ax.grid(True, alpha=0.3)
#     ax.set_title(combo["label"], fontsize=10, fontweight='bold')
    
#     # Print statistics
#     print(f"\n{combo['label']}")
#     print(f"  PLP: n={len(df_plp):4d} | mean={df_plp.auth_reported_score.mean():.3f} | std={df_plp.auth_reported_score.std():.3f}")
#     print(f"  BLB: n={len(df_blb):4d} | mean={df_blb.auth_reported_score.mean():.3f} | std={df_blb.auth_reported_score.std():.3f}")

# plt.tight_layout()
# plt.show()


# In[ ]:





# In[233]:


df_point_distr = df


# In[234]:


len(df_point_distr)


# In[235]:


# df_std = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')
df_std = df_std[~df_std.Dataset.isin(["TP53_Fayer_2021_meta","SFPQ_unpublished","F9_Popp_2025_model","TSC2_rapgap_unpublished","TSC2_tuberin_unpublished"])]


# In[236]:


# df_test = df_test[~df_test.Dataset.isin(["TP53_Fayer_2021_meta","SFPQ_unpublished","F9_Popp_2025_model"])]


# In[237]:


df_std.Dataset.nunique()


# In[238]:


df_clingen.Dataset.nunique()


# In[454]:


df_OP = pd.read_csv("/data/ross/assay_calibration/paper/brnich_evidence.csv")
df_OP['Dataset'] = df_OP['Dataset'].map(old_names_dict)
df_OP_unique = df_OP.Dataset.unique()


# In[464]:


len(df_OP[(df_OP["Evidence Code Normal"] == "Not available") & (df_OP["Evidence Code Abnormal"] == "Not available")])


# In[239]:


# Define the evidence levels we care about
levels = [1, 2, 4, 8]

# Initialize counters
excalibr_path_counts = {lvl: 0 for lvl in levels}
excalibr_ben_counts  = {lvl: 0 for lvl in levels}

# Iterate over datasets (rows)
for dataset_name, row in df_point_distr.iterrows():
    for lvl in levels:
        # Pathogenic: +lvl evidence
        if row[lvl] > 0:
            excalibr_path_counts[lvl] += 1
        # Benign: -lvl evidence
        if row[-lvl] > 0:
            excalibr_ben_counts[lvl] += 1

print("Datasets reaching pathogenic evidence ±X:", excalibr_path_counts)
print("Datasets reaching benign evidence ±X:", excalibr_ben_counts)


# In[240]:


# df_clingen = pd.read_csv('/data/ross/assay_calibration/Dan_fxn_calibrations_new_2025_2018_clust_calib_2025_12_27.csv.gz')
# df_clingen_unique = df_clingen.Dataset.unique()


# In[241]:


# p = df_clingen[df_clingen.Dataset == "TSC2_rapgap_unpublished"]['Pathogenic controls_clinvar_18_25'].unique()[0] + df_clingen[df_clingen.Dataset == "TSC2_tuberin_unpublished"]['Pathogenic controls_clinvar_18_25'].unique()[0]
# b = df_clingen[df_clingen.Dataset == "TSC2_rapgap_unpublished"]['Benign controls_clinvar_18_25'].unique()[0] + df_clingen[df_clingen.Dataset == "TSC2_tuberin_unpublished"]['Benign controls_clinvar_18_25'].unique()[0]
# p,b


# In[448]:


len(df_std_unique), len(df_OP_unique)


# In[451]:


df_OP[df_OP.Dataset.isna()]


# In[456]:


df_std_unique = df_std['Dataset'].unique()
for ds in df_OP_unique:
    if ds not in df_std_unique:
        print(ds)


# In[418]:


[op for op in df_OP["OddsNormal"].unique() if op == "0.0"]


# In[444]:


import numpy as np
import pandas as pd

# Odds thresholds for pathogenic (Abnormal) and benign (Normal)
# Mapping to ACMG-like points ±1,2,4,8
# path_thresholds = {
#     1: 2.1,    # supporting
#     2: 4.3,    # moderate
#     4: 18.7,   # strong
#     8: 350     # very strong
# }

# ben_thresholds = {
#     1: 1/2.1,    # supporting benign
#     2: 1/4.3,    # moderate
#     4: 1/18.7,   # strong
#     ### BS3 VERY STRONG NOT DEFINED IN BRNICH
#     # 8: 1/350     # very strong 
# }

path_codes = ['PS3_supporting','PS3_moderate','PS3_strong','PS3_very_strong']
ben_codes = ['BS3_supporting','BS3_moderate','BS3_strong','BS3_very_strong']

# Initialize counters
auth_path_counts = {1: set(), 2: set(), 4: set(), 8: set()}
auth_ben_counts  = {1: set(), 2: set(), 4: set(), 8: set()}

# df_test = pd.read_csv('/data/ross/assay_calibration/Dan_fxn_calibrations_new_2025_2018_clust_calib_2025_12_27.csv.gz')

# Iterate over unique datasets
for ds_unprocessed in df_std['Dataset'].unique():
    
    assert ds_unprocessed in df_OP_unique
    
    # if ds_unprocessed not in df_clingen_unique:
    #     continue
    
    ds = ds_unprocessed#.replace("Jia_2021", "Scott_2022")

    # if ds != "TSC2_combined_unpublished":
        # odds_path = df_clingen[df_clingen.Dataset == ds]['OddsAbnormal_clinvar_18_25'].unique()[0]
        # odds_ben  = df_clingen[df_clingen.Dataset == ds]['OddsNormal_clinvar_18_25'].unique()[0]
        # p_control_count = df_clingen[df_clingen.Dataset == ds]['Pathogenic controls_clinvar_18_25'].unique()[0]
        # b_control_count  = df_clingen[df_clingen.Dataset == ds]['Benign controls_clinvar_18_25'].unique()[0]
    # else:
    #     odds_path = 33.1
    #     odds_ben = 0.343
    #     p_control_count = 11
    #     b_control_count = 185

    # odds_path = df_clingen[df_clingen.Dataset == ds]['OddsAbnormal'].unique()[0]
    # odds_ben  = df_clingen[df_clingen.Dataset == ds]['OddsNormal'].unique()[0]
    # p_control_count = assay_method_map[assay_method_map.dataset == ds]["n_plp"].item()
    # b_control_count  = assay_method_map[assay_method_map.dataset == ds]["n_blb"].item()

    # print(odds_path, odds_ben, p_control_count, b_control_count)
    
    # control_count = p_control_count + b_control_count

    path_evidence = df_OP[df_OP.Dataset == ds]["Evidence Code Abnormal"].item()
    ben_evidence = df_OP[df_OP.Dataset == ds]["Evidence Code Normal"].item()

    for i,code in enumerate(path_codes):
        pt_i = 2**i
        if path_evidence == code:
            auth_path_counts[pt_i].add(ds)
            print(f"{ds}: +{pt_i}",end=' ')
            for j in range(i-1, -1, -1):
                pt_j = 2**j
                auth_path_counts[pt_j].add(ds)
                print(f"+{pt_j}",end=' ')
            break

    for i,code in enumerate(ben_codes):
        pt_i = 2**i
        if ben_evidence == code:
            auth_ben_counts[pt_i].add(ds)
            print(f"{ds}: -{pt_i}",end=' ')
            for j in range(i-1, -1, -1):
                pt_j = 2**j
                auth_ben_counts[pt_j].add(ds)
                print(f"-{pt_j}",end=' ')
            break
        
    print()
        
    
    # for pt, thresh in path_thresholds.items():
        # try:
        #     if odds_path >= thresh:
        #         auth_path_counts[pt].add(ds)
        # except TypeError as e:
        #     if pt == 1 or (control_count > 10 and  pt == 2):
        #         auth_path_counts[pt].add(ds)
    
    # for pt, thresh in ben_thresholds.items():
        # try:
        #     if odds_ben <= thresh:
        #         auth_ben_counts[pt].add(ds)
        # except TypeError as e:
        #     if pt == 1 or (control_count > 10 and  pt == 2):
        #         auth_ben_counts[pt].add(ds)

# for thresh, ds_set in auth_path_counts.items():
#     if "TSC2_rapgap_unpublished" in ds_set:
#         print(f"rapgap +{thresh}")
#     if "TSC2_tuberin_unpublished" in ds_set:
#         print(f"tuberin +{thresh}")
        
# for thresh, ds_set in auth_ben_counts.items():
#     if "TSC2_rapgap_unpublished" in ds_set:
#         print(f"rapgap -{thresh}")
#     if "TSC2_tuberin_unpublished" in ds_set:
#         print(f"tuberin -{thresh}")

# Convert to counts
auth_path_counts = {k: len(v) for k, v in auth_path_counts.items()}
auth_ben_counts  = {k: len(v) for k, v in auth_ben_counts.items()}

print("Datasets reaching pathogenic evidence ±X:", auth_path_counts)
print("Datasets reaching benign evidence ±X:", auth_ben_counts)

# Datasets reaching pathogenic evidence ±X: {1: 36, 2: 27, 4: 12, 8: 0}
# Datasets reaching benign evidence ±X: {1: 43, 2: 31, 4: 8, 8: 0}


# In[445]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_evidence_comparison(excalibr_path_counts, excalibr_ben_counts, 
                             auth_path_counts, auth_ben_counts,
                             save_path=None, figsize=(14, 6)):
    """
    Create a grouped bar plot comparing Excalibr vs OddsPath/Author evidence counts.
    
    Parameters
    ----------
    excalibr_path_counts : dict
        Dictionary mapping evidence level to count of datasets with pathogenic evidence (Excalibr)
    excalibr_ben_counts : dict
        Dictionary mapping evidence level to count of datasets with benign evidence (Excalibr)
    auth_path_counts : dict
        Dictionary mapping evidence level to count of datasets with pathogenic evidence (OddsPath)
    auth_ben_counts : dict
        Dictionary mapping evidence level to count of datasets with benign evidence (OddsPath)
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot.
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    # Evidence levels
    levels = [1, 2, 4, 8]
    
    # Excalibr strength colors (red for pathogenic, blue for benign)
    excalibr_colors = {
        8: '#943744', 4: '#B85C6B', 2: '#D68F99', 1: '#E6B1B8',
        0: '#E0E0E0',
        -1: '#99C8DC', -2: '#7AB5D1', -4: '#4B91A6', -8: '#2E6B7E',
    }
    
    # OddsPath colors - purple for pathogenic, teal/green for benign (very different!)
    oddspath_colors = {
        8: '#6B2C91', 4: '#8E4BB8', 2: '#A96BC9', 1: '#C49EDB',  # Purple gradient
        0: '#999999',
        -1: '#66C2A5', -2: '#3D9970', -4: '#2A7A5C', -8: '#1A5C42',  # Teal/green gradient
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set up bar positions
    x = np.arange(len(levels))
    width = 0.35
    
    # --- Pathogenic Evidence Plot ---
    excalibr_path_vals = [excalibr_path_counts[lvl] for lvl in levels]
    auth_path_vals = [auth_path_counts[lvl] for lvl in levels]
    
    # Plot Excalibr bars with individual colors
    for i, lvl in enumerate(levels):
        ax1.bar(x[i] - width/2, excalibr_path_vals[i], width, 
               color=excalibr_colors[lvl], alpha=0.9, 
               edgecolor='black', linewidth=1.2,
               label='ExCALIBR' if i == 0 else '')
    
    # Plot OddsPath bars with individual colors
    for i, lvl in enumerate(levels):
        ax1.bar(x[i] + width/2, auth_path_vals[i], width, 
               color=oddspath_colors[lvl], alpha=0.9, 
               edgecolor='black', linewidth=1.2,
               label=r'Brnich $\it{et\ al.}$' if i == 0 else '')

    FONTSIZE_SMALL = 11
    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 12
    FONTSIZE_LEGEND = 12
    FONTSIZE_PANEL_LETTER = 18
    
    # Add value labels on bars
    for i in range(len(levels)):
        # Excalibr labels
        # if excalibr_path_vals[i] > 0:
        ax1.text(x[i] - width/2, excalibr_path_vals[i],
                f'{int(excalibr_path_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        # OddsPath labels
        # if auth_path_vals[i] > 0:
        ax1.text(x[i] + width/2, auth_path_vals[i],
                f'{int(auth_path_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
    
    ax1.set_xlabel('Evidence Level (Pathogenic)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax1.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    # ax1.set_title('Pathogenic Evidence (+1, +2, +4, +8)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'+{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax1.legend(fontsize=FONTSIZE_LEGEND, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.05, 1.08, '(A)', transform=ax1.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')
    
    # --- Benign Evidence Plot ---
    excalibr_ben_vals = [excalibr_ben_counts[lvl] for lvl in levels]
    auth_ben_vals = [auth_ben_counts[lvl] for lvl in levels]
    
    # Plot Excalibr bars with individual colors
    for i, lvl in enumerate(levels):
        ax2.bar(x[i] - width/2, excalibr_ben_vals[i], width, 
               color=excalibr_colors[-lvl], alpha=0.9, 
               edgecolor='black', linewidth=1.2,
               label='ExCALIBR' if i == 0 else '')
    
    # Plot OddsPath bars with individual colors
    for i, lvl in enumerate(levels):
        ax2.bar(x[i] + width/2, auth_ben_vals[i], width, 
               color=oddspath_colors[-lvl], alpha=0.9, 
               edgecolor='black', linewidth=1.2,
               label=r'Brnich $\it{et\ al.}$' if i == 0 else '')
    
    # Add value labels on bars
    for i in range(len(levels)):
        # Excalibr labels
        # if excalibr_ben_vals[i] > 0:
        ax2.text(x[i] - width/2, excalibr_ben_vals[i],
                f'{int(excalibr_ben_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        # OddsPath labels
        # if auth_ben_vals[i] > 0:
        ax2.text(x[i] + width/2, auth_ben_vals[i],
                f'{int(auth_ben_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
    
    ax2.set_xlabel('Evidence Level (Benign)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax2.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    # ax2.set_title('Benign Evidence (-1, -2, -4, -8)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'-{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax2.legend(fontsize=FONTSIZE_LEGEND, frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax2.text(-0.05, 1.08, '(B)', transform=ax2.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')
    
    # Overall title
    # fig.suptitle('ExCALIBR vs OddsPath: Datasets Reaching Each Evidence Level', 
    #              fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    # else:
    plt.show()
    
    return fig, (ax1, ax2)


# Two-panel plot (separate pathogenic and benign)
fig, axes = plot_evidence_comparison(
    excalibr_path_counts, excalibr_ben_counts,
    auth_path_counts, auth_ben_counts,
    save_path='/data/ross/assay_calibration/paper/num_datasets_reach_evidence.png'
)


# print(summary_df)


# In[446]:


dre_dir = "/data/ross/assay_calibration/paper/datasets_reached_evidence"
os.makedirs(dre_dir, exist_ok=True)
with open(f"{dre_dir}/excalibr_path_counts.pkl","wb") as f:
    pickle.dump(excalibr_path_counts, f)
with open(f"{dre_dir}/excalibr_ben_counts.pkl","wb") as f:
    pickle.dump(excalibr_ben_counts, f)
with open(f"{dre_dir}/auth_path_counts.pkl","wb") as f:
    pickle.dump(auth_path_counts, f)
with open(f"{dre_dir}/auth_ben_counts.pkl","wb") as f:
    pickle.dump(auth_ben_counts, f)


# In[259]:


### df_std = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')


# In[260]:


# df_std.Dataset.nunique()


# In[261]:


# df_all = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')
# df_all.drop_duplicates('Dataset')['nucleotide_or_aa'].value_counts()


# In[262]:


importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import insert_evidence_into_dataframe

'''
for clingen analysis we must consider all variants and not exclude searching over 2025 clinvar variants and non-controls
'''
df_with_evidence = insert_evidence_into_dataframe(
    df=df_clingen,
    variant_to_oob_points_all=variant_to_oob_points,
    dataset_configs=new_dataset_configs,
    make_variant_id=make_variant_id
)


# In[263]:


df_std.Dataset.nunique()


# In[264]:


# for i,row in df_with_evidence.iterrows():
#     if row['is_filtered']:
#         continue
#     if row['Dataset'] not in datasets_new_configs:
#         continue
    
#     clingen_class = row[clingen_class_col]
#     clingen_evidence = row[clingen_evidence_col]
#     if not pd.isna(clingen_class) and not pd.isna(clingen_evidence) and clingen_class != 'VUS':
#         clingen_evidence = clingen_evidence.split(',')

#         print(clingen_class, clingen_evidence, row['Applied Evidence Codes (Met)_ClinGen_repo'], sep=' | ')


# In[265]:


from acmg_evidence_codes import classify_acmg

remove_codes = {
    # Benign Evidence Codes to Remove
    # "BP4", "BP4_Supporting", "BP4_Moderate", "BP4_Strong", "BP4_Very",
    "BS3", "BS3_Supporting", "BS3_Moderate", "BS3_Strong", "BS3_Very",

    # Pathogenic Evidence Codes to Remove
    "PS3", "PS3_Supporting", "PS3_Moderate", "PS3_Strong", "PS3_Very",
    # "PP3", "PP3_Supporting", "PP3_Moderate", "PP3_Strong", "PP3_Very",
}

def filter_and_recalculate(evidence_string):
    if pd.isna(evidence_string):
        return "VUS", "" 
    
    # Split the string into a list
    evidence_list = evidence_string.split(",")
    old_classification = classify_acmg([code.strip() for code in evidence_list])

    # Remove unwanted codes
    filtered_evidence = [code.strip() for code in evidence_list if code.strip() not in remove_codes]

    # Convert back to comma-separated string
    filtered_evidence_string = ",".join(filtered_evidence) if filtered_evidence else ""

    # If nothing remains after filtering, return VUS
    updated_classification = classify_acmg(filtered_evidence) if filtered_evidence else "VUS"

    print(f'{evidence_string} ({old_classification}) --> {filtered_evidence_string} ({updated_classification})')

    return updated_classification, filtered_evidence_string

df_with_evidence[['Updated_Classification_ClinGen_repo', 'Updated_Evidence Codes_ClinGen_repo']] = df_with_evidence['Applied Evidence Codes (Met)_ClinGen_repo'].apply(lambda x: pd.Series(filter_and_recalculate(x)))

pd.set_option('display.max_columns', None)

# df_with_evidence = df_with_evidence.add_suffix("_ClinGen_repo")

# #add gold stan transcript
# gold_standard['gold_stan_transcript'] = gold_standard['Ref_seq_transcript_ID'].str.replace(
#     r"(NM_\d+)\.\d+", r"\1", regex=True
# )

# # Convert all relevant columns in both DataFrames to string type before merging
# columns_left_clingen = ['Gene',"gold_stan_transcript", 'hgvs_c','hgvs_p']
# columns_right_clingen = ['HGNC Gene Symbol_ClinGen_repo',"Transcript_ID_clingen_ClinGen_repo",'HGVS_c_clingen_ClinGen_repo', "Protein_Change_clingen_ClinGen_repo"]

# for col in columns_left_clingen:
#     gold_standard[col] = gold_standard[col].astype(str)

# for col in columns_right_clingen:
#     ClinGen_repo[col] = ClinGen_repo[col].astype(str)

# #merge
# gold_standard_clingen = pd.merge(
#     gold_standard, #<--- dataframe of choice (in my case its the pp dataframe without gold standard variants)
#     ClinGen_repo,
#     left_on=columns_left_clingen,
#     right_on=columns_right_clingen,
#     how='left'
# )

#columns to drop 

# columns_drop_final = [
# 'Variation_ClinGen_repo','HGVS Expressions_ClinGen_repo', 'HGNC Gene Symbol_ClinGen_repo',
# 'Transcript_ID_clingen_ClinGen_repo', 'HGVS_c_clingen_ClinGen_repo', 'Protein_Change_clingen_ClinGen_repo',
# 'gold_stan_transcript']


# gold_standard_clingen_2 = gold_standard_clingen.drop(columns=columns_drop_final)


# In[266]:


clingen_class_col = "Updated_Classification_ClinGen_repo"
clingen_evidence_col = "Updated_Evidence Codes_ClinGen_repo"

confusion = {
    'auth': np.zeros((3, 2), dtype=int),
    'excalibr': np.zeros((3, 2), dtype=int)
}

# rows: 0=Pathogenic, 1=Indeterminate, 2=Benign
# cols: 0=PLP, 1=BLB

def to_row(label):
    # +1, 0, -1 -> row index
    if label == 1:
        return 0
    elif label == 0:
        return 1
    elif label == -1:
        return 2

def to_col(label):
    # PLP=+1, BLB=-1
    return 0 if label == 1 else 1


def update_confusion(conf, y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        conf['TP'] += 1
    elif y_true == -1 and y_pred == -1:
        conf['TN'] += 1
    elif y_true == -1 and y_pred == 1:
        conf['FP'] += 1
    elif y_true == 1 and y_pred == -1:
        conf['FN'] += 1


seen_ids, seen_genes = set(), set()
for i,row in df_with_evidence.iterrows():
    if row['is_filtered']:
        continue
    if row['Dataset'] not in datasets_new_configs:
        continue
    
    clingen_class = row[clingen_class_col]
    clingen_evidence = row[clingen_evidence_col]
    if not pd.isna(clingen_class) and not pd.isna(clingen_evidence) and clingen_class != 'VUS':
        clingen_evidence = clingen_evidence.split(',')

        for evidence_line in clingen_evidence:
            assert not ("PS3" in evidence_line or "BS3" in evidence_line)
                # continue # ignore those with experimental evidence

        points, auth_class = row['danz_points'], row['author_label_standardized']

        if points < 0 and clingen_class in ["Likely Pathogenic","Pathogenic"]:
            print(f"{row['Dataset']}: {points} {auth_class} - {clingen_class} - {clingen_evidence}")

        if clingen_class == "Likely Pathogenic" or clingen_class == "Pathogenic":
            clingen_label = 1
        elif clingen_class == "Likely Benign" or clingen_class == "Benign":
            clingen_label = -1
        else:
            raise ValueError(clingen_class)

        auth_label = 0
        if auth_class == "Abnormal":
            auth_label = 1
        elif auth_class == "Normal":
            auth_label = -1

        excalibr_label = np.sign(points)# if abs(points) >= 4 else 0

        y_true = clingen_label              # +1 (PLP) or -1 (BLB)
        y_auth = auth_label                # +1, 0, -1
        y_exc  = excalibr_label            # +1, 0, -1

        if row['ID'] in seen_ids:
            continue
        seen_ids.add(row['ID'])
        seen_genes.add(row['Gene'])
        
        # Update matrices
        confusion['auth'][to_row(y_auth), to_col(y_true)] += 1
        confusion['excalibr'][to_row(y_exc), to_col(y_true)] += 1

print(len(seen_genes), 'genes retained clingen evidence')


# In[267]:


#### import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_2x3_confusions_nature(conf_dict, figsize=(13, 4)):

    # --- SAME colormaps as letters=True ---
    blue_colors = ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E']
    red_colors  = ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744']
    gray_colors = ['#F5F5F5', '#CCCCCC', '#999999', '#666666']

    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", blue_colors)
    red_cmap  = LinearSegmentedColormap.from_list("red_gradient", red_colors)
    gray_cmap = LinearSegmentedColormap.from_list("gray_gradient", gray_colors)

    # --- Manual layout copied from your original ---
    fig = plt.figure(figsize=figsize)

    left_margin = 0.08
    bottom_margin = 0.15
    top_margin = 0.12
    plot_width = 0.35
    space_left = 0.05

    ax_ex = fig.add_axes([left_margin, bottom_margin, plot_width, 1 - bottom_margin - top_margin])
    ax_auth = fig.add_axes([left_margin + plot_width + space_left, bottom_margin,
                            plot_width, 1 - bottom_margin - top_margin])

    axes = {
        "ExCALIBR Evidence": (ax_ex, conf_dict['excalibr']),
        "Author Annotations": (ax_auth, conf_dict['auth'])
    }

    def plot_heatmap(mat, ax):
        rows, cols = mat.shape  # rows=2, cols=3

        for i in range(rows):
            row_max = mat[i].max()

            for j in range(cols):
                value = mat[i, j]

                # --- COLUMN SEMANTICS (Benign, Indet, Path) ---
                if j == 0:        # Benign / Normal
                    cmap = blue_cmap
                elif j == 1:      # Indeterminate
                    cmap = gray_cmap
                elif j == 2:      # Pathogenic / Abnormal
                    cmap = red_cmap

                norm = value / row_max if row_max > 0 else 0
                color = cmap(norm)

                rect = mpatches.Rectangle(
                    (j, i), 1, 1,
                    facecolor=color,
                    edgecolor='white',
                    linewidth=2.5
                )
                ax.add_patch(rect)

                text_color = 'white' if norm > 0.45 else 'black'
                if j == 1:  # Indeterminate always black text
                    text_color = 'white' if norm > 0.7 else 'black'

                ax.text(
                    j + 0.5, i + 0.5, f"{value:,}",
                    ha='center', va='center',
                    fontsize=16,# fontweight='bold',
                    color=text_color
                )

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        ax.set_aspect('equal')

    # --- Plot panels ---
    for title, (ax, mat) in axes.items():
        plot_heatmap(mat, ax)

        ax.set_facecolor('#F9F9F9')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)

        ax.set_xticks([0.5, 1.5, 2.5])

        if "ExCALIBR" in title:
            ax.set_xticklabels(
                ['Benign', 'Indeterminate', 'Pathogenic'],
                fontsize=12
            )
        else:
            ax.set_xticklabels(
                ['Normal', 'Indeterminate', 'Abnormal'],
                fontsize=12
            )

        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['P/LP', 'B/LB'][::-1], fontsize=13)

        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Panel letters (same placement) ---
    ax_ex.text(-0.10, 1.11, "(A)", transform=ax_ex.transAxes,
               fontsize=18, fontweight='bold', va='top')
    ax_auth.text(-0.10, 1.11, "(B)", transform=ax_auth.transAxes,
                 fontsize=18, fontweight='bold', va='top')

    
    ax_ex.set_xlabel('Evidence Direction', fontsize=14)
    ax_ex.set_ylabel('ClinGen Classification', fontsize=14)
        
    ax_auth.set_xlabel('Functional Annotation', fontsize=14)
    ax_auth.set_ylabel('')

    return fig


def convert_3x2_to_2x3(mat_3x2):
    # rows: P, I, B
    # cols: PLP, BLB
    return np.array([
        [mat_3x2[2,1], mat_3x2[1,1], mat_3x2[0,1]],  # BLB row
        [mat_3x2[2,0], mat_3x2[1,0], mat_3x2[0,0]],  # PLP row
    ])

fig = plot_2x3_confusions_nature({
    'auth': convert_3x2_to_2x3(confusion['auth']),
    'excalibr': convert_3x2_to_2x3(confusion['excalibr'])
})
plt.show()
fig.savefig('/data/ross/assay_calibration/paper/clingen_confusion.png',dpi=300, bbox_inches='tight')


# In[268]:


# df_test = pd.read_csv('/data/ross/assay_calibration/Dan_fxn_calibrations_new_2025_2018_clust_calib_2025_12_27.csv.gz')

# [col for col in df_test.columns if "points" in col.lower()]


# In[271]:


from src.assay_calibration.plot_utils.utils import compute_classification_metrics

def latex_performance_table(conf_dict):
    ex = compute_classification_metrics(pd.DataFrame(conf_dict['excalibr']))
    au = compute_classification_metrics(pd.DataFrame(conf_dict['auth']))

    def pct(n, d):
        return 100 * n / d if d > 0 else 0.0

    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    total = ex['total']  # same for both

    table = f"""
\\begin{{table}}[!tb]
\\centering
\\caption{{Performance comparison of out-of-bag \\excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants. Pathogenic and benign refer to the direction of evidence assigned.}}
\\label{{tab:author_performance}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Calibrated Evidence & Author Annotations \\\\
\\midrule
Total variants & {total:,} & {total:,} \\\\
Determinate assignments & {ex['determinate']:,} ({pct(ex['determinate'], total):.1f}\\%) & {au['determinate']:,} ({pct(au['determinate'], total):.1f}\\%) \\\\
Indeterminate assignments & {ex['uncertain']:,} ({pct(ex['uncertain'], total):.1f}\\%) & {au['uncertain']:,} ({pct(au['uncertain'], total):.1f}\\%) \\\\
\\midrule
Accuracy & {fmt(ex['accuracy'])} & {fmt(au['accuracy'])} \\\\
Sensitivity & {fmt(ex['sensitivity'])} & {fmt(au['sensitivity'])} \\\\
Specificity & {fmt(ex['specificity'])} & {fmt(au['specificity'])} \\\\
MCC & {fmt(ex['mcc'])} & {fmt(au['mcc'])} \\\\
\\midrule
LR$^+$ (pathogenic vs. benign) & {fmt(ex['lr_plus_standard'],1)} & {fmt(au['lr_plus_standard'],1)} \\\\
LR$^+$ (pathogenic vs. rest) & {fmt(ex['lr_plus_pathogenic'],1)} & {fmt(au['lr_plus_pathogenic'],1)} \\\\
LR$^+$ (benign vs. rest) & {fmt(ex['lr_plus_benign'],1)} & {fmt(au['lr_plus_benign'],1)} \\\\
DOR (pathogenic vs. benign) & {fmt(ex['dor_standard'],1)} & {fmt(au['dor_standard'],1)} \\\\
DOR (pathogenic vs. rest) & {fmt(ex['dor_pathogenic'],1)} & {fmt(au['dor_pathogenic'],1)} \\\\
DOR (benign vs. rest) & {fmt(ex['dor_benign'],1)} & {fmt(au['dor_benign'],1)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(table)

latex_performance_table({
    'excalibr': convert_3x2_to_2x3(confusion['excalibr']),
    'auth': convert_3x2_to_2x3(confusion['auth'])
})


# In[270]:


len(datasets_new_configs)


# In[ ]:





# In[ ]:





# In[ ]:





# # playground

# In[708]:


check = df_std#[pp_revel['Dataset'].isin(datasets_no_calc)]

variant_effect_measurements = len(check['ID'].unique())

check2 = check.drop_duplicates('ID')

aa = check2[check2['nucleotide_or_aa'] == 'aa']

nuc = check2[check2['nucleotide_or_aa'] == 'nucleotide']

aa_un = aa.drop_duplicates(['Gene','aa_ref','aa_pos','aa_alt'])

nuc_un = nuc.drop_duplicates(['Gene','ref_allele','alt_allele','hg38_start'])

full = pd.concat([aa_un, nuc_un]) #unique variant counts

variant_effect_measurements, len(full)


# In[709]:


# Get the first entry of each dataset
first_entries = df_std.groupby('Dataset').first().reset_index()

# Count how many have "IGVF_produced"
num_igvf = (first_entries['IGVF_produced'] == 'Yes').sum()

print(num_igvf)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#### importlib.reload(src.assay_calibration.plot_utils.utils)
from src.assay_calibration.plot_utils.utils import calculate_diagnostic_odds_ratio

# dor_results = calculate_diagnostic_odds_ratio(all_danz_oob_full, all_clinvar_full)


# In[264]:


all_clinvar_full[:,2].sum()


# In[279]:


len(datasets_new_configs)


# In[283]:


df_std = pd.read_csv("/data/ross/assay_calibration/scoresets_12_01_25/final_pillar_data_with_clinvar_18_25_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_111225.csv")


# In[284]:


for dataset in df_std.Dataset.unique():
    if dataset not in new_dataset_configs:
        print(dataset)


# In[110]:


# not_allowed = datasets_new_configs + ["F9_Popp_2025_model"]

allowed = ['TARDBP_Bolognesi_Faure_2019', 'BRCA2_unpublished',
       'BRCA2_Sahu_2023_exon13_Cisplatin',
       'BRCA2_Sahu_2023_exon13_global_score',
       'BRCA2_Sahu_2023_exon13_Olaparib', 'BRCA2_Sahu_2023_exon13_SGE',
       'BRCA2_Sahu_2025_HDR', 'CBS_Sun_2020_high_B6',
       'CBS_Sun_2020_low_B6',
       'HMBS_van_Loggerenberg_2023_erythroid',
       'HMBS_van_Loggerenberg_2023_ubquitous',
       'CALM1_CALM2_CALM3_Weile_2017', 'TPK1_Weile_2017','PAX6_McDonnell_2024_BLX_no_geneticin',
 'CHEK2_Gebbia_2024',
 'SGCB_Li_2023',
 'PAX6_McDonnell_2024_LE9_geneticin',
 'PAX6_McDonnell_2024_LE9_no_geneticin',
 'PAX6_McDonnell_2024_BLX_geneticin']

# Filter to variants NOT in calibrated datasets
df_not_calibrated = df_std[df_std['Dataset'].isin(allowed)].copy()

print(f"Total datasets in df_std: {df_std['Dataset'].nunique():,}")
print(f"Datasets in allowed: {len(allowed)}")
print(f"\nRows in non OP datasets: {len(df_not_calibrated):,}")

# Count genes and datasets FIRST (from full data with duplicates)
n_unique_genes = df_not_calibrated['Gene'].nunique()
n_unique_datasets = df_not_calibrated['Dataset'].nunique()

# THEN deduplicate for unique variant count
group_cols = ['Dataset', 'Gene', 'Chrom', 'hg38_start', 'ref_allele', 'alt_allele']
df_unique_variants = df_not_calibrated.drop_duplicates(subset=group_cols, keep='first')
n_unique_variants = len(df_unique_variants)

# print(f"\n{'='*80}")
# print("NON-CALIBRATED VARIANTS SUMMARY")
# print('='*80)
print(f"Unique genes: {n_unique_genes:,}")
print(f"Unique datasets: {n_unique_datasets:,}")
print(f"Unique variants: {n_unique_variants:,}")
print(f"Total rows (with duplicates): {len(df_not_calibrated):,}")

# # Show dataset breakdown
# print(f"\nNon-calibrated datasets (top 10 by variant count):")
# dataset_counts = df_unique_variants['Dataset'].value_counts().head(10)
# for dataset, count in dataset_counts.items():
#     print(f"  {dataset}: {count:,} unique variants")


# In[297]:


df_oddspath = pd.read_csv("/data/ross/assay_calibration/OP_clinvar18_25_122325.csv")


# In[358]:


# df_oddspath


# In[ ]:





# In[299]:


danzs_new_configs_oob[datasets_new_configs.index("BRCA1_Findlay_2018")]


# In[735]:


df_tsc2 = df_std[df_std.Dataset == "TSC2_tuberin_unpublished"]


# In[749]:


import numpy as np

# --- Splice filter (same logic as your method) ---
detects_splice = df_tsc2["splice_measure"].iloc[0] == "Yes"

if not detects_splice:
    df_tsc2 = df_tsc2[
        df_tsc2["simplified_consequence"].str.lower() != "splice region"
    ]
    df_tsc2 = df_tsc2[
        df_tsc2["simplified_consequence"].str.lower() != "splice_site_variant"
    ]
    df_tsc2 = df_tsc2[
        (df_tsc2["spliceAI_DS_AG"] < 0.2) &
        (df_tsc2["spliceAI_DS_AL"] < 0.2) &
        (df_tsc2["spliceAI_DS_DG"] < 0.2) &
        (df_tsc2["spliceAI_DS_DL"] < 0.2)
    ]

# --- ClinVar reliability filter ---
zero_star_statuses = {
    np.nan,
    'no assertion criteria provided',
    'no assertion provided',
    'no interpretation for the single variant',
    'no classification provided',
    '-'
}

plp_set = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}

mask = (
    (df_tsc2["Flag"] != "*")
    & df_tsc2["clinvar_sig_2025"].isin(plp_set)
    & ~df_tsc2["clinvar_star_2025"].isin(zero_star_statuses)
)

df_plp = df_tsc2.loc[mask].copy()
auth_scores = df_plp["auth_reported_score"]


# In[751]:


df_plp['ID'].nunique()


# In[740]:


plt.hist(auth_scores)


# In[752]:


plt.hist(auth_scores)


# In[ ]:





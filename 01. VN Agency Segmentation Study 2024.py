# Databricks notebook source
# MAGIC %run /Repos/dung_nguyen_hoang@mfcgd.com/Utilities/Functions

# COMMAND ----------

import pyspark.sql.functions as F 
import pyspark.sql.types as T
from pyspark.sql import Window
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Load paths

# COMMAND ----------

cutoff_date= '2024-04-30'
mth_partition = cutoff_date[:7] #'2024-03'

in_path = '/dbfs/mnt/lab/vn/project/cpm/datamarts/'
# tagtdm_daily_path = '/dbfs/mnt/prod/Curated/VN/Master/VN_CURATED_DATAMART_DB/TAGTDM_DAILY'
agent_frm_path = 'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/AGENT_RFM/monthend_dt=' + mth_partition
agent_scorecard_path = 'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/AGENT_SCORECARD'
agent_mapping_path ='abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_REPORTS_DB/LOC_TO_SM_MAPPING/'
channel_path = 'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TAGTDM_MTHEND'
branch_path = 'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_CAS_DB/TBRANCHES/CAS_TBRANCHES.parquet'
customer_seg_path = f'/mnt/prod/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/INCOME_BASED_DECILE_AGENCY/image_date={cutoff_date}'
customer_path= f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TCUSTDM_MTHEND/image_date={cutoff_date}'
# each agt_code is mapped to a loc_cd which subsequently mapped under a series of sm_code/names

policy_dm_path = f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TPOLIDM_MTHEND/image_date={cutoff_date}'

# output path to store working data by mth_partitions
out_path = '/dbfs/mnt/lab/vn/project/scratch/agent_activation/'

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. load agent tables and filter for agency channel

# COMMAND ----------

# load branch info
br_df = spark.read.parquet(branch_path).toPandas()
# load agent df, scorecard and agent hierarchy mapping
agent_df = pd.read_parquet(f'{in_path}TPARDM_MTHEND/')
agent_scorecard_df = spark.read.parquet(agent_scorecard_path).toPandas()
agent_mapping_df = spark.read.parquet(agent_mapping_path).toPandas()

# retrieve agency flag to identify the agency channel
channel_df = spark.read.parquet(channel_path) #load agent channel info for filtering purpose
filtered_channel_df = channel_df.filter(channel_df.image_date == cutoff_date)
filtered_channel_df = filtered_channel_df[['agt_code','channel']].toPandas().drop_duplicates()
agency_agents_list = filtered_channel_df[filtered_channel_df['channel']=='Agency']['agt_code'].unique()

# filter for non terminated agents in agency channel
agent_frm_df = spark.read.parquet(agent_frm_path).toPandas() #load agent frm table
active_agents_df = agent_frm_df[(agent_frm_df['agt_status']=='Active') 
                                # & (agent_frm_df['AGENCY_IND']==1) # this field doesn't flag out agency PSM+ so not used here
                                ].drop_duplicates()  
agency_agents_df = active_agents_df[active_agents_df['agt_code'].isin(agency_agents_list)]

# COMMAND ----------

# filter for latest image date
col_list=['agt_cd','tier','14m_per','no_sbw_completed','last_mth_pol','last_3m_pol','last_6m_pol','last_9m_pol','last_12m_pol','last_12m_ape','last_12m_prd']
latest_df = agent_df[col_list][agent_df['image_date']==cutoff_date].drop_duplicates()

# append the latest agent tier view to identify non-active agent later
#latest_scorecard_df = agent_scorecard_df[agent_scorecard_df['monthend_dt']==pd.to_datetime(cutoff_date)][['agt_code','agent_tier','agent_taskforce']].drop_duplicates()
latest_scorecard_df = agent_scorecard_df[pd.to_datetime(agent_scorecard_df['monthend_dt']).dt.date == pd.to_datetime(cutoff_date).date()][['agt_code', 'agent_tier', 'agent_taskforce']].drop_duplicates()

# COMMAND ----------

agency_agents_df = agency_agents_df.merge(latest_scorecard_df, on='agt_code',how='left')
# Append Branch name
agency_agents_df = agency_agents_df.merge(br_df[['BR_CODE','BR_NM']].drop_duplicates(), left_on='br_code', right_on='BR_CODE')
#agency_agents_df.shape

# COMMAND ----------

# merge useful columns
col_list = ['agt_cd','tier','14m_per','no_sbw_completed','last_mth_pol','last_3m_pol','last_6m_pol','last_9m_pol','last_12m_pol','last_12m_ape','last_12m_prd']
agent_merged_df = agency_agents_df.merge(latest_df[col_list].drop_duplicates(), left_on='agt_code', right_on='agt_cd', how='left')
#agent_merged_df = agent_merged_df.rename(columns={'last_12m_ape': 'last_yr_ape'})
agent_merged_df.shape
#agent_merged_df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. categorize agent features for segmentation

# COMMAND ----------


# Define bins and labels for each column to be binned
bins_labels = [
    # For 'agt_tenure_mths', create two categories: '<=1 year' and '>1 year'
    ('agt_tenure_mths', [0, 12, float('inf')], ['<=1 year', '>1 year']),
    
    # For '14m_per', create two categories: '<=70%' and '70%+'
    ('14m_per', [0, 0.7, 1], ['<=70%', '70%+']),
    
    # For 'last_12m_prd', create two categories: '<= 2 product types' and '> 2 product types'
    ('last_12m_prd', [-1, 2, float('inf')], ['<=2 product types', '>2 product types']),
    
    # For 'last_12m_ape', create four categories: '0-5000', '5000-7000', '7000-10000' and '10000+'
    ('last_12m_ape', [0, 5000, 7000, 10000, float('inf')], ['<=5k', '5-7k', '7-10k', '10k+']),
    
    # For '1yr_agent_activeness', create three categories: 'Low', 'Medium', and 'High'
    ('1yr_agent_activeness', [0, 0.3, 0.6, 1], ['Low', 'Medium', 'High']),
    
    # For 'cus_existing', create two categories: '<= 2 customers' and '> 2 customers'
    ('cus_existing', [0, 2, float('inf')], ['<=2 customers', '>2 customers']),
    
    # For 'no_sbw_completed', create two categories: 'No trainings completed' and 'at least 1 training completed'
    ('no_sbw_completed', [-1, 0, float('inf')], ['No trainings completed', 'at least 1 training completed'])
]

# Apply the function to each feature
for column, bins, labels in bins_labels:
    create_categorical(agent_merged_df, column, bins, labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Append agent hierarchy relationship

# COMMAND ----------

# Define the columns to keep
cols_kept = [
    'eval_dt', 'agt_code', 'agt_tenure_mths', 'tier', 'agent_tier', 'agent_taskforce', 'loc_code', 'br_code', 'BR_NM',
    'unit_code', 'no_sbw_completed', 'no_sbw_completed_cat', 'last_mth_pol','last_3m_pol','last_6m_pol','last_9m_pol','last_12m_pol', 
    'last_12m_ape', 'last_12m_prd', 'agt_tenure_mths_cat', '14m_per','14m_per_cat', 'last_12m_ape_cat', '1yr_agent_activeness', '1yr_agent_activeness_cat',
    'last_12m_prd_cat', 'cus_existing_cat'
]

# Keep only the defined columns in the DataFrame
agent_merged_df = agent_merged_df[cols_kept]

# Define the columns to use for mapping managers to agents
sim_mapping_df = agent_mapping_df[['loc_cd', 'manager_name_0', 'manager_name_1', 'manager_name_2', 'manager_name_3', 'manager_name_4',
                                   'manager_name_5', 'manager_name_6', 'RH_name']].copy()

sim_mapping_df['first_sm_name'] = sim_mapping_df[['manager_name_0', 'manager_name_1', 'manager_name_2', 'manager_name_3',
                                                  'manager_name_4', 'manager_name_5', 'manager_name_6']].fillna(method='ffill').iloc[:, 0]

sim_mapping_df = sim_mapping_df[['loc_cd', 'first_sm_name', 'RH_name']].drop_duplicates()

# Merge the manager mapping into the main DataFrame
agent_merged_df = agent_merged_df.merge(sim_mapping_df, left_on='loc_code', right_on='loc_cd', how='left')

# Define the columns to use for appending agent info for up manager level 1
#cols = ['agt_code','agent_tier','1yr_agent_activeness_cat','last_yr_ape_cat']

# Rename the columns for merging
'''manager_0_df = agent_merged_df[cols].rename(
    columns={
        'agt_code':'manager_code_0',
        'agent_tier':'agent_tier_manager_0',
        '1yr_agent_activeness_cat':'1yr_agent_activeness_cat_mgr_0',
        'last_yr_ape_cat':'last_yr_ape_cat_mgr_0'
    }
)

# Merge the manager info into the main DataFrame
agent_merged_df = agent_merged_df.merge(manager_0_df, on='manager_code_0', how='left')
'''
# Create a new column to indicate if manager level 0 is active or not
#(agent_merged_df['agent_tier_manager_0'] != '3mZ'), # Change to 9mZ
agent_merged_df['manager_0_active'] = np.where( 
    (agent_merged_df['last_9m_pol'] > 0), # New definition for Active/Inactive agent
    'Active', 
    'Inactive'
)
#agent_merged_df['manager_0_active'] = 'Active'
#grouped = agent_merged_df.groupby('manager_code_0')['last_9m_pol'].transform(lambda x: (x.notnull() & (x == 0)).all())
#agent_merged_df.loc[~grouped, 'manager_0_active'] = 'Inactive'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 build agent waterfall to flag out candidate agents for campaign targeting 

# COMMAND ----------

# Add a new column 'f_agt_inactive' to flag inactive agents
#agent_merged_df['f_agt_inactive'] = agent_merged_df['agent_tier'].apply(lambda x: 1 if x == '3mZ' else 0)
agent_merged_df['f_agt_inactive'] = agent_merged_df['last_9m_pol'].apply(lambda x: 1 if x == 0 else 0)

# Add a new column 'f_target_agent' to flag target agents
agent_merged_df['f_target_agent'] = np.where(
    (agent_merged_df['agt_tenure_mths_cat'] == '>1 year') & 
    (agent_merged_df['last_12m_ape_cat'].isin(['5-7k','7-10k','10k+'])) &
    (agent_merged_df['cus_existing_cat'] == '>2 customers') &
    (agent_merged_df['14m_per_cat'] == '70%+') #&  
    #(agent_merged_df['last_12m_prd_cat'] == '>2 product types') & # Remove this filter
    ,#(agent_merged_df['manager_0_active']=='Active'), 
    1, 
    0
)
print("Number of targeted agents for activation:", len(agent_merged_df[agent_merged_df['f_target_agent']==1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. load customer tables
# MAGIC
# MAGIC Filter priortized customers for campaign :
# MAGIC - Decile < 3 
# MAGIC - protection_income_grp in [3. 20% Income Protection, 4. below 20% Income Protection, 5. No dependent/Protection Pols]
# MAGIC - income_segment in [1 VIP, 2 HNW, 3 High Income, 4 Mid Income]
# MAGIC - remove those with term products only

# COMMAND ----------

# load customer and customer segmentation
cols = ['po_num', 'min_prem_dur', 'first_pol_eff_dt', 'sex_code', 'client_tenure', 'tot_face_amt_usd', 'pol_count', 'ins_typ_count', 'term_pol', 'endow_pol',
        'health_indem_pol', 'whole_pol', 'investment_pol', 'inforce_pol', 'f_owner_is_agent', 'agt_tenure_yrs', 'mdrt_flag', 'mdrt_tot_flag', 'mdrt_cot_flag', 'active_1m_flag', 'multi_prod', 'annual_flag', 'valid_email', 'valid_mobile', 'total_ape', 'plan_nbv', 'rider_ape', 'rider_cnt', 'cli_nbv_margin', 'channel_final', 'CITY', 'adj_mthly_incm', 'unassigned_ind', 'NO_DPND', 'f_HCM', 'f_HN', 'pol_cnt', '10yr_pol_cnt', 'f_vip_elite', 'f_vip_plat', 'f_vip_gold', 'f_vip_silver', 'existing_vip_seg', 'next_mat_date_mth', 'lst_purchase_mth', 'lst_eff_dt', 'last_termination_date', 'min_mat_date', 'protection_fa', 'protection_fa_all', 'f_term_pol', 'f_endow_pol', 'f_health_indem_pol', 'f_whole_pol', 'f_investment_pol', 'inforce_ind', 'f_with_rider', 'dpnd_grandpa_ind', 'dpnd_parent_ind', 'dpnd_spouse_ind', 'dpnd_child_ind', 'dpnd_sibling_ind', 'dpnd_oth_ind', 'f_dependent_ind', 'income_decile', 'protection_fa_imp', 'protection_fa_all_imp', 'f_with_dependent', 'protection_gap', 'protection_gap_all', 'wallet_rem'
        ]
cus_seg_df = spark.read.parquet(customer_seg_path)

cus_df= spark.read.format("parquet").load(customer_path, header=True)

policy_df = spark.read.parquet(policy_dm_path).alias('a').join(cus_seg_df.alias('b'), on='po_num')\
    .select('a.po_num', 'a.pol_eff_dt', 'a.sa_code')
    
#window = Window.partitionBy('po_num').orderBy(desc('pol_eff_dt'))
policy_df = policy_df.withColumn('rn', F.row_number().over(Window.partitionBy('po_num').orderBy(F.col('pol_eff_dt').desc())))
policy_df = policy_df.filter(F.col('rn')==1).drop_duplicates().toPandas()
#policy_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Filter for prioritized customers for agent reactivation campaign

# COMMAND ----------

cus_seg_df = cus_seg_df.toPandas()
fil_cus_seg_df = cus_seg_df[(cus_seg_df['f_owner_is_agent']==0)
    #(cus_seg_df['decile'].isin([1,2])) & 
    #(cus_seg_df['protection_income_grp'].isin(['5. No dependent/Protection Pols','4. below 20% Income Protection', '3. 20% Income Protection'])) & 
    #(cus_seg_df['income_segment'].isin(['1 VIP', '2 HNW', '3 High Income', '4 Mid Income'])) &
    #(~cus_seg_df['segmentation_rules'].isin(['1.2.2 Term Only : No TROP','2.2.2 Term Only : No TROP', '1.2.1 Term Only : TROP']))
    ]

# Get the list of these POs as filter
fil_cus_list = fil_cus_seg_df['po_num'].unique().tolist()
#fil_cus_seg_df.shape
print("Number of customers in-scope:", len(fil_cus_list))

# COMMAND ----------

policy_fil_df = policy_df[policy_df['po_num'].isin(fil_cus_list)].drop_duplicates(subset=['po_num', 'sa_code'])[['po_num', 'sa_code']]
#policy_fil_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>2.2 Identify agents who are in-scope and out-of-scope</strong>

# COMMAND ----------

# Merge agent_merged_df with policy_fil_df to get the agents serving the target customers
merged_df = agent_merged_df.merge(policy_fil_df, how='inner', left_on='agt_code', right_on='sa_code')

# Calculate 'po_num_count' for each 'agt_code'
po_num_count = merged_df.groupby('agt_code')['po_num'].nunique().reset_index()
po_num_count.columns = ['agt_code', 'po_num_count']

# Add 'po_num_count' to agent_merged_df
agent_merged_df = agent_merged_df.merge(po_num_count, on='agt_code', how='inner')

# Add 'in_scope' column based on the match between 'agt_code' and 'sa_code'
agent_merged_df['in_scope'] = 0
agent_merged_df.loc[agent_merged_df['agt_code'].isin(policy_fil_df['sa_code']), 'in_scope'] = 1
agent_merged_df['in_scope'] = agent_merged_df['in_scope'].astype(int)

agent_merged_df = agent_merged_df.copy()

conditions = [
    (agent_merged_df['last_mth_pol'] > 0),
    (agent_merged_df['last_3m_pol'] > 0),
    (agent_merged_df['last_6m_pol'] > 0),
    (agent_merged_df['last_9m_pol'] > 0),
    (agent_merged_df['last_12m_pol'] >0)
]

choices = ['1mA', '3mA', '6mA', '9mA', '12mA']

# Default value if none of the conditions are met
agent_merged_df['last_pol_cat'] = np.select(conditions, choices, default='12m-Inactive')

agent_inscope_df = agent_merged_df[agent_merged_df['in_scope']==1]

print("Unique agents in-scope:", len(pd.unique(agent_inscope_df['agt_code'])))

# COMMAND ----------

agent_merged_df.to_csv(f'/dbfs/mnt/lab/vn/project/scratch/agent_activation/agent_merged_total.csv', header=True, index=False)
#display(agent_inscope_df)

# COMMAND ----------

# Run Waterfall for Active agents
active_agents_df = agent_inscope_df[agent_inscope_df['f_agt_inactive']==0]
print("Number of active agents:", len(pd.unique(active_agents_df['agt_code'])))

# Define the conditions
conditions = [
    (active_agents_df['agt_tenure_mths_cat'] == '>1 year'),
    (active_agents_df['last_12m_ape_cat'].isin(['5-7k','7-10k','10k+'])), # Change filter here for wider range of selection
    (active_agents_df['cus_existing_cat'] == '>2 customers'),
    (active_agents_df['14m_per_cat'] == '70%+'),
    #(inactive_agencts_df['last_yr_prd_cat'] == '>2 product types'), # Remove this filter
    #(inactive_agencts_df['manager_0_active']=='Active')
]

# Apply the conditions one by one and print the number of agents filtered out after each condition
df = active_agents_df.copy()
initial_count = len(df)
for i, condition in enumerate(conditions, start=1):
    filtered_df = df[condition]
    filtered_out_count = len(df) - len(filtered_df)
    print(f"Number of agents filtered out after condition {i}: {filtered_out_count}")
    df = filtered_df

# COMMAND ----------

# filter for inactive agents
#inactive_agencts_df = agent_merged_df[agent_merged_df['f_agt_inactive']==0]
inactive_agencts_df = agent_inscope_df[agent_inscope_df['f_agt_inactive']==1]
print("Number of inactive agents:", len(pd.unique(inactive_agencts_df['agt_code'])))

# Define the conditions
conditions = [
    (inactive_agencts_df['agt_tenure_mths_cat'] == '>1 year'),
    (inactive_agencts_df['last_12m_ape_cat'].isin(['5-7k','7-10k','10k+'])), # Change filter here for wider range of selection
    (inactive_agencts_df['cus_existing_cat'] == '>2 customers'),
    (inactive_agencts_df['14m_per_cat'] == '70%+'),
    #(inactive_agencts_df['last_yr_prd_cat'] == '>2 product types'), # Remove this filter
    #(inactive_agencts_df['manager_0_active']=='Active')
]

# Apply the conditions one by one and print the number of agents filtered out after each condition
df = inactive_agencts_df.copy()
initial_count = len(df)
for i, condition in enumerate(conditions, start=1):
    filtered_df = df[condition]
    filtered_out_count = len(df) - len(filtered_df)
    print(f"Number of agents filtered out after condition {i}: {filtered_out_count}")
    df = filtered_df


# COMMAND ----------

# PRINT OUT THE # OF TARGET AGENTS LEFT FOR ACTIVE AGENTS
target_agents = active_agents_df[active_agents_df['f_agt_inactive']==0]
print(target_agents.shape)

# COMMAND ----------

# pivot table for Active agents
pivot_df = target_agents.groupby(['agt_tenure_mths_cat','last_pol_cat','last_12m_ape_cat','cus_existing_cat','14m_per_cat',#'last_yr_prd_cat'
                                    ])\
                            .agg({'agt_code': 'nunique'}).reset_index()
display(pivot_df)

# COMMAND ----------

# PRINT OUT THE # OF TARGET AGENTS LEFT FOR INACTIVE AGENTS
target_agents = inactive_agencts_df[inactive_agencts_df['f_agt_inactive']==1]
print(target_agents.shape)


# COMMAND ----------

# pivot table
pivot_df = target_agents.groupby(['agt_tenure_mths_cat','last_pol_cat','last_12m_ape_cat','cus_existing_cat','14m_per_cat',
                                        #'last_yr_prd_cat'
                                        ]).agg({'agt_code': 'nunique'}).reset_index()
display(pivot_df)

# COMMAND ----------

# Select only columns needed from the below list:
cols=['cli_num','cur_age','actvnes_stat','new_exist_stat','nat_code','occp_clas','no_dpnd','move_ind']
# Should only reload when there's update in the target customer segment
filtered_cus_df = cus_df.filter(cus_df['cli_num'].isin(fil_cus_list)).select(*cols)

filtered_cus_pd = filtered_cus_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC <strong> Store and reload the filtered_cus_df to save time</strong>

# COMMAND ----------

filtered_cus_pd.to_parquet(f'/dbfs/mnt/lab/vn/project/scratch/agent_activation/filtered_cus.parquet', engine='pyarrow')
#filtered_cus_pd = pd.read_parquet(f'/dbfs/mnt/lab/vn/project/scratch/agent_activation/filtered_cus.parquet')

# COMMAND ----------

cols = ['po_num', 'min_prem_dur', 'first_pol_eff_dt', 'sex_code', 'client_tenure', 'tot_face_amt_usd', 'pol_count', 'ins_typ_count', 'term_pol', 'endow_pol',
        'health_indem_pol', 'whole_pol', 'investment_pol', 'inforce_pol', 'f_owner_is_agent', 'agt_tenure_yrs', 'mdrt_flag', 'mdrt_tot_flag', 'mdrt_cot_flag', 'active_1m_flag', 'multi_prod', 'annual_flag', 'valid_email', 'valid_mobile', 'total_ape', 'plan_nbv', 'rider_ape', 'rider_cnt', 'cli_nbv_margin', 'channel_final', 'CITY', 'adj_mthly_incm', 'unassigned_ind', 'f_HCM', 'f_HN', 'pol_cnt', '10yr_pol_cnt', 'f_vip_elite', 'f_vip_plat', 'f_vip_gold', 'f_vip_silver', 'existing_vip_seg', 'next_mat_date_mth', 'lst_purchase_mth', 'lst_eff_dt', 'last_termination_date', 'min_mat_date', 'f_term_pol', 'f_endow_pol', 'f_health_indem_pol', 'f_whole_pol', 'f_investment_pol', 'inforce_ind', 'f_with_rider', 'dpnd_grandpa_ind', 'dpnd_parent_ind', 'dpnd_spouse_ind', 'dpnd_child_ind', 'dpnd_sibling_ind', 'dpnd_oth_ind', 'f_dependent_ind', 'income_decile', 'protection_fa', 'protection_fa_all',
        'protection_fa_imp', 'protection_fa_all_imp', 'f_with_dependent', 'protection_gap', 'protection_gap_all', 'wallet_rem', 'cur_age', 'actvnes_stat',
        'new_exist_stat', 'nat_code', 'occp_clas', 'no_dpnd', 'move_ind', 'sa_code' #, 'decile'
        ]
cus_merged_df = fil_cus_seg_df.merge(filtered_cus_pd,left_on="po_num",right_on="cli_num",how="left") \
                              .merge(policy_fil_df,on="po_num",how="left")

cus_merged_df = cus_merged_df[cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Merge prioritized customers with agent table

# COMMAND ----------

cus_agent_merged_df = cus_merged_df.merge(agent_merged_df, left_on='sa_code', right_on='agt_code', how='left')
cus_agent_merged_df['image_date'] = cutoff_date
print('# of unique inforce customers:', cus_agent_merged_df.shape[0])
for col in cus_agent_merged_df.columns:
    print(col)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TARGET CUSTOMER INCL. AGENT ACTIVATION DATE IS STORED HERE (USED BY "VN CSEG for M25_2024_ver2")

# COMMAND ----------

cus_agent_merged_df.to_parquet(f'{out_path}', partition_cols=['image_date'], engine='pyarrow')

# COMMAND ----------

# MAGIC %md
# MAGIC # The Below Section is for the Agent Segmentation Deck

# COMMAND ----------

# MAGIC %md
# MAGIC ###Slide 3 Agent Profiling

# COMMAND ----------

# get the agent counts for MPro distribution
tier_counts = target_agents['tier'].value_counts().reset_index()
tier_counts

# COMMAND ----------

# get the branch counts for target agents
branch_counts = target_agents['BR_NM'].value_counts()
small_branches = branch_counts[branch_counts < 20].index
target_agents['BR_NM_grouped'] = np.where(target_agents['BR_NM'].isin(small_branches), 'Others', target_agents['BR_NM'])

target_agents['BR_NM_grouped'].value_counts().reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slide 5

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>TARGET agents for Agency</strong>

# COMMAND ----------

# Filter for target agents
target_cus_agents_df = cus_agent_merged_df[cus_agent_merged_df['is_target_agent'] == 1]

# Split the DataFrame into active and inactive agents
active_cus_agents_df = target_cus_agents_df[target_cus_agents_df['f_agt_inactive'] == 0]
inactive_cus_agents_df = target_cus_agents_df[target_cus_agents_df['f_agt_inactive'] == 1]

# Group by 'f_agt_inactive' and calculate the mean, sum or count of the relevant columns
profile = target_cus_agents_df.groupby('f_agt_inactive').agg({
    'po_num':'nunique',    
    'no_dpnd': 'mean',  # Average number of dependent family members
    'cur_age_x': 'mean',  # Average age
    'adj_mthly_incm': 'mean',  # Average income
    'client_tenure': 'mean',  # Average vintage
    'inforce_count': 'mean',  # Total product holding
    'total_ape_usd': 'mean',  # Total APE
    'rider_cnt': 'mean',  # Total rider holding
}).reset_index()

profile

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>ALL agents for Agency</strong>

# COMMAND ----------

# Group by 'f_agt_inactive' and calculate the mean, sum or count of the relevant columns
profile_all_agents = cus_agent_merged_df.groupby('f_agt_inactive').agg({
    'po_num':'nunique',
    'no_dpnd': 'mean',  # Average number of dependent family members
    'cur_age_x': 'mean',  # Average age
    'adj_mthly_incm': 'mean',  # Average income
    'client_tenure': 'mean',  # Average vintage
    'inforce_count': 'mean',  # Total product holding
    'total_ape_usd': 'mean',  # Total APE
    'rider_cnt': 'mean',  # Total rider holding
}).reset_index()

profile_all_agents

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random checking

# COMMAND ----------

cus_agent_merged_df['po_num'].nunique()

# COMMAND ----------

cus_agent_merged_df[cus_agent_merged_df['f_agt_inactive'].isnull()]['sa_code'].nunique()

# COMMAND ----------

cus_agent_merged_df['f_agt_inactive'].value_counts(dropna=False)

# COMMAND ----------

chk = cus_agent_merged_df[cus_agent_merged_df['f_agt_inactive'].isnull()]['sa_code'].unique()
agent_frm_df[agent_frm_df['agt_code'].isin(chk)]['agt_status'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slide 4

# COMMAND ----------

def filter_agents(df, f_agt_inactive):
    print(df.shape)

    # Define the conditions
    conditions = [
        (df['agt_tenure_mths_cat'] == '>1 year'),
        (df['last_yr_ape_cat'].isin(['5-7k','7-10k','10k+'])),
        (df['cus_existing_cat'] == '>2 customers'),
        (df['14m_per_cat'] == '70%+') #,
        #(df['last_yr_prd_cat'] == '> 2 product types'),
        #(df['manager_0_active'] == 'Active')
    ]

    # Apply the conditions one by one and print the number of unique "po_num" after each condition
    initial_unique_po_num = df['po_num'].nunique()
    print(f"Initial unique po_num: {initial_unique_po_num}")
    for i, condition in enumerate(conditions, start=1):
        filtered_df = df[condition]
        unique_po_num = filtered_df['po_num'].nunique()
        filtered_out_unique_po_num = initial_unique_po_num - unique_po_num
        print(f"Unique po_num filtered out after condition {i}: {filtered_out_unique_po_num}")
        print(f"Remaining unique po_num after condition {i}: {unique_po_num}")
        df = filtered_df
        initial_unique_po_num = unique_po_num

# Filter for active agents
active_agents_df = cus_agent_merged_df[cus_agent_merged_df['f_agt_inactive'] == 0]
filter_agents(active_agents_df, f_agt_inactive=0)

# Filter for inactive agents
inactive_agents_df = cus_agent_merged_df[cus_agent_merged_df['f_agt_inactive'] == 1]
filter_agents(inactive_agents_df, f_agt_inactive=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### For all the Appendix slides

# COMMAND ----------

# Define the columns used for profiling and their bins
profile_columns = {
     'no_dpnd': [0, 1, 2, 3, np.inf],  # Bins for number of dependents
    'cur_age_x': [0, 25, 35, 55, np.inf],  # Bins for age
    'adj_mthly_incm': [0,1000, 2000, 3000, np.inf],  # Bins for income
     'client_tenure': [0, 1, 2, 3, 4, 5, np.inf],  # Bins for client tenure
     'inforce_count': [0, 1, 2, 3, np.inf],  # Bins for inforce count
    'total_ape_usd': [0, 1000, 2000, 3000, np.inf],  # Bins for total APE
     'rider_cnt': [0, 1, 2, 3,  np.inf],  # Bins for rider count
}

def plot_composition(df, column, title, bins=None):
    if bins is not None:
        df[column] = pd.cut(df[column], bins)
    # Print the value counts in table format
    print(df[column].value_counts(dropna=False).sort_index().reset_index())
    # plt.figure(figsize=(10, 6))
    # sns.countplot(x=column, data=df, order=df[column].value_counts().index.sort_values())
    # plt.title(title)
    # plt.xticks(rotation=45)
    # plt.show()

# Calculate the value counts for each column and plot the compositions for active agents
for column, bins in profile_columns.items():
    print(f"Processing column: {column} for active agents")
    plot_composition(active_cus_agents_df, column, f"{column} composition (Active Agents)", bins)

# Calculate the value counts for each column and plot the compositions for inactive agents
for column, bins in profile_columns.items():
    print(f"Processing column: {column} for inactive agents")
    plot_composition(inactive_cus_agents_df, column, f"{column} composition (Inactive Agents)", bins)

# COMMAND ----------

# Calculate the value counts for each column and plot the compositions for ALL active agents
for column, bins in profile_columns.items():
    print(f"Processing column: {column} for active agents")
    plot_composition(active_agents_df, column, f"{column} composition (Active Agents)", bins)

# COMMAND ----------


# Calculate the value counts for each column and plot the compositions for ALL inactive agents
for column, bins in profile_columns.items():
    print(f"Processing column: {column} for inactive agents")
    plot_composition(inactive_agents_df, column, f"{column} composition (Inactive Agents)", bins)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slide 14

# COMMAND ----------

# Get the top 15 cities for active and inactive agents
top_cities_active = active_cus_agents_df['city'].value_counts().nlargest(15).index
top_cities_inactive = inactive_cus_agents_df['city'].value_counts().nlargest(15).index

# Replace the cities not in the top 15 with 'Other' for active and inactive agents
active_cus_agents_df['city_grouped'] = active_cus_agents_df['city'].where(active_cus_agents_df['city'].isin(top_cities_active), 'Other')
inactive_cus_agents_df['city_grouped'] = inactive_cus_agents_df['city'].where(inactive_cus_agents_df['city'].isin(top_cities_inactive), 'Other')

# Plot the composition of the grouped cities for active and inactive agents
plot_composition(active_cus_agents_df, 'city_grouped', "Grouped city composition (Active Agents)")
plot_composition(inactive_cus_agents_df, 'city_grouped', "Grouped city composition (Inactive Agents)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Target Agents Base Segmentation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Load the dataset and select only customers mapped to active agents

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.getOrCreate()
merged_df = spark.read.parquet(f'{out_path[5:]}merged_target_activation.parquet').where(F.col('agt_cd').isNotNull())

# Check for the number of unique active agents, excluding 'Pure_UCM' customers
active_agt_list = merged_df.filter(F.col('pure_unassigned_label') != 'Pure_UCM')\
                    .select('agt_cd').distinct().rdd.flatMap(lambda x: x).collect()
print('# of agents:', len(active_agt_list))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Find the group of variable correlations with highest density

# COMMAND ----------

# Add a column to convert agent's tier to numeric (1-7)
merged_df = merged_df.withColumn(
    "f_current_tier",
     F.when(F.col("current_tier") == "Unranked" , F.lit(1))
    .when(F.col("current_tier") == "Silver"   , F.lit(2))
    .when(F.col("current_tier") == "Gold"     , F.lit(3))
    .when(F.col("current_tier") == "Platinum" , F.lit(4))
    .when(F.col("current_tier") == "MDRT"     , F.lit(5))
    .when(F.col("current_tier") == "COT"      , F.lit(6))
    .when(F.col("current_tier") == "TOT"      , F.lit(7))
    .otherwise(F.lit(0))
    ).withColumn(
    "agt_ape_usd",
    (F.col('agt_total_ape')/23.145).cast('float')
    ).toPandas()

# COMMAND ----------

merged_pd = merged_df

f_cols = ["segment", "current_tier", "br_nm", "agt_tenure_mths_cat", "cus_existing_cat", "last_yr_ape_cat", "14m_per_cat", "agt_cd",
          "f_agt_inactive", "agt_ape_usd", "agt_tenure_mths", "14m_per", "last_yr_prd", "last_yr_ape", "1yr_agent_activeness", "po_num_count", "all_pol_cnt", "gap_to_next_tier", "1yr_agent_activeness_cat", "po_num_count_cat", "all_pol_cnt_cat", "gap_to_next_tier_cat", "first_sm_name", "rh_name"
          ]
n_cols = ["agt_ape_usd", "agt_tenure_mths", "14m_per", "last_yr_prd", "last_yr_ape", "1yr_agent_activeness", "po_num_count", "all_pol_cnt"]
#c_cols = ["age_grp", "current_tier", "br_nm"]

group_nm = ['Top Performers', 'High Performers', 'The Slackers']

# Define the percentile thresholds for each segment
percentile_threshold = [0.75, 0.25]  # Top 25%, 75% and the rest of agents

# Define the thresholds for each metric
agt_ape_usd_high, agt_ape_usd_moderate = np.quantile(merged_pd["agt_ape_usd"], percentile_threshold)
agt_tenure_mths_high, agt_tenure_mths_moderate = np.quantile(merged_pd["agt_tenure_mths"], percentile_threshold)
m14_per_high, m14_per_moderate = np.quantile(merged_pd["14m_per"], percentile_threshold)
last_yr_prd_high, last_yr_prd_moderate = np.quantile(merged_pd["last_yr_prd"], percentile_threshold)
last_yr_ape_high, last_yr_ape_moderate = np.quantile(merged_pd["last_yr_ape"], percentile_threshold)
yr1_agent_activeness_high, yr1_agent_activeness_moderate = np.quantile(merged_pd["1yr_agent_activeness"], percentile_threshold)
po_num_count_high, po_num_count_moderate = np.quantile(merged_pd["po_num_count"], percentile_threshold)
all_pol_cnt_high, all_pol_cnt_moderate = np.quantile(merged_pd["all_pol_cnt"], percentile_threshold)

# Create a function to assign agents to segments
def assign_segment(row):
    (
        agt_ape_usd,
        agt_tenure_mths,
        m14_per,
        last_yr_prd,
        last_yr_ape,
        yr1_agent_activeness,
        po_num_count,
        all_pol_cnt,
    ) = row

    conditions = (
        #(row["agt_ape_usd"] >= agt_ape_usd_high) &
        #(row["14m_per"] >= m14_per_high) &
        #(row["last_yr_prd"] >= last_yr_prd_high) &
        #(row["last_yr_ape"] >= last_yr_ape_high) &
        (row["po_num_count"] >= po_num_count_high) |
        (row["all_pol_cnt"] >= all_pol_cnt_high)
    )

    if conditions.all():
        return group_nm[0]  # Top Performers

    conditions = (
        #(row["agt_ape_usd"] >= agt_ape_usd_moderate) &
        #(row["14m_per"] >= m14_per_moderate) &
        #(row["last_yr_prd"] >= last_yr_prd_moderate) &
        #(row["last_yr_ape"] >= last_yr_ape_moderate) &
        (row["po_num_count"] >= po_num_count_moderate) |
        (row["all_pol_cnt"] >= all_pol_cnt_moderate)
    )

    if conditions.all():
        return group_nm[1]  # High Performers

    return group_nm[2]  # The Slackers

# Apply the function to the DataFrame
merged_pd["segment"] = merged_pd[n_cols].apply(assign_segment, axis=1)


# COMMAND ----------

bins_labels = [
    ('1yr_agent_activeness', [0, 0.3, 0.6, 0.9, float('inf')], 
     ['9m+', '6-9m', '3-6m', '<3m']),
    ('po_num_count', [0, 3, 5, 7, 10, float('inf')],
     ['<=3', '4-5', '5-7', '7-10', '10+']),
    ('all_pol_cnt', [0, 10, 20, 30, 50, float('inf')],
     ['<=10', '11-20', '21-30', '31-50', '51+']),
]

for column, bins, labels in bins_labels:
    create_categorical(merged_pd, column, bins, labels)

# COMMAND ----------

# Rank the segments
ranked_agents = merged_pd[f_cols].sort_values(by="segment", ascending=False).drop_duplicates()

# Group the data by the "segment" column and calculate statistics
for group in group_nm:
    df = ranked_agents[ranked_agents['segment'] == group]
    n_agt_list = df['agt_cd'].nunique()
    print(f"This is what the {n_agt_list} of {group} looks like:")
    calculate_summary_stats(df, n_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Add extra categories and save result for further manipulation

# COMMAND ----------

# Save raw data for future analysis
merged_pd.drop(columns=['__index_level_0__','f_current_tier'], inplace=True)
merged_pd.to_csv(f'{out_path}merged_segmentations.csv', header=True, index=False)
ranked_agents.to_csv(f'{out_path}ranked_agents.csv', header=True, index=False)

# COMMAND ----------

ranked_agents.dtypes

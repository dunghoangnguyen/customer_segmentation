# Databricks notebook source
# MAGIC %run /Repos/dung_nguyen_hoang@mfcgd.com/Utilities/Functions

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Initialization

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from datetime import datetime, timedelta

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Prepare tables

# COMMAND ----------

image_date = '2024-02-29'
image_date_sht = image_date[:7].replace('-', '')
image_year = int(image_date[:4])

cseg_path = f'/mnt/prod/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/INCOME_BASED_DECILE_AGENCY/image_date={image_date}'
aseg_path = f'/mnt/lab/vn/project/cpm/datamarts/TPARDM_MTHEND/image_date={image_date}'
mclass_path = '/mnt/lab/vn/project/scratch/gen_rep_2023/prod_existing/11_multiclass_scored_base/'
target_path = f'/mnt/lab/vn/project/scratch/agent_activation/image_date={image_date}'
policy_path = f'/mnt/prod/Curated/VN/Master/VN_CURATED_DATAMART_DB/TPOLIDM_MTHEND/image_date={image_date}'
claim_path = f'/mnt/prod/Published/VN/Master/VN_PUBLISHED_CASM_CAS_SNAPSHOT_DB/TCLAIM_DETAILS/image_date={image_date}'
out_path = '/dbfs/mnt/lab/vn/project/scratch/agent_activation/'

print(image_date, image_date_sht, image_year)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Load and immediate tables for data preparation

# COMMAND ----------

# MAGIC %md
# MAGIC <strong> 1.2.1 Load tables</strong>

# COMMAND ----------

cseg_df = spark.read.parquet(f'{cseg_path}').toPandas()
aseg_df = spark.read.parquet(f'{aseg_path}').toPandas()
target_activation_df = spark.read.parquet(f'{target_path}').toPandas()

mclass_df = spark.read.csv(f'{mclass_path}multiclass_scored_{image_date_sht}.csv', header=True, inferSchema=True).toPandas()

target_activation_df.drop(columns=['channel','cur_age_y','rn','loc_code','__index_level_0__'], inplace=True)
target_activation_df.rename(columns={'cur_age_x': 'cur_age', 'tier': 'current_tier', 'channel_final': 'channel', 'protection_gap_v2': 'protection_gap'}, inplace=True)
print('# customers with 0 APE:', target_activation_df.shape[0])
# Remove customers having 0 APE
target_activation_df = target_activation_df[target_activation_df['total_ape_usd']>0]
print('# customers w/o 0 APE:', target_activation_df.shape[0])

# COMMAND ----------

# Temporarily add the 'total_ape' for selling agents until it's added to the Agent Segmentation
policy_df = spark.read.parquet(policy_path)
policy_df = policy_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)

agt_tot_ape_df = policy_df.groupby('wa_code')\
    .agg(
        F.sum(F.when(F.col('POL_EFF_DT') <= image_date, F.col('TOT_APE')/23.145)).cast('float').alias('agt_total_ape')
    ).toPandas().drop_duplicates()

# COMMAND ----------

# TEmporarily add the 'claim_6m_cnt' and 'claim_6m_amt' for customers until they're added to the Customer Segmentations
claim_df = spark.read.parquet(claim_path)
claim_df = claim_df.filter(
    (F.months_between(F.col('CLM_APROV_DT'), F.add_months(F.to_date(F.lit(image_date), 'yyyy-MM-dd'), -6)) <= 6) &
    (F.col('CLM_APROV_DT') <= F.to_date(F.lit(image_date), 'yyyy-MM-dd')) &
    (F.col('CLM_STAT_CODE').isin(['A'])) &
    (F.col('CLM_CODE').isin([3, 7, 8, 11, 27]))
).select('POL_NUM','CLM_ID','CLM_APROV_AMT')\
    
claim_df = claim_df.groupby('POL_NUM')\
    .agg(
        F.count('CLM_ID').alias('claim_6m_cnt'),
        F.sum('CLM_APROV_AMT').alias('claim_6m_amt')
    )

# Lowercase all column names in the DataFrame
for col in claim_df.columns:
    claim_df = claim_df.withColumnRenamed(col, col.lower())
#print(claim_df.count())

# COMMAND ----------

claim_po_df = policy_df.join(claim_df, on='pol_num')\
    .groupby('po_num')\
    .agg(
        F.sum(F.col('claim_6m_cnt')).cast('float').alias('clm_6m_cnt'),
        F.sum(F.col('claim_6m_amt')/23.145).cast('float').alias('clm_6m_amt')
    ).toPandas().drop_duplicates()

claim_po_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC <strong> 1.2.2 Immediate tables</strong>

# COMMAND ----------

# Get only multiclass scores that have been successfully deployed
mclass_df = mclass_df[mclass_df['DEPLOYMENT_APPROVAL_STATUS']=='APPROVED']
mclass_df['po_num'] = mclass_df['po_num'].astype(str)

target_po_list = target_activation_df['po_num'].unique()
target_agt_list = target_activation_df['agt_code'].unique()

cseg_df = cseg_df[cseg_df['po_num'].isin(target_po_list)]
aseg_df = aseg_df[aseg_df['agt_cd'].isin(target_agt_list)]

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Identify customer segments through Product Affinity

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Merge multiclass scores to target customers/agents

# COMMAND ----------

mclass_cols = ['po_num','rep_purchase_comb_health_base_PREDICTION','rep_purchase_comb_health_rider_PREDICTION', 'rep_purchase_comb_inv_base_PREDICTION','rep_purchase_comb_riders_PREDICTION','rep_purchase_comb_term_base_PREDICTION', 'rep_purchase_comb_PREDICTION'
               ]
cseg_cols = ['po_num','sex_code','dpnd_child_ind','dpnd_spouse_ind','existing_vip_seg','f_trmn_0_6m','f_trmn_6_12m','f_trmn_12_18m','f_vip_elite','f_vip_gold','f_vip_plat','f_vip_silver','ins_typ_count','total_ape','tot_face_amt_usd','wallet_rem','claim_amount'
             ]

aseg_cols = ['agt_cd','next_tier','next_tier_benchmark','gap_to_next_tier','all_pol_cnt'
]

merged_target_activation_df = target_activation_df\
  .merge(mclass_df[mclass_cols], on='po_num', how='left')\
  .merge(cseg_df[cseg_cols], on='po_num', how='left')\
  .merge(claim_po_df, on='po_num', how='left')\
  .merge(aseg_df[aseg_cols], left_on='agt_code', right_on='agt_cd', how='left')\
  .merge(agt_tot_ape_df, left_on='agt_code', right_on='wa_code', how='left')

merged_target_activation_df.columns = map(str.lower, merged_target_activation_df.columns)

# Select numeric columns
numeric_columns = merged_target_activation_df.select_dtypes(include=['float32', 'float64']).columns

# Fill NaN with 0 for each numeric column
for col in numeric_columns:
    merged_target_activation_df[col] = merged_target_activation_df[col].fillna(0)

# Add 6m claim over APE ratio column
merged_target_activation_df['clm_6m_ratio'] = merged_target_activation_df['clm_6m_amt']*100 / merged_target_activation_df['total_ape'].round(4)

# Fill NaN with N/A for category and object columns
categorical_columns = merged_target_activation_df.select_dtypes(include=['category']).columns

for col in categorical_columns:
    merged_target_activation_df[col] = merged_target_activation_df[col].cat.add_categories("N/A")
    merged_target_activation_df[col] = merged_target_activation_df[col].fillna("N/A")

print(merged_target_activation_df.shape)
merged_target_activation_df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Group all metrics for slicers
# MAGIC <strong>Customer base – 80k</strong><br>
# MAGIC •	Age segment - <25, 25-35, 35-45, 45-55, >55<br>
# MAGIC •	Income segment - $500-1k, 1-1.5k, 1.5-2k, 2-3k, 3-5k, 5k+<br>
# MAGIC •	Family status – single, married, married with children<br>
# MAGIC •	No. of children – 1, 2, 3, >3<br>
# MAGIC •	Current Product Holding<br>
# MAGIC •	Product level propensity (based on multiclass model)<br>
# MAGIC •	Location/ Branch/ SM<br>
# MAGIC •	Total APE<br>
# MAGIC •	Total Policy Face Amount<br>
# MAGIC •	Protect Gap<br>
# MAGIC •	MOB<br>
# MAGIC •	successful claims in last 6 months<br>
# MAGIC
# MAGIC <strong>Agent Base – 20k</strong>
# MAGIC •	Active/ Inactive<br>
# MAGIC •	Active – 0-3M, 3-6M, 6-9M, >9M<br>
# MAGIC •	Location/ Branch/ SM<br>
# MAGIC •	APE under agent<br>
# MAGIC •	No. of customers<br>
# MAGIC •	No. of policies<br>
# MAGIC •	MOB<br>
# MAGIC •	MDRT/ Manupro status<br>
# MAGIC •	APE required to reach next level (Gold to Plat, Silver to Gold etc.)<br>

# COMMAND ----------

# Define bins and labels for each column to be binned
bins_labels = [
    # For 'total_ape_usd'
    ('total_ape_usd', [0, 1000, 2000, 3000, 5000, 7000, 10000, float('inf')], 
     ['1. <=1k', '2. 1-2k', '3. 2-3k', '4. 3-5k', '5. 5-7k', '6. 7-10k', '7. >10k']),
    ('adj_mthly_incm', [500, 1000, 1500, 2000, 3000, 5000, float('inf')],
     ['1. 500-1k', '2. 1-1.5k', '3. 1.5-2k', '4. 2-3k', '5. 3-5k', '6. >5k']),
    ('protection_income%', [0, 10.01, 25, 50, float('inf')],
     ['1. >90%', '2. 75-90%', '4. 50-75%', '5. <50%']),
    ('no_dpnd', [0, 0.1, 1, 2, float('inf')],
     ['0', '1', '2', '3+']),
    ('clm_6m_cnt', [0, 0.1, 1, 2, 4, float('inf')],
     ['0', '1', '2', '3-4', '5+']),
    ('clm_6m_ratio', [0, 0.25, 0.5, 0.75, float('inf')],
     ['<= 25%', '<= 50%', '<=75%', '>75%']),
    ('ins_typ_count', [0, 1, 2, float('inf')],
     ['1', '2', '2+']),
    # Add new features
    ('cur_age', [0, 24, 30, 35, 40, 45, 50, 55, float('inf')],
     ['<25', '25-30', '31-35', '36-40', '41-45', '46-50', '51-55', '55+']),
    ('gap_to_next_tier', [0, 20000, 50000, 75000, 100000, 150000, 200000, 250000, 300000, float('inf')],
     ['<=20m', '20-50m', '50-75m', '75-100m', '100-150m', '150-200m', '200-250m', '250-300m', '300m+'])
]

# Apply the function to each feature
for column, bins, labels in bins_labels:
    create_categorical(merged_target_activation_df, column, bins, labels)

# COMMAND ----------

mar_conditions = [
    (merged_target_activation_df['dpnd_child_ind']==1),
    (merged_target_activation_df['dpnd_spouse_ind']==1)
]

mar_choices = ['1. Married w/ kids', '2. Married']

add_group_column(merged_target_activation_df, mar_conditions, mar_choices, 'mar_stat_cat', '3. Unknown')

vip_conditions = [
    (merged_target_activation_df['f_vip_elite']==1),
    (merged_target_activation_df['f_vip_plat']==1),
    (merged_target_activation_df['f_vip_gold']==1),
    (merged_target_activation_df['f_vip_silver']==1)
]

vip_choices = ['1. Elite Platinum', '2. Platinum', '3. Gold', '4. Silver']

add_group_column(merged_target_activation_df, vip_conditions, vip_choices, 'cus_vip_cat', '5. Unknown')

group_conditions = [
    ((merged_target_activation_df['mar_stat_cat'].isin(['2. Married', '3. Unknown'])) &
     (merged_target_activation_df['no_dpnd_cat'].isin(['2','3+']))),
    ((merged_target_activation_df['mar_stat_cat'].isin(['1. Married w/ kids'])) &
     (merged_target_activation_df['no_dpnd_cat'].isin(['1','2','3+']))),
    ((merged_target_activation_df['mar_stat_cat'].isin(['2. Married', '3. Unknown'])) &
     (merged_target_activation_df['no_dpnd_cat']=='1')),
    (merged_target_activation_df['no_dpnd_cat']=='0')
]

group_choices = ['1. Family Guardians', '2. Mature Couples', '3. Younger Couples', '4. Personal Protection Seekers']

add_group_column(merged_target_activation_df, group_conditions, group_choices, 'group', '5. Unknown')

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.3 Store detailed data for future sizing

# COMMAND ----------

# Save raw data for future analysis
merged_target_activation_df.to_parquet(f'{out_path}merged_target_activation.parquet', engine='pyarrow')
#merged_target_activation_df.to_csv(f'{out_path}merged_target_activation.csv', header=True, index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Analysis

# COMMAND ----------

# Reload the saved data
merged_target_activation_df = pd.read_parquet(f'{out_path}merged_target_activation.parquet', engine='pyarrow')
merged_target_activation_df.shape

# COMMAND ----------

# Remove unassigned customers from analysis
nonucm_target_activation_df = merged_target_activation_df[merged_target_activation_df['pure_unassigned_label']=='Not Pure UCM']
nonucm_target_activation_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Customer Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC <strong> Break-down by Insurance type affinity, age group, income band, protection gap and tenure

# COMMAND ----------

result_df = nonucm_target_activation_df.groupby(['segmentation_rules', 'rep_purchase_comb_prediction', 'age_grp', 'adj_mthly_incm_cat', #'total_ape_usd_cat', 
                                                 'protection_income_grp', 'tenure_grp', #'br_nm'
                                                 ])\
    .agg({
        'po_num': 'nunique'
    })\
    .reset_index()\
    .loc[lambda x: x['po_num'] > 0]

# Displaying the result
#result_df

# COMMAND ----------

result_df.to_csv(f'{out_path}target_activation_analysis_v1.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Break down by Age group, Income band, Family status, Dependants, Product holding, Protection gap, 6m Claim and Tenure

# COMMAND ----------

result_df = nonucm_target_activation_df.groupby(['mar_stat_cat', 'no_dpnd_cat', 'age_grp', 'adj_mthly_incm_cat', 'protection_income_grp', 'protection_income%_cat', 'ins_typ_count_cat', 'clm_6m_ratio_cat', 'tenure_grp', 'cus_vip_cat'
                                                 ])\
    .agg({
        'po_num': 'nunique'
    }).reset_index()\
    .loc[lambda x: x['po_num'] > 0]

#result_df

# COMMAND ----------

result_df.to_csv(f'{out_path}target_activation_analysis_v2.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Group by New Segment (group), Protection Gap (protection_income%_cat) and Product holding(segmentation_rules)</strong>

# COMMAND ----------

result_df = nonucm_target_activation_df.groupby(['group', 'protection_income%_cat', 'segmentation_rules',
                                                 'cur_age_cat'
                                                 ])\
    .agg({
        'po_num': 'nunique'
    }).reset_index()\
    .loc[lambda x: x['po_num'] > 0]

result_df

# COMMAND ----------

result_df.to_csv(f'{out_path}target_activation_analysis_v3.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Categorization and Profiling

# COMMAND ----------

# MAGIC %md
# MAGIC ### The 4 main groups:<br>
# MAGIC <strong>1. Family Guardians<br>
# MAGIC 2. Mature Couples<br>
# MAGIC 3. Younger Couples<br>
# MAGIC 4. Personal Protection Seekers<br></strong>

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Family Guardians</strong>

# COMMAND ----------

# Select the columns of interest for statistics calculations
columns_of_interest = ['cur_age', 'adj_mthly_incm', 'no_dpnd', 'ins_typ_count', 'total_ape_usd', 'tot_face_amt_usd', 'protection_gap', 'protection_income%', 'client_tenure', 'clm_6m_ratio']

# COMMAND ----------

group1 = nonucm_target_activation_df[nonucm_target_activation_df['mar_stat_cat'].isin(['2. Married', '3. Unknown']) &
                                     nonucm_target_activation_df['no_dpnd_cat'].isin(['2','3+'])]

print('# customers in group:', group1.shape[0])

calculate_summary_stats(group1, columns_of_interest)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Mature Couples</strong>

# COMMAND ----------

group2 = nonucm_target_activation_df[nonucm_target_activation_df['mar_stat_cat'].isin(['1. Married w/ kids']) &
                                     nonucm_target_activation_df['no_dpnd_cat'].isin(['1','2','3+'])]

print('# customers in group:', group2.shape[0])

calculate_summary_stats(group2, columns_of_interest)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Younger Couples</strong>

# COMMAND ----------

group3 = nonucm_target_activation_df[nonucm_target_activation_df['mar_stat_cat'].isin(['2. Married', '3. Unknown']) &
                                     nonucm_target_activation_df['no_dpnd_cat'].isin(['1'])]

print('# customers in group:', group3.shape[0])

calculate_summary_stats(group3, columns_of_interest)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Personal Protection Seekers</strong>

# COMMAND ----------

group4 = nonucm_target_activation_df[nonucm_target_activation_df['no_dpnd_cat']=='0']

print('# customers in group:', group4.shape[0])

calculate_summary_stats(group4, columns_of_interest)

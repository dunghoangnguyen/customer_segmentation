# Databricks notebook source
# MAGIC %run /Repos/dung_nguyen_hoang@mfcgd.com/Utilities/Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libl and global params

# COMMAND ----------

# Setting up parameters
from datetime import datetime, timedelta
import calendar
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# Get the last month-end from current system date
#last_mthend = datetime.strftime(datetime.now().replace(day=1) - timedelta(days=1), '%Y-%m-%d')

x = 1 # Change to number of months ago (0: last month-end, 1: last last month-end, ...)
today = datetime.now()
first_day_of_current_month = today.replace(day=1)
current_month = first_day_of_current_month

for i in range(x):
    first_day_of_previous_month = current_month - timedelta(days=1)
    first_day_of_previous_month = first_day_of_previous_month.replace(day=1)
    current_month = first_day_of_previous_month

last_day_of_x_months_ago = current_month - timedelta(days=1)
last_mthend = last_day_of_x_months_ago.strftime('%Y-%m-%d')
last_mthend_sht = last_mthend[0:7]
print("Selected last_mthend = ", last_mthend)
print("Selected last_mthend_sht = ", last_mthend_sht)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load paths and source tables

# COMMAND ----------

lab_path = '/mnt/lab/vn/project/scratch/cseg_cltv/'

tblSrc1 = 'temp3/'

path_list = [lab_path,]
file_list = [tblSrc1,]

# COMMAND ----------

df_list = load_parquet_files(path_list, file_list)

# COMMAND ----------

generate_temp_view(df_list)

# COMMAND ----------

old_cus_df = spark.sql(f"""
select  distinct
        po_num, cus_gender as gender, client_tenure,
        case when client_tenure < 2 then '1. Less than 2 years'
             when client_tenure < 5 then '2. Less than 5 years'
             when client_tenure between 5 and 7 then '3. 5-7 years'
             when client_tenure > 7 and client_tenure <= 10 then '4. 7-10 years'
             when client_tenure > 10 and client_tenure <= 15 then '5. 10-15 years'
             when client_tenure > 15 and client_tenure <= 20 then '6. 15-20 years'
             else '7. More than 20 years'
        end client_tenure_band, 
        year(first_pol_eff_dt) first_pol_yr, pol_count, ins_typ_count,
        inforce_ind, nvl(f_owner_is_agent,0) f_owner_is_agent, agt_tenure_yrs,
        case when agt_tenure_yrs is null then '0. Unassigned' 
             when agt_tenure_yrs < 2 then '1. Less than 2 years'
             when agt_tenure_yrs < 5 then '2. Less than 5 years'
             when agt_tenure_yrs between 5 and 7 then '3. 5-7 years'
             when agt_tenure_yrs > 7 and agt_tenure_yrs <= 10 then '4. 7-10 years'
             when agt_tenure_yrs > 10 and agt_tenure_yrs <= 15 then '5. 10-15 years'
             when agt_tenure_yrs > 15 and agt_tenure_yrs <= 20 then '6. 15-20 years'
             else '7. More than 20 years'
        end agt_tenure_band, 
        valid_email, valid_mobile, --coverage_ape, 
        total_ape, nvl(city, 'Not Declared') city, customer_segment, nvl(unassigned_ind,0) unassigned_ind, 
        cus_age_band, dependent_age_band, existing_vip_seg,
        nvl(dpnd_child_ind,0) dpnd_child_ind, 10yr_pol_cnt
from    temp3
where   1=1
    --and 10yr_pol_cnt>0
    and inforce_ind=1
    and image_date='{last_mthend}'
order by
        client_tenure desc
""")

old_cus_df.groupBy('client_tenure_band')\
    .agg(
        F.countDistinct('po_num').alias('no_customers')
    ).orderBy('client_tenure_band').display()
old_cus_df.createOrReplaceTempView('old_cus')

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select  *
# MAGIC from    old_cus
# MAGIC where   1=1
# MAGIC     --and 10yr_pol_cnt>0
# MAGIC     and client_tenure_band in ('6. 15-20 years', '7. More than 20 years')
# MAGIC     and gender <> 'Unknown'

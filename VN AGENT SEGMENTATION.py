# Databricks notebook source
# MAGIC %md
# MAGIC # Review, Analyize and Predict the Agents' Performances

# COMMAND ----------

# MAGIC %run /Repos/dung_nguyen_hoang@mfcgd.com/Utilities/Functions

# COMMAND ----------

import pyspark.sql.functions as F 
import pyspark.sql.types as T
from pyspark.sql import Window
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1. Modify the config and paths for upcoming month

# COMMAND ----------

# Get the current date
current_date = pd.Timestamp.now()

# Calculate the last month-end
last_month_end = current_date - pd.DateOffset(days=current_date.day)
max_date = last_month_end
max_date_str = last_month_end.strftime('%Y-%m-%d')

# Calculate the last month end
last_month = pd.to_datetime(max_date) - pd.DateOffset(days=max_date.day)
min_last_mth = (last_month + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

# Calculate the month-end of 3 months ago
last_3_months_ago = max_date - pd.DateOffset(months=3)
last_3_months_ago_month_end = last_3_months_ago + pd.offsets.MonthEnd(0)
min_last_3mth = last_3_months_ago_month_end.strftime('%Y-%m-%d')

# Calculate the month-end of exactly 1 year from max_date
min_date = max_date - pd.DateOffset(years=1)
min_date_month_end = min_date + pd.offsets.MonthEnd(0)
min_date_str = min_date_month_end.strftime('%Y-%m-%d')

mth_partition = max_date_str[:7]
current_year = max_date.year
snapshot = mth_partition.replace('-','') # used to filter lapse socre and persistency score

print(f"max_date_str: {max_date_str}")
print(f"min_last_mth: {min_last_mth}")
print(f"min_last_3mth: {min_last_3mth}")
print(f"min_date_str: {min_date_str}")
print(f"snapshot: {snapshot}")

# offline data, manual work for manuprolist and VN model output. please control the format to align with previous month
# list of selected comp_pro
list_comp_pro = ('01', '98')
# manu pro levels
manupro_level = ('S','G','P')
# Convert yyyymmdd_MDRT_list.xlsx into vn_mdrt_yyyymmdd.csv and upload to DBx hive_metastore/default
path_mdrt = 'vn_mdrt_20230920'
# Convert yyyymmdd_Platinum_list.xlsx into vn_mp_yyyymm.csv and upload to DBx hive_metastore/default
path_manupro = 'vn_mp_' + snapshot

# model output tables, change the date and data availability
multiclass_path = '/dbfs/mnt/lab/vn/project/scratch/gen_rep_2023/prod_existing/11_multiclass_scored_base/multiclass_scored_' + snapshot + '.csv'
leads_model_path = '/dbfs/mnt/prod/vn/project/scratch/gen_rep_2023/prod_existing/8_model_score_existing/'
lapse_path = '/dbfs/mnt/lab/vn/project/lapse/pre_lapse_deployment/lapse_mthly/lapse_score.parquet/'

# pre_lapse_revamp_path = '/dbfs/mnt/lab/vn/project/lapse/pre_lapse_revamp_3/snapshots/snapshots_202309/'

# link to Manulife Acandemy manupro
mp_link_url = 'https://manulife-mba.axonify.com/training/index.html#hub/search/community-1535/articles/1'

#2024 MDRT requirment
ape_benchmark = dict({'Silver': 60000.000, 'Gold': 360000.000, 'Platinum': 600000.000, 'MDRT': 721626.600, 'COT': 2164879.800, 'TOT': 4329759.600, '^.^': 4329759.600})

# output files to
out_path = '/dbfs/mnt/lab/vn/project/cpm/datamarts/'
dm_path = '/mnt/prod/Curated/VN/Master/VN_CURATED_DATAMART_DB/'
ams_path = '/mnt/prod/Published/VN/Master/VN_PUBLISHED_CASM_AMS_SNAPSHOT_DB/'
#bak_path = '/mnt/prod/Published/VN/Master/VN_PUBLISHED_AMS_BAK_DB/'
cas_path = '/mnt/prod/Published/VN/Master/VN_PUBLISHED_CASM_CAS_SNAPSHOT_DB/'
alt_path = '/mnt/prod/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/'
cpm_path = '/mnt/prod/Curated/VN/Master/VN_CURATED_CAMPAIGN_DB/'
nbv_path = '/mnt/prod/Published/VN/Master/VN_PUBLISHED_CAMPAIGN_FILEBASED_DB/'
gen_path = '/mnt/prod/Curated/VN/Master/VN_CURATED_CUSTOMER_ANALYTICS_DB/'
mod_path = '/mnt/lab/vn/project/lapse/pre_lapse_deployment/lapse_mthly/'

tbl_src1 = 'TAGTDM_MTHEND/'
#tbl_src2 = 'TAMS_CANDIDATES/'
#tbl_src3 = 'tams_locations/'
#tbl_src4 = 'TAMS_AGENTS/'
tbl_src5 = 'AGENT_RFM/'
tbl_src6 = 'tpolidm_MTHEND/'
tbl_src7 = 'tcoverages/'
tbl_src8 = 'nbv_margin_histories/'
tbl_src9 = 'TCLIENT_DETAILS/'
tbl_src10 = 'vn_plan_code_map/'
tbl_src11 = 'tclaim_details/'
tbl_src12 = 'TPLANS/'
tbl_src13 = 'TAMS_AGT_ACUMS_BK/'
tbl_src14 = 'lapse_score.parquet/'
tbl_src15 = 'EXISTING_CUSTOMER_SCORE/'

path_list = [dm_path, ams_path, alt_path,
             cas_path, cpm_path, #bak_path,
             gen_path, #mod_path, 
             nbv_path]
tbl_list = [tbl_src1, 
            #tbl_src2, tbl_src3, tbl_src4,
            tbl_src5, tbl_src6, tbl_src7,
            tbl_src8, tbl_src9, tbl_src10, 
            tbl_src11, tbl_src12,
            tbl_src13, #tbl_src14, 
            tbl_src15]


# COMMAND ----------

df_list = load_parquet_files(path_list, tbl_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.2 Filter and intermediate data

# COMMAND ----------

# Select data from snapshot month
#df_list['TAGTDM_MTHEND']    = df_list.pop('TAGTDM_MTHEND_backup')
df_list['TAGTDM_MTHEND']    = df_list['TAGTDM_MTHEND'][df_list['TAGTDM_MTHEND']['image_date'] == max_date_str]
df_list['TPOLIDM_MTHEND']   = df_list['TPOLIDM_MTHEND'][df_list['TPOLIDM_MTHEND']['image_date'] == max_date_str]
df_list['TCOVERAGES']       = df_list['TCOVERAGES'][df_list['TCOVERAGES']['image_date'] == max_date_str]
df_list['TCLIENT_DETAILS']  = df_list['TCLIENT_DETAILS'][df_list['TCLIENT_DETAILS']['image_date'] == max_date_str]
df_list['TCLAIM_DETAILS']   = df_list['TCLAIM_DETAILS'][df_list['TCLAIM_DETAILS']['image_date'] == max_date_str]
df_list['TPLANS']           = df_list['TPLANS'][df_list['TPLANS']['image_date'] == max_date_str]
agent_rfm = df_list['AGENT_RFM'][df_list['AGENT_RFM']['monthend_dt'] == mth_partition]

# rename model_base_existing and lapse_score.parquet in df_list
df_list['leads_existing_model'] = df_list.pop('EXISTING_CUSTOMER_SCORE')
#df_list['lapse'] = df_list.pop('lapse_score.parquet')

tclient_details = df_list['TCLIENT_DETAILS'][['ID_NUM', 'CLI_NUM', 'BIRTH_DT', 'SEX_CODE', 'CLI_NM']].filter(F.to_date(F.col('BIRTH_DT')) < '2200-01-01')
df_list['NBV_MARGIN'] = df_list.pop('NBV_MARGIN_HISTORIES')
df_list['PLAN_CODE_MAP'] = df_list.pop('VN_PLAN_CODE_MAP')

# get list of single premium products
sp_plan = df_list['TPLANS'].filter(F.col('SNGL_PREM_IND') == 'Y').select('PLAN_CODE').rdd.flatMap(lambda x: x).collect()
#print(sp_plan)

# Generate temp views for data extraction and merging
generate_temp_view(df_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2. Load dependency model outputs

# COMMAND ----------

multiclass = pd.read_csv(multiclass_path) # new file for csv every month
lapse = spark.read.parquet('/mnt/lab/vn/project/lapse/pre_lapse_deployment/lapse_mthly/lapse_score.parquet/') \
        .filter(F.col('month_snapshot') == snapshot).toPandas()
leads_existing_model = df_list['leads_existing_model'].filter(F.col('image_date') == max_date_str).toPandas()
print("No of leads from HP model: ", leads_existing_model.shape[0])
print("No of policies from lapse model: ", lapse.shape[0])
#leads_existing_model.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3. Load PAR related data

# COMMAND ----------

sql_string = '''
select * 
from {0} 
;
'''.format(path_mdrt)
mdrt = sql_to_df(sql_string, 1, spark)
print("No MDRT agents: ", mdrt.count())
mdrt.createOrReplaceTempView('mdrt')

sql_string = '''
with sbw_lst as (
    select  agt_code, sum(case when sbw01 is not null then 1 else 0 end +
                      case when sbw02 is not null then 1 else 0 end  +
                      case when sbw03 is not null then 1 else 0 end  +
                      case when sbw04 is not null then 1 else 0 end  +
                      case when sbw05 is not null then 1 else 0 end  +
                      case when sbw06 is not null then 1 else 0 end  +
                      case when sbw07 is not null then 1 else 0 end  +
                      case when sbw08 is not null then 1 else 0 end  +
                      case when sbw09 is not null then 1 else 0 end  +
                      case when sbw10 is not null then 1 else 0 end  +
                      case when sbw11 is not null then 1 else 0 end  +
                      case when sbw12 is not null then 1 else 0 end) as no_sbw_completed
    from    hive_metastore.vn_curated_reports_db.cpd_sbw_data
    where   compro in ('01','98')
        and stat_cd = '01'
    group by agt_code
    having no_sbw_completed > 0
)
select distinct
        agt.agt_code, agt_nm, agt_join_dt, agt_stat_code, br_code, can_num, chnl_cd, cntrct_eff_dt, comp_prvd_num, fc_ind, iqa_ind, mba_ind, mdrt_ind, pend_cd, rank_code, recrut_by, stat_cd, agt_stat, team_code, trmn_dt, trmn_reasn, unit_code, agt_term_dt, birth_dt, email_addr, id_num, sex_code as sex_code_agt, loc_cd, loc_desc, channel, edu_info, highest_edu, aprv_dt_by_mof, prd_cls_typ, mgr_cd, stru_grp_cd, cmp_ind, epos_ind, agent_group, agent_group_desc, dtk_ind, case mpro_title
            when 'S' then 'Silver'
            when 'G' then 'Gold'
            when 'P' then 'Platinum'
        end as mpro_title,
        mpro_title_eff_dt, mpro_title_pending_eff_dt, mpro_termination_dt, mar_stat_cd, ins_exp_ind,
        nvl(mdrt.` Ranking `,
            nvl(case mpro_title
                    when 'S' then 'Silver'
                    when 'G' then 'Gold'
                    when 'P' then 'Platinum' end,
                'Unranked')
            ) as tier,
        to_date('{1}') as max_date,
        cast(months_between(to_date('{1}'), agt_join_dt) as int) as tenure_mth,
        nvl(sbw_lst.no_sbw_completed,0) as no_sbw_completed
from    tagtdm_mthend agt left join
        (select distinct agt_code, mar_stat_cd, ins_exp_ind
         from   agent_rfm) 
        rfm     on agt.agt_code=rfm.agt_code left join
        mdrt    on agt.agt_code=mdrt.`Agt Code` left join
        sbw_lst on agt.agt_code=sbw_lst.agt_code 
where   1=1
    and channel='Agency'
    and agt_stat_code='1'
    and agt_term_dt is null
    and comp_prvd_num in {0};
'''.format(list_comp_pro, max_date_str)
agt_all = sql_to_df(sql_string, 1, spark)
print("No agents selected: ",agt_all.count())
#print("No of null agt_join_dt: ", agt_all.filter(F.col('agt_join_dt').isNull()).count())

#tmp_pd = tagt.groupby(['tier'])['agt_code'].nunique().fillna(0)
#tmp_pd
#del tmp_pd

agt_all.createOrReplaceTempView('agt_all')

sql_string = '''
with cli_agt_id as (
    select  distinct
            cli_num, cli.id_num as cli_id_num_agt_buy
    from    tclient_details cli inner join
            agt_all agt on cli.id_num=agt.id_num
), cli_dtl as (
    select  distinct
            cli_num, 
            sex_code,
            year('{0}') - year(birth_dt) as cli_age_this_year
    from    tclient_details
), cli_frst_pol as (
    select  po_num, frst_pol_num
    from    (select  po_num, pol_num as frst_pol_num, 
                    row_number() over (partition by po_num order by pol_eff_dt) rn
            from    tpolidm_mthend
            where   pol_stat_cd not in ('A','N','R')) t
    where   rn=1
)
select  cvg.pol_num, cvg.cli_num, cvg_eff_dt, cvg.plan_code, cast(cvg_prem as int) as cvg_prem, cvg.cvg_typ, cvg_reasn, occp_clas, par_code, 
        bnft_dur, prem_dur, cvg.rel_to_insrd, 
        smkr_code, cvg.sex_code, cvg_eff_age, cvg_stat_cd, cvg_del_dt, face_amt, cli_id_num_agt_buy,
        pol_app_dt, pol_eff_dt, sbmt_dt, pol_trmn_dt, 
        case cast(pmt_mode as int) 
             when 1 then 'monthly'
             when 3 then 'quarterly'
             when 6 then 'half-yearly'
             when 12 then 'yearly'
        end as pmt_mode, 
        sa_code as agt_code, wa_code as wa_cd_1, 'None' as wa_cd_2, pol.plan_code as plan_code_base, 'None' as plan_prem, dist_chnl_cd, pol_stat_cd, pd_to_dt, datediff(pol_trmn_dt, pol_eff_dt) as term_pol_last_days, agt.id_num as agt_id_num_agt_buy, 
        case when cvg_stat_cd in ('A','N','R') then 0 
             else cvg_prem*12/cast(pmt_mode as int)
        end as APE, 
        agt_nm, agt_join_dt, agt_stat_code, br_code, can_num, chnl_cd, cntrct_eff_dt, comp_prvd_num, fc_ind, iqa_ind, mba_ind, mdrt_ind, agt.pend_cd, rank_code, recrut_by, stat_cd, agt_stat, team_code, trmn_dt, trmn_reasn, unit_code, agt_term_dt, birth_dt, email_addr, id_num, sex_code_agt, loc_cd, loc_desc, channel, edu_info, highest_edu, aprv_dt_by_mof, prd_cls_typ, mgr_cd, stru_grp_cd, cmp_ind, epos_ind, agent_group, agent_group_desc, dtk_ind, mpro_title,
        mpro_title_eff_dt, mpro_title_pending_eff_dt, mpro_termination_dt, mar_stat_cd, ins_exp_ind,
        pol.po_num,
        po.cli_age_this_year as po_age_this_year,
        ins.cli_age_this_year as in_age_this_year,
        po.sex_code as po_gender,
        ins.sex_code as in_gender,
        year(cvg.cvg_eff_dt) as cvg_eff_yr,
        substr(to_date(cvg.cvg_eff_dt),1,7) as cvg_eff_yr_mth,
        cast(months_between(cvg.cvg_eff_dt, pol.pol_eff_dt) + 1 as int) as cvg_eff_mth_from_pol_eff,
        cast(months_between(cvg.cvg_eff_dt, agt.agt_join_dt) + 1 as int) as cvg_eff_mth_from_agt_join,
        cast(datediff(cvg.cvg_eff_dt,agt.agt_join_dt)/7 as int) + 1 as cvg_eff_week_from_agt_join,
        cast(months_between(pol.pol_eff_dt, agt.agt_join_dt) + 1 as int) as pol_eff_mth_from_agt_join,
        datediff(pol.pol_eff_dt, agt.agt_join_dt) as pol_eff_days_from_agt_join,
        pol.pol_stat_desc as status,
        case when cvg.cvg_stat_cd in ('1','2','3','5','7') then 1 else 0 end as cvg_active,
        case when pol.pol_stat_cd in ('1','2','3','5','7') then 1 else 0 end as pol_active,
        case when pol.pol_stat_cd in ('A','N') then 1 else 0 end as is_non_taken,
        case when pol.pol_stat_cd = 'B' then 1 else 0 end as lapsed,
        case when pol.pol_stat_cd = 'R' then 1 else 0 end as rejected,
        year(pol.pol_eff_dt) as pol_eff_yr,        
        agt.tier,
        substr(to_date(cvg.cvg_eff_dt),1,7) as year_month,
        least(pol.pol_trmn_dt, cvg.cvg_del_dt) as min_trmn_dt,
        datediff(least(pol.pol_trmn_dt, cvg.cvg_del_dt), pol.pol_eff_dt) as pol_eff_days,
        case when fpl.frst_pol_num is not null then 1 else 0 end as if_first_pol,
        case when pol.pol_stat_cd not in ('A','N','R') and fpl.frst_pol_num is null then 1 else 0 end as repeated_sales,
        nvl(nbv.customer_needs,pln.customer_needs) as type,
        cast(nvl(nbv.nbv_margin_agency,pln.nbv_margin_agency) as float) as nbv_margin,
        to_date('{0}') as image_date
from    tcoverages cvg inner join
        tpolidm_mthend   pol on cvg.pol_num   = pol.pol_num  inner join
        agt_all         agt on pol.wa_code   = agt.agt_code left join
        cli_agt_id      cli on pol.po_num    = cli.cli_num  left join
        cli_dtl         po  on pol.po_num    = po.cli_num   left join
        cli_dtl         ins on pol.insrd_num = ins.cli_num  left join
        cli_frst_pol    fpl on pol.pol_num   = fpl.frst_pol_num left join
        nbv_margin      nbv on nbv.plan_code = cvg.plan_code and floor(months_between(cvg.cvg_eff_dt,nbv.effective_date)) between 0 and 2 left join
        plan_code_map   pln on pln.plan_code = cvg.plan_code
where   agt.agt_code is not null
    or  cvg_eff_dt > '{1}';
'''.format(max_date_str, min_last_mth)
master_df = sql_to_df(sql_string, 1, spark)
print("No coverages selected: ", master_df.count())
#print("No null 'agt_join_dt': ", master_df.filter(F.col('agt_join_dt').isNull()).count())

# Convert all DecimalType columns to FloatType
decimal_columns = [col.name for col in master_df.schema.fields if isinstance(col.dataType, T.DecimalType)]

# Loop over columns and cast to FloatType
for col in decimal_columns:
    master_df = master_df.withColumn(col, master_df[col].cast(T.FloatType()))

sql_string = '''
select distinct 
  tclaim.clm_id
, to_date(tclaim.event_dt) event_dt
, to_date(tclaim.clm_recv_dt) clm_recv_dt
, to_date(tclaim.clm_aprov_dt) clm_aprov_dt
, tclaim.clm_stat_code
, tclaim.clm_reasn_cd
, tclaim.plan_code
, tclaim.clm_code
, tclaim.clm_aprov_amt
, case 
    when tclaim.clm_stat_code = 'A' then tclaim.clm_aprov_amt 
    else 0 
  end as adj_aprov_amt 
, tclaim.clm_prvd_amt
, tclaim.pol_num
, tclaim.cli_num
from    tclaim_details as tclaim inner join
        tpolidm_mthend as pol on tclaim.pol_num=pol.pol_num inner join
        agt_all as agt on pol.wa_code=agt.agt_code
where   to_date(clm_recv_dt) < '{0}'
    and to_date(event_dt) < '{0}'
    and to_date(clm_aprov_dt) < '{0}';
'''.format(snapshot)
tclaim_df = sql_to_df(sql_string, 1, spark)
print("No claims selected: ", tclaim_df.count())
# Convert all DecimalType columns to FloatType
decimal_columns = [col.name for col in tclaim_df.schema.fields if isinstance(col.dataType, T.DecimalType)]

# Loop over columns and cast to FloatType
for col in decimal_columns:
    tclaim_df = tclaim_df.withColumn(col, tclaim_df[col].cast(T.FloatType()))

query = '''
select agt_cd as wa_cd_1, cast(acum_bal as float) as m19_per
from    hive_metastore.vn_published_ams_bak_db.TAMS_AGT_ACUMS_BK
where agt_cd in (
    select agt_code from agt_all
) and run_num = '{0}'
and acum_cd = 'A214'
;
'''.format(snapshot)
m19_per_vn = spark.sql(query)
print("No agents with 19m per: ", m19_per_vn.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4. Intermediate data

# COMMAND ----------

master_df = master_df.join(m19_per_vn, on='wa_cd_1', how='left')
# write the master Spark dataframe to csv/parquet
master_df.write.mode('overwrite').parquet(f'/mnt/lab/vn/project/scratch/master_df')
# Reload the master dataset
master_df = spark.read.parquet('/mnt/lab/vn/project/scratch/master_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5. Process for policy tables

# COMMAND ----------

windowSpec = Window.partitionBy("cli_num").orderBy("cvg_eff_dt").rowsBetween(Window.unboundedPreceding, Window.currentRow)

master_df = master_df.withColumn("frst_pol_dt", F.min("cvg_eff_dt").over(windowSpec))
#master_df = master_df.withColumn("repeated_sales",
#                                 F.when((F.col("cvg_eff_dt") > F.col("frst_pol_dt")) &
#                                        (F.col("pol_eff_days") > 21), 1).otherwise(0))

# COMMAND ----------

def adjust_ape(eff_days, pmt, prem, ape):
    if eff_days is None:
        return ape
    elif eff_days <= 21 or eff_days >= 365:
        return 0 if eff_days <= 21 else ape
    elif eff_days < 365 and pmt == 12:
        return ape
    else:
        if pmt == 3:
            return prem*(F.floor(eff_days/92)+1)
        elif pmt == 6:
            return prem*(F.floor(eff_days/183)+1)
        elif pmt == 1:
            return prem*(F.floor(eff_days/31)+1)
        else:
            return 0

master_df = master_df.withColumn('adjust_ape', F.struct('pol_eff_days', 'pmt_mode', 'cvg_prem', 'APE'))
master_df = master_df.withColumn('adjust_ape', F.udf(adjust_ape, T.FloatType())('pol_eff_days', 'pmt_mode', 'cvg_prem', 'APE'))

# factor in single premium to recalculate adjusted ape
master_df = master_df.withColumn("sp_ind", F.when(F.col("plan_code").isin(sp_plan), 1).otherwise(0))
master_df = master_df.withColumn("adjust_ape",
                F.when((F.col("sp_ind") == 0), F.col("adjust_ape"))
                 .otherwise(F.when((F.col("pol_eff_days").isNull()) | (F.col("pol_eff_days") <= 21), 0)
                             .otherwise(F.col("cvg_prem")*0.1)).cast('float'))

master_df = master_df.fillna(value=0, subset=["adjust_ape"])
master_df.filter(F.col('wa_cd_1').isin(['10193','10539'])).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6. Add related columns to df for further calculation

# COMMAND ----------

#spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true") 
#df = master_df.toPandas()
#nbv = margin.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7. Add estimation of family relationship and nbv margin

# COMMAND ----------

# write the master Spark dataframe to csv/parquet
master_df.write.mode('overwrite').parquet(f'/mnt/lab/vn/project/scratch/master_df')
# Reload the master dataset
master_df = spark.read.parquet('/mnt/lab/vn/project/scratch/master_df')

# COMMAND ----------

# convert beneficiary code to string and map to text values
rel_dict = {2: 'Spouse', 1: 'Self', 0: 'Others', 52: 'Others', 51: 'Parent', 
            3: 'Child', 5: 'Others', 4: 'Others', 10: 'Others', 31: 'Others'}

# UDF to map beneficiary code to text
rel_udf = udf(lambda x: rel_dict.get(int(x), None), T.StringType())

# convert columns and add new columns
master_df = (master_df
             .withColumn('insrd', rel_udf(F.col('rel_to_insrd').cast(T.IntegerType())))
             .withColumn('be_is_cli', (F.col('insrd') == 'Self').cast(T.IntegerType()))
             .withColumn('po_is_cli', (F.col('po_num') == F.col('cli_num')).cast(T.IntegerType()))
             .withColumn('po_cli_age_gap',
                        F.when(F.col('po_is_cli') == 1, None).otherwise(F.col('po_age_this_year') - F.col('in_age_this_year')).cast(T.IntegerType()))
             )

# define UDF to determine relationship based on the age of the policyholder
@udf(T.StringType())
def which_rela(a, b, c, d):
    if a == 1 or b is None:
        y = 'self'
    else:
        if 18 <= b <= 40:
            y = 'child'
        else:
            if b > 40:
                y = 'others'
            else:
                if b < -17:
                    y = 'parent'
                else:
                    if c != d:
                        y = 'spouse'
                    else:
                        y = 'others'
    return y

# add new column for policyholder-insured relationship
master_df = master_df.withColumn('po_ins_rela', 
                                 which_rela(F.col('po_is_cli'), F.col('po_cli_age_gap'), 
                                            F.col('po_gender'), F.col('in_gender')))

# add new columns for product type, nbv margin and nbv
master_df = (master_df
             .withColumn('nbv', F.col('adjust_ape') * F.col('nbv_margin'))
             .withColumn('prd_cus_mix', F.concat_ws('_',F.col('type'),F.col('po_ins_rela')))
            )

# convert date columns to periods as temp columns
master_df = master_df.withColumn('max_date', F.lit(max_date))
master_df = master_df.withColumn('cvg_eff_month', F.date_format(F.col('cvg_eff_dt'), 'yyyy-MM'))
master_df = master_df.withColumn('max_month', F.date_format(F.col('max_date'), 'yyyy-MM'))

# calculate the difference in months
master_df = master_df.withColumn('last_n_month_eff_cvg', 
                   (F.months_between(F.col('cvg_eff_month'), F.col('max_month')).cast('int') - 1))
# drop temporary columns
master_df = master_df.drop('cvg_eff_month', 'max_month')

master_df = master_df.filter((F.col('max_date') >= F.col('cvg_eff_dt')) & (F.col('max_date') >= F.col('pol_eff_dt')))\
             .orderBy('pol_num')

master_df.filter(F.col('wa_cd_1').isin(['10193','10539'])).display()

# COMMAND ----------

# Store df to parquet
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
#master_df.write.mode('overwrite').parquet(f'/mnt/lab/vn/project/scratch/master_df')
master_df.write.mode('overwrite').partitionBy('image_date').parquet(f"/dbfs/mnt/lab/vn/project/cpm/datamarts/TAGTPAR_MTHEND")
master_df = spark.read.parquet('/dbfs/mnt/lab/vn/project/cpm/datamarts/TAGTPAR_MTHEND').filter(F.col('image_date') == max_date_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8. Prepare for other tables

# COMMAND ----------

sub = master_df.select('pol_num', 'wa_cd_1').dropDuplicates()
pol_agt = sub.rdd.collectAsMap()

sub = master_df.select('cli_num', 'wa_cd_1').dropDuplicates()
cli_agt = sub.rdd.collectAsMap()

sub = master_df.select('po_num', 'wa_cd_1').dropDuplicates()
po_agt = sub.rdd.collectAsMap()

sub = master_df.select('pol_num', 'po_num').dropDuplicates()
po_pol_dict = sub.rdd.collectAsMap()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9. Prepare agt table

# COMMAND ----------

# DBTITLE 1,policies
agt = agt_all.select('agt_code','agt_nm','rank_code', 'br_code', 'tenure_mth', 'agt_join_dt', 'tier', 'no_sbw_completed').toPandas()
#agt = tagt[['agt_code','agt_nm','rank_code', 'br_code', 'tenure_mth', 'agt_join_dt', 'tier']]
agt = agt.rename(columns={'agt_code': 'agt_cd'})
agt_list = agt_all.select('agt_code').dropDuplicates().collect()
#print(agt_list)
agt[agt['tier'] != 'Unranked']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10. Prepare aggregate level KPIs

# COMMAND ----------

# APE
f1 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.sum(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('adjust_ape'))).alias('last_yr_ape'),
         F.sum(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('adjust_ape'))).alias('last_mth_ape'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('adjust_ape'))).alias('last_12m_ape'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('adjust_ape'))).alias('last_3m_ape'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('adjust_ape'))).alias('last_6m_ape'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('adjust_ape'))).alias('last_9m_ape'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('adjust_ape'))).alias('ape_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('adjust_ape'))).alias('total_ape')
         )\
    .fillna(0)
    #.toPandas())

# NBV
f2 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.sum(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('nbv'))).alias('last_yr_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('nbv'))).alias('last_mth_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('nbv'))).alias('last_12m_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('nbv'))).alias('last_3m_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('nbv'))).alias('last_6m_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('nbv'))).alias('last_9m_nbv'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('nbv'))).alias('nbv_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('nbv'))).alias('total_nbv')
         )\
    .fillna(0)
    #.toPandas())

# PO
f3 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('po_num'))).alias('last_yr_po'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('po_num'))).alias('last_mth_po'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('po_num'))).alias('last_12m_po'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('po_num'))).alias('last_3m_po'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('po_num'))).alias('last_6m_po'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('po_num'))).alias('last_9m_po'),
         F.countDistinct(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('po_num'))).alias('po_ytd'),
         F.countDistinct(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('po_num'))).alias('total_po')
         )\
    .fillna(0)
    #.toPandas())

# Customer
f4 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('cli_num'))).alias('last_yr_cus'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('cli_num'))).alias('last_mth_cus'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('cli_num'))).alias('last_12m_cus'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('cli_num'))).alias('last_3m_cus'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('cli_num'))).alias('last_6m_cus'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('cli_num'))).alias('last_9m_cus'),
         F.countDistinct(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('cli_num'))).alias('cus_ytd'),
         F.countDistinct(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('cli_num'))).alias('total_cus')
         )\
    .fillna(0)
    #.toPandas())

# Policy
f5 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('pol_num'))).alias('last_yr_pol'),
         F.countDistinct(F.col('pol_num')).alias('all_pol_cnt'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('pol_num'))).alias('last_mth_pol'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('pol_num'))).alias('last_12m_pol'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('pol_num'))).alias('last_3m_pol'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('pol_num'))).alias('last_6m_pol'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('pol_num'))).alias('last_9m_pol'),
         F.countDistinct(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('pol_num'))).alias('pol_ytd'),
         F.countDistinct(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('pol_num'))).alias('total_pol')
         )\
    .fillna(0)
    #.toPandas())

# Not-taken pol
f6 = master_df.filter(F.col('is_non_taken') == 1)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('pol_num'))).alias('last_mth_NT'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('pol_num'))).alias('last_12m_NT'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('pol_num'))).alias('last_3m_NT'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('pol_num'))).alias('last_6m_NT'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('pol_num'))).alias('last_9m_NT'))\
    .fillna(0)
    #.toPandas())

# Rejected pol
f7 = master_df.filter(F.col('rejected') == 1)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('pol_num'))).alias('last_mth_rej'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('pol_num'))).alias('last_12m_rej'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('pol_num'))).alias('last_3m_rej'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('pol_num'))).alias('last_6m_rej'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('pol_num'))).alias('last_9m_rej'))\
    .fillna(0)
    #.toPandas())

# Lapsed pol
f8 = master_df.filter(F.col('lapsed') == 1)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('pol_num'))).alias('last_mth_lap'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('pol_num'))).alias('last_12m_lap'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('pol_num'))).alias('last_3m_lap'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('pol_num'))).alias('last_6m_lap'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('pol_num'))).alias('last_9m_lap'),
         F.countDistinct(F.when(F.months_between(F.col('pol_trmn_dt'),F.col('pol_eff_dt')) <= 14, F.col('pol_num'))).alias('lap_14m_pol'),
         F.countDistinct(F.when(F.months_between(F.col('pol_trmn_dt'),F.col('pol_eff_dt')) <= 20, F.col('pol_num'))).alias('lap_20m_pol'),
         F.countDistinct(F.when(F.months_between(F.col('pol_trmn_dt'),F.col('pol_eff_dt')) <= 26, F.col('pol_num'))).alias('lap_26m_pol')
         )\
    .fillna(0)
    #.toPandas())

# Product type
f9 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('type'))).alias('last_yr_prd'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('type'))).alias('last_mth_prd'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('type'))).alias('last_12m_prd'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('type'))).alias('last_3m_prd'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('type'))).alias('last_6m_prd'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('type'))).alias('last_9m_prd'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('type'))).alias('prd_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('type'))).alias('total_prd')
         )\
    .fillna(0)
    #.toPandas())

# Family member
f10 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R'])) == False)\
    .groupby('wa_cd_1')\
    .agg(F.countDistinct(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('po_ins_rela'))).alias('last_mth_fam'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('po_ins_rela'))).alias('last_12m_fam'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('po_ins_rela'))).alias('last_3m_fam'), 
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('po_ins_rela'))).alias('last_6m_fam'),
         F.countDistinct(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('po_ins_rela'))).alias('last_9m_fam'))\
    .fillna(0)
    #.toPandas())

# Rider APE
f11 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R']) == False) &
                      (F.col('cvg_typ') == 'R'))\
    .groupby('wa_cd_1')\
    .agg(F.sum(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('adjust_ape'))).alias('last_yr_rid'),
         F.sum(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('adjust_ape'))).alias('last_mth_rid'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('adjust_ape'))).alias('last_12m_rid'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('adjust_ape'))).alias('last_3m_rid'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('adjust_ape'))).alias('last_6m_rid'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('adjust_ape'))).alias('last_9m_rid'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('adjust_ape'))).alias('rid_ape_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('adjust_ape'))).alias('total_rid_ape')
         )\
    .fillna(0)
    #.toPandas())

# Repeated sales
f12 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R']) == False) &
                      (F.col('repeated_sales') == 1))\
    .groupby('wa_cd_1')\
    .agg(F.sum(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('adjust_ape'))).alias('last_yr_rep'),
         F.sum(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('adjust_ape'))).alias('last_mth_rep'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('adjust_ape'))).alias('last_12m_rep'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('adjust_ape'))).alias('last_3m_rep'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('adjust_ape'))).alias('last_6m_rep'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('adjust_ape'))).alias('last_9m_rep'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('adjust_ape'))).alias('rep_ape_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('adjust_ape'))).alias('total_rep_ape')
         )\
    .fillna(0)
    #.toPandas())

# Repeated NBV
f13 = master_df.filter((F.col('pol_stat_cd').isin(['A','N','R']) == False) &
                      (F.col('repeated_sales') == 1))\
    .groupby('wa_cd_1')\
    .agg(F.sum(F.when(F.year(F.col('cvg_eff_dt'))==current_year-1, F.col('nbv'))).alias('last_yr_rep_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg') == -1), F.col('nbv'))).alias('last_mth_rep_nbv'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-12, -1)), F.col('nbv'))).alias('last_12m_rep_nbv'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-3, -1)), F.col('nbv'))).alias('last_3m_rep_nbv'), 
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-6, -1)), F.col('nbv'))).alias('last_6m_rep_nbv'),
         F.sum(F.when((F.col('last_n_month_eff_cvg').between(-9, -1)), F.col('nbv'))).alias('last_9m_rep_nbv'),
         F.sum(F.when((F.year(F.col('cvg_eff_dt'))==current_year) & 
                      (F.col('last_n_month_eff_cvg') <= -1), F.col('nbv'))).alias('rep_nbv_ytd'),
         F.sum(F.when(F.col('last_n_month_eff_cvg') <= -1, F.col('nbv'))).alias('total_rep_nbv')
         )\
    .fillna(0)
    #.toPandas())

f = f1.join(f2, on='wa_cd_1', how='left')\
    .join(f3, on='wa_cd_1', how='left')\
    .join(f4, on='wa_cd_1', how='left')\
    .join(f5, on='wa_cd_1', how='left')\
    .join(f6, on='wa_cd_1', how='left')\
    .join(f7, on='wa_cd_1', how='left')\
    .join(f8, on='wa_cd_1', how='left')\
    .join(f9, on='wa_cd_1', how='left')\
    .join(f10, on='wa_cd_1', how='left')\
    .join(f11, on='wa_cd_1', how='left')\
    .join(f12, on='wa_cd_1', how='left')\
    .join(f13, on='wa_cd_1', how='left')

# Avg ticket size
# Avg po size
f = f.withColumn('last_yr_pol_size', F.when(F.col('last_yr_pol') != 0, F.col('last_yr_ape')/F.col('last_yr_pol')).otherwise(1))\
    .withColumn('last_mth_pol_size', F.when(F.col('last_mth_pol') != 0, F.col('last_mth_ape')/F.col('last_mth_pol')).otherwise(1))\
    .withColumn('last_12m_pol_size', F.when(F.col('last_12m_pol') != 0, F.col('last_12m_ape')/F.col('last_12m_pol')).otherwise(1))\
    .withColumn('last_3m_pol_size', F.when(F.col('last_3m_pol') != 0, F.col('last_3m_ape')/F.col('last_3m_pol')).otherwise(1))\
    .withColumn('last_6m_pol_size', F.when(F.col('last_6m_pol') != 0, F.col('last_6m_ape')/F.col('last_6m_pol')).otherwise(1))\
    .withColumn('last_9m_pol_size', F.when(F.col('last_9m_pol') != 0, F.col('last_9m_ape')/F.col('last_9m_pol')).otherwise(1))\
    .withColumn('last_yr_po_size', F.when(F.col('last_yr_po') != 0, F.col('last_yr_ape')/F.col('last_yr_po')).otherwise(1))\
    .withColumn('last_mth_po_size', F.when(F.col('last_mth_po') != 0, F.col('last_mth_ape')/F.col('last_mth_po')).otherwise(1))\
    .withColumn('last_12m_po_size', F.when(F.col('last_12m_po') != 0, F.col('last_12m_ape')/F.col('last_12m_po')).otherwise(1))\
    .withColumn('last_3m_po_size', F.when(F.col('last_3m_po') != 0, F.col('last_3m_ape')/F.col('last_3m_po')).otherwise(1))\
    .withColumn('last_6m_po_size', F.when(F.col('last_6m_po') != 0, F.col('last_6m_ape')/F.col('last_6m_po')).otherwise(1))\
    .withColumn('last_9m_po_size', F.when(F.col('last_9m_po') != 0, F.col('last_9m_ape')/F.col('last_9m_po')).otherwise(1))\
    .fillna(1)

# Not-taken percentage
# Rejected percentage
# Lapsed percentage
f = f.withColumn('last_mth_NT_per', F.when(F.col('last_mth_pol') != 0, 1-(F.col('last_mth_NT')/F.col('last_mth_pol'))).otherwise(1))\
    .withColumn('last_12m_NT_per', F.when(F.col('last_12m_pol') != 0, 1-(F.col('last_12m_NT')/F.col('last_12m_pol'))).otherwise(1))\
    .withColumn('last_3m_NT_per', F.when(F.col('last_3m_pol') != 0, 1-(F.col('last_3m_NT')/F.col('last_3m_pol'))).otherwise(1))\
    .withColumn('last_6m_NT_per', F.when(F.col('last_6m_pol') != 0, 1-(F.col('last_6m_NT')/F.col('last_6m_pol'))).otherwise(1))\
    .withColumn('last_9m_NT_per', F.when(F.col('last_9m_pol') != 0, 1-(F.col('last_9m_NT')/F.col('last_9m_pol'))).otherwise(1))\
    .withColumn('last_mth_rej_per', F.when(F.col('last_mth_pol') != 0, 1-(F.col('last_mth_rej')/F.col('last_mth_pol'))).otherwise(1))\
    .withColumn('last_12m_rej_per', F.when(F.col('last_12m_pol') != 0, 1-(F.col('last_12m_rej')/F.col('last_12m_pol'))).otherwise(1))\
    .withColumn('last_3m_rej_per', F.when(F.col('last_3m_pol') != 0, 1-(F.col('last_3m_rej')/F.col('last_3m_pol'))).otherwise(1))\
    .withColumn('last_6m_rej_per', F.when(F.col('last_6m_pol') != 0, 1-(F.col('last_6m_rej')/F.col('last_6m_pol'))).otherwise(1))\
    .withColumn('last_9m_rej_per', F.when(F.col('last_9m_pol') != 0, 1-(F.col('last_9m_rej')/F.col('last_9m_pol'))).otherwise(1))\
    .withColumn('last_mth_lap_per', F.when(F.col('last_mth_pol') != 0, 1-(F.col('last_mth_lap')/F.col('last_mth_pol'))).otherwise(1))\
    .withColumn('last_12m_lap_per', F.when(F.col('last_12m_pol') != 0, 1-(F.col('last_12m_lap')/F.col('last_12m_pol'))).otherwise(1))\
    .withColumn('last_3m_lap_per', F.when(F.col('last_3m_pol') != 0, 1-(F.col('last_3m_lap')/F.col('last_3m_pol'))).otherwise(1))\
    .withColumn('last_6m_lap_per', F.when(F.col('last_6m_pol') != 0, 1-(F.col('last_6m_lap')/F.col('last_6m_pol'))).otherwise(1))\
    .withColumn('last_9m_lap_per', F.when(F.col('last_9m_pol') != 0, 1-(F.col('last_9m_lap')/F.col('last_9m_pol'))).otherwise(1))\
    .withColumn('14m_per', F.when(F.col('lap_14m_pol') != 0, 1-(F.col('lap_14m_pol')/F.col('all_pol_cnt'))).otherwise(1))\
    .withColumn('20m_per', F.when(F.col('lap_20m_pol') != 0, 1-(F.col('lap_20m_pol')/F.col('all_pol_cnt'))).otherwise(1))\
    .withColumn('26m_per', F.when(F.col('lap_26m_pol') != 0, 1-(F.col('lap_26m_pol')/F.col('all_pol_cnt'))).otherwise(1))\
    .withColumn('12M_NT_rate', F.when(F.col('last_12m_NT') != 0, 1-(F.col('last_12m_NT')/F.col('last_12m_pol'))).otherwise(1))\
    .fillna(1)

f.filter(F.col('wa_cd_1').isin(['10193','10539'])).display()

# COMMAND ----------

tclaim = tclaim_df.toPandas()
final = f.toPandas()

final = final.rename(columns = {'wa_cd_1':'agt_cd'})
final = agt.merge(final, on='agt_cd', how='inner').reset_index()

#final.head(10)
final.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11. Validate and double check

# COMMAND ----------

f1_chk = f1.toPandas().drop('wa_cd_1', axis='columns')
f2_chk = f2.toPandas().drop('wa_cd_1', axis='columns')
f3_chk = f3.toPandas().drop('wa_cd_1', axis='columns')

f1_chk = print(f1_chk.sum())
f2_chk = print(f2_chk.sum())
f3_chk = print(f3_chk.sum())

del f1_chk
del f2_chk
del f3_chk

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 12. Generate KPIs for final table

# COMMAND ----------

avg_sales = master_df.filter((F.col('last_n_month_eff_cvg').between(-12, -1)) & 
                             (~F.col('pol_stat_cd').isin(['A', 'N', 'R'])))\
                            .groupBy(['wa_cd_1', 'year_month']).agg(F.sum('adjust_ape').alias('ape'))\
                            .groupBy(['year_month']).agg(F.mean('ape').cast('integer').alias('avg_ape_tier'))\
                            .withColumnRenamed('year_month', 'date')
avg_sales.display()

mth_sales = master_df.filter((F.col('last_n_month_eff_cvg').between(-12, -1)) & 
                             (~F.col('pol_stat_cd').isin(['A', 'N', 'R'])))\
                            .groupBy(['wa_cd_1', 'year_month']).agg(F.sum('adjust_ape').alias('avg_ape_agt'))\
                            .withColumnRenamed('wa_cd_1', 'agt_cd')\
                            .withColumnRenamed('year_month', 'date')
#mth_sales.display()

avg_prd = master_df.filter((F.col('last_n_month_eff_cvg').between(-12, -1)) & 
                           (~F.col('pol_stat_cd').isin(['A', 'N', 'R'])))\
                          .groupBy(['type']).agg(F.sum('adjust_ape').cast('integer').alias('sum_ape_tier'))\
                          .withColumnRenamed('type', 'prd_type')

mth_prd = master_df.filter((F.col('last_n_month_eff_cvg').between(-12, -1)) & 
                           (~F.col('pol_stat_cd').isin(['A', 'N', 'R'])))\
                          .groupBy(['wa_cd_1', 'type']).agg(F.sum('adjust_ape').alias('sum_ape_agt'))\
                          .withColumnRenamed('wa_cd_1', 'agt_cd').withColumnRenamed('type', 'prd_type')
#mth_prd.display()
#ape_all_agt = master_df.filter(F.col('year_month') == mth_partition)\
#                            .groupBy(['wa_cd_1']).agg(F.sum('adjust_ape').alias('adjust_ape')).toPandas()

#ape_all_agt[ape_all_agt['wa_cd_1'] == '10539']

# COMMAND ----------

# Prepare claim KPIs
tclaim['agt_cd'] = tclaim['pol_num'].map(lambda x: pol_agt[x] if x in pol_agt else np.nan)
tclaim['clm_recv_dt'] = pd.to_datetime(tclaim['clm_recv_dt'])
#tclaim['adj_aprov_amt'] = tclaim['adj_aprov_amt']*1000
#tclaim['clm_prvd_amt'] = tclaim['clm_prvd_amt']*1000

y3 = tclaim[(tclaim['clm_recv_dt'] <= max_date)
            &(tclaim['clm_recv_dt'] >= max_date - pd.offsets.DateOffset(years=3))
            ].groupby(['agt_cd'])['clm_id'].nunique().astype(int)

y3.name = 'claim_cnt_last_3_yr'

y3_amt = tclaim[(tclaim['clm_recv_dt'] <= max_date)
                &(tclaim['clm_recv_dt'] >= max_date - pd.offsets.DateOffset(years=3))
               ].groupby(['agt_cd'])['clm_prvd_amt'].sum()

y3_amt.name = 'claim_amt_last_3_yr'

y3_appr = tclaim[(tclaim['clm_recv_dt'] <= max_date)
                &(tclaim['clm_recv_dt'] >= max_date - pd.offsets.DateOffset(years=3))
                &(tclaim['clm_stat_code'] == 'A')
               ].groupby(['agt_cd'])['clm_id'].nunique().astype(int)

y3_appr.name = 'claim_appr_cnt_last_3_yr'

y3_appr_amt = tclaim[(tclaim['clm_recv_dt'] <= max_date)
                   &(tclaim['clm_recv_dt'] >= max_date - pd.offsets.DateOffset(years=3))
                    &(tclaim['clm_stat_code'] == 'A')
                ].groupby(['agt_cd'])['clm_aprov_amt'].sum()

y3_appr_amt.name = 'claim_appr_amt_last_3_yr'

y1_wip = tclaim[(tclaim['clm_recv_dt'] <= max_date)
                &(tclaim['clm_recv_dt'] >= max_date - pd.offsets.DateOffset(years=1))
                &(tclaim['clm_stat_code'] == 'I')
               ].groupby(['agt_cd'])['clm_id'].nunique().astype(int)

y1_wip.name = 'claim_wip_last_1_yr'

y3_appr_rate = y3_appr/y3
y3_appr_rate.name = 'claim_appr_rate_last_3_yr'

clm = pd.concat([y3, y3_amt, y3_appr, y3_appr_amt, y1_wip], axis = 1).fillna(0).astype(int)
clm['claim_appr_rate_last_3_yr'] = clm['claim_appr_cnt_last_3_yr']/clm['claim_cnt_last_3_yr']

#clm['chart'] = '3yr_claim'
#clm['chart_kpi'] = 'pol_cnt_or_amt'

clm = clm.reset_index()

final = final.merge(clm, on='agt_cd', how='left')

final[final['agt_cd'].isin(['10193','10539'])]

# COMMAND ----------

# from lapse model > filter by snapshot please
lapse['agt_cd'] = lapse['pol_num'].map(lambda x: pol_agt[x] if x in pol_agt else np.nan)
lapse = lapse[(~lapse['agt_cd'].isnull())&
              (lapse['month_snapshot'] == snapshot)].reset_index(drop = True)

# lapse model changed to add 1 month buffer
lps = lapse[#(lapse['pd_to_dt'].dt.year >= int(max_date_str[:4]))&
            (lapse['lapse_score']>=0.05)] 
lps = lps.sort_values('lapse_score',ascending = False).groupby('agt_cd').head(3)
lps = lps[['agt_cd','pol_num','lapse_score']]#,'po_num','cli_nm','pmt_mode','pol_eff_dt','pd_to_dt']]
lps_diag = lps.groupby(['agt_cd'])['pol_num'].apply(list).reset_index().rename(columns={'pol_num': 'lps_pol_list'})
lps_diag['lps_pol_list'] = lps_diag['lps_pol_list'].map(lambda x: list(set(x)))

final = final.merge(lps_diag, on='agt_cd', how='left')

final[final['agt_cd'].isin(['10193','10539'])]

# COMMAND ----------

# leads from existing customers
# leads models output
#leads_existing_model = leads_existing_model[leads_existing_model['image_date'] == max_date_str]
leads_existing_model['po_num'] = leads_existing_model['po_num'].astype(int)
leads_existing_model['agt_cd'] = leads_existing_model['po_num'].map(lambda x: po_agt[x] if x in po_agt else np.nan)
#leads_existing_model = leads_existing_model[leads_existing_model['agt_cd'].isin(agt_list)]
leads_existing_model = leads_existing_model.sort_values('p_1',ascending = False).groupby('agt_cd').head(10)
#leads_existing_model.head(2)
multiclass.columns = [i.replace('rep_purchase_comb_','') if 'rep_purchase_comb_' in i else i for i in multiclass.columns]
multiclass.columns = [i.replace('_PREDICTION','') if '_PREDICTION' in i else i for i in multiclass.columns]
multiclass = multiclass[multiclass['DEPLOYMENT_APPROVAL_STATUS']=='APPROVED']

multiclass['which_prd'] = multiclass[['health_base','inv_base','health_rider','riders']].idxmax(axis=1)
multiclass['inv_base'] = multiclass['which_prd'].map(lambda x: '*' if x == 'inv_base' else '')
#use 0.1 as cut off as the model use softmax as activation function
multiclass['health_base'] = multiclass['health_base'].map(lambda x: '*' if x > 0.1 else '')
multiclass['riders'] = multiclass['riders'].map(lambda x: '*' if x > 0.1 else '')
multiclass['health_rider'] = multiclass['health_rider'].map(lambda x: '*' if x > 0.1 else '')
#multiclass.head(2)

nm = pd.merge(leads_existing_model, multiclass, on = 'po_num', how = 'left')

nm = nm[~nm['which_prd'].isnull()]
nm['pol_num'] = nm['po_num'].astype(str).map(lambda x: po_pol_dict[x] if x in po_pol_dict else np.nan)
nm = nm[~nm['pol_num'].isnull()]
nm = nm.sort_values('p_1',ascending = False).groupby('agt_cd').head(10)
nm = nm[['agt_cd','po_num','pol_num','health_base','inv_base','health_rider','riders','p_1']]

nm_diag = nm.sort_values('p_1',ascending = False).groupby('agt_cd').head(3)
nm_diag = nm_diag.groupby(['agt_cd'])['pol_num'].apply(list).reset_index().rename(columns={'pol_num': 'cpm_pol_list'})

final = final.merge(nm_diag, on='agt_cd', how='left')

final[final['agt_cd'].isin(['10193','10539'])]

# COMMAND ----------

# KPIs for APE tier (non-Platinum)
#agt_tier = agt.rename(columns = {'agt_code':'agt_cd'})

next_tier = dict({'Unranked': 'Silver', 'Silver': 'Gold', 'Gold': 'Platinum', 'Platinum': 'MDRT', 'MDRT': 'COT', 'COT': 'TOT', 'TOT': 'TOT'})
ape_all_agt = final.groupby(['agt_cd'])['last_mth_ape'].sum()
last_mth_ape_dict = ape_all_agt.to_dict()
last_mth_ape_rank_dict = ape_all_agt.rank(ascending=False).astype(int).to_dict()
#display(last_mth_ape_rank_dict)
#agt_tier['last_mth_ape'] = agt_tier['agt_cd'].map(lambda x: last_mth_ape_dict[x] if x in last_mth_ape_dict else 0)
final['rank_of_last_mth_ape'] = final['agt_cd'].map(lambda x: last_mth_ape_rank_dict[x] if x in last_mth_ape_rank_dict else np.nan)
final['next_tier'] = final['tier'].map(lambda x: next_tier[x])
final['next_tier_benchmark'] = final['next_tier'].map(lambda x: ape_benchmark[x] if x in ape_benchmark else np.nan)
final['tenure_ym'] = final['tenure_mth'].map(lambda x: str(divmod(x, 12)[0]) + 'y ' + str(divmod(x, 12)[1]) + 'm')

final[final['agt_cd'].isin(['10193','10539'])]

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Additional APE related metrics</strong>

# COMMAND ----------

final['gap_to_next_tier'] = final['next_tier_benchmark'] - final['ape_ytd'].fillna(0)
final['gap_to_next_tier'] = final['gap_to_next_tier'].map(lambda x: 0 if x < 0 else x)

final[final['agt_cd'].isin(['10193','10539'])]

# COMMAND ----------

# list out top 10 agents ranked by last month APE
final[(final['rank_of_last_mth_ape']<10)&
      #(final['tier']=='COT')&
      (final['last_mth_ape']!=0)].sort_values(by=['rank_of_last_mth_ape'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 13. Save result

# COMMAND ----------

final['image_date'] = max_date_str
final.to_parquet(f'{out_path}TPARDM_MTHEND/', partition_cols=['image_date'], engine='pyarrow')

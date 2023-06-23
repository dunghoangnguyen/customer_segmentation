# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import joblib

import copy 
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

# COMMAND ----------

# Setting up parameters
from datetime import datetime, timedelta
import calendar

# Get the last month-end from current system date
#last_mthend = datetime.strftime(datetime.now().replace(day=1) - timedelta(days=1), '%Y-%m-%d')

x = 2 # Change to number of months ago (0: last month-end, 1: last last month-end, ...)
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
# MAGIC Get Source Tables

# COMMAND ----------

tagtdm = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TAGTDM_DAILY', header=True)
agent_tier = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_AMS_DB/TAMS_AGENTS', header=True)
agent_scorecard = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/AGENT_SCORECARD/', header=True)
banca_banks = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_AMS_DB/TAMS_LOCATIONS', header=True)
tcustdm_daily = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TCUSTDM_DAILY', header=True)
tpolicys = spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_CASM_CAS_SNAPSHOT_DB/TPOLICYS/image_date={last_mthend}', header=True)
tcoverages = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_CAS_DB/TCOVERAGES', header=True)
tclient_policy_links = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_CAS_DB/TCLIENT_POLICY_LINKS', header=True)
tfield_values = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_CAS_DB/TFIELD_VALUES', header=True)
tpos = spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_REPORTS_DB/TPOS_COLLECTION/image_date={last_mthend}', header=True)

#unique per plan code
#select *, row_number() over(PARTITION BY plan_code order by effective_qtr asc) as rown from vn_published_campaign_db.vn_plan_code_map

vn_plan_code_map = spark.read.format("csv").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/nbv_margins/vn_plan_code_map_unique.csv', header = True)


#nbv margins not in PROD
#unique per plan code and effective qtr
nbv_margin_histories = spark.read.format("csv").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/nbv_margins/nbv_margins.csv', header = True)

#tpolidm_mthend

tpolidm_mthend = spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TPOLIDM_MTHEND/image_date={last_mthend}', header=True)

#tagtdm_mthend

tagtdm_mthend= spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TAGTDM_MTHEND/image_date={last_mthend}', header=True)

tcustdm_mthend= spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_DATAMART_DB/TCUSTDM_MTHEND/image_date={last_mthend}', header=True)


#propensity scores

existing_score = spark.read.format("csv").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/score/ex_score.csv', header = True) 
new_score = spark.read.format("csv").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/score/new.csv', header = True) 

#customer lifestage

lifestage  = spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/CUST_LIFESTAGE/monthend_dt={last_mthend_sht}', header=True)
cus_rfm  = spark.read.format("parquet").load(f'abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/CUS_RFM/monthend_dt={last_mthend_sht}', header=True)

#lapse (scores is up to Feb-23)
#get rown =1
#select * , row_number() over (PARTITION BY pol_num order by month_sp desc) AS ROWN from vn_lab_project_lapse_model_db.lapse_mthly
lapse_score_next_due = spark.read.format("csv").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/score/lapse_scores_next_due.csv', header = True) 


#early lapse
#lapse prediction at UW
early_lapse = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Curated/VN/Master/VN_CURATED_CUSTOMER_ANALYTICS_DB/EARLY_LAPSE_UW_POLICY_SCORE_DM', header = True)

#MOVE
muser_flat = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_MOVE5_MONGO_DB/MUSER_FLAT' , header = True)
manulifemember_flat = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_MOVE5_MONGO_DB/MANULIFEMEMBER_FLAT' , header = True)
movekey_flat = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_MOVE5_MONGO_DB/MOVEKEY_FLAT' , header = True)
userstate_flat = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_MOVE5_MONGO_DB/USERSTATE_FLAT' , header = True)
hit_data = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_ADOBE_MOVE5_DB/HIT_DATA' , header = True)

#CWS
sf_account   = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_SFDC_EASYCLAIMS_DB/ACCOUNT' , header = True)
cws_hit_data = spark.read.format("parquet").load('abfss://prod@abcmfcadovnedl01psea.dfs.core.windows.net/Published/VN/Master/VN_PUBLISHED_ADOBE_PWS_DB/HIT_DATA' , header = True)


# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Intermediate Tables</strong>

# COMMAND ----------

cli_type = tcustdm_daily.select('cli_num', 'cli_typ')
cli_contact = tcustdm_daily.select('cli_num', 'mobl_phon_num', 'email_addr')

agent_tier2 = agent_tier.filter(F.col('mdrt_desc').isin(['MDRT','TOT','COT']))\
    .select(
        'agt_code',
        'mdrt_ind',
        'mdrt_desc',
        'fc_ind',
        'fc_desc',
        'mba_ind',
        'mba_desc'
    )

agent_scorecard = agent_scorecard.filter(F.col('monthend_dt') == last_mthend)\
    .join(agent_tier2, on='agt_code', how='left')\
    .withColumn('mdrt_flag', F.when(F.col('agent_group')=='MDRT', 1).otherwise(0))\
    .withColumn('mdrt_tot_flag', F.when(F.col('mdrt_desc')=='TOT', 1).otherwise(0))\
    .withColumn('mdrt_cot_flag', F.when(F.col('mdrt_desc')=='COT', 1).otherwise(0))\
    .withColumn('fc_flag', F.when(F.col('agent_tier')=='FC', 1).otherwise(0))\
    .withColumn('mba_flag', F.when(F.col('agent_tier')=='MBA', 1).otherwise(0))\
    .withColumn('active_1m_flag', F.when(F.col('agent_tier')=='1mAA', 1).otherwise(0))\
    .withColumn('active_3m_flag', F.when(F.col('agent_tier')=='3mAA', 1).otherwise(0))\
    .select(
        agent_scorecard['agt_code'],
        'agent_group',
        'agent_tier',
        'agent_cluster',
        'agent_taskforce',
        'mdrt_flag',
        'mdrt_tot_flag',
        'mdrt_cot_flag',
        'fc_flag',
        'mba_flag',
        'active_1m_flag',
        'active_3m_flag',
    )


tclient_policy_links = tclient_policy_links.filter((tclient_policy_links.LINK_TYP == "O") & (tclient_policy_links.REC_STATUS == "A"))

tcoverages_all = tcoverages.select('pol_num','plan_code','vers_num','cvg_eff_dt','xpry_dt','ins_typ', 'dscnt_prem', 'prem_dur')\
                            .withColumn('prem_dur_pre', F.least(F.floor(F.datediff(F.lit(last_mthend), F.col('cvg_eff_dt'))/365.25), F.col('prem_dur')))\
                            .withColumn('prem_dur_pre', F.when(F.col('prem_dur_pre')>F.col('prem_dur'),F.col('prem_dur')).otherwise(F.col('prem_dur_pre')).cast('int'))

banca_banks  = banca_banks.filter((banca_banks.CHNL == "BK")).sort('loc_cd').dropDuplicates()

tfield_values = tfield_values.filter(tfield_values.FLD_NM == 'INS_TYP')

# Calculate the date of past premium due
tpolicys = tpolicys\
            .withColumn('prev_due', F.when(F.col('pmt_mode') =='12', F.add_months(F.col('pd_to_dt'),-12))
                                     .when(F.col('pmt_mode') =='01',F.add_months(F.col('pd_to_dt'),-1))
                                     .when(F.col('pmt_mode') =='03',F.add_months(F.col('pd_to_dt'),-3))
                                     .when(F.col('pmt_mode') =='06',F.add_months(F.col('pd_to_dt'),-6)))\
            .withColumn('inforce_yr', F.floor(F.datediff(F.coalesce('pol_trmn_dt', F.lit(last_mthend)), 'pol_eff_dt')/365.25))


# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Identify first and second policies</strong>

# COMMAND ----------

# Get insurance type and APE of the first and second (if any) policy
tpoli_first = tpolidm_mthend.select('po_num', 'pol_num', 'plan_code', 'pol_eff_dt', 'tot_ape')\
    .join(tpolicys.select('pol_num', 'ins_typ_base'), on='pol_num')\
    .withColumn('rn', F.row_number().over(Window.partitionBy('po_num').orderBy('pol_eff_dt')))\
    .join(tfield_values.select('fld_valu', 'fld_valu_desc_eng'), on=tpolicys['ins_typ_base'] == tfield_values['fld_valu'], how='left')\
    .groupBy('po_num')\
    .pivot('rn', [1, 2])\
    .agg(
        F.first('plan_code').alias('plan_code'),
        F.first('ins_typ_base').alias('ins_typ_base'),
        F.first('pol_eff_dt').alias('pol_eff_dt'),
        F.first('tot_ape').alias('tot_ape'),
        F.first('fld_valu_desc_eng').alias('ins_typ_desc')
    )\
    .select(
        'po_num',
        F.col('1_plan_code').alias('plan_code_1'),
        F.col('1_ins_typ_base').alias('ins_typ_base_1'),
        F.to_date(F.col('1_pol_eff_dt')).alias('pol_eff_dt_1'),
        F.col('1_tot_ape').cast('int').alias('tot_ape_1'),
        F.col('1_ins_typ_desc').alias('ins_typ_desc_1'),
        F.col('2_plan_code').alias('plan_code_2'),
        F.to_date(F.col('2_pol_eff_dt')).alias('pol_eff_dt_2'),
        F.col('2_tot_ape').cast('int').alias('tot_ape_2'),
        F.col('2_ins_typ_desc').alias('ins_typ_desc_2'),
        (F.floor(F.datediff(F.col('2_pol_eff_dt'), F.col('1_pol_eff_dt'))/365.25)).alias('yr_2nd_prod')
    )
#tpoli_first.display(20)

# COMMAND ----------

#early_lapse_sum = early_lapse.groupBy(F.col('pol_num'))\
#    .agg(F.max(F.col('p_1')).alias('p_1'),
#         F.min(F.col('decile')).alias('decile'),
#         F.max(F.col('pol_eff_dt')).alias('lst_eff_dt'))

lapse_score_next_due = lapse_score_next_due.withColumn('lapse_score', F.col('lapse_score').cast('float'))

# COMMAND ----------

# MAGIC %md
# MAGIC Create Views

# COMMAND ----------

tagtdm.createOrReplaceTempView("tagtdm")
#agent_tier2.createOrReplaceTempView("agent_tier2")
agent_scorecard.createOrReplaceTempView("agent_scorecard")
banca_banks.createOrReplaceTempView("banca_banks")
tcustdm_daily.createOrReplaceTempView("tcustdm_daily")
cli_contact.createOrReplaceTempView("cli_contact")
cli_type.createOrReplaceTempView("cli_type")

tpolicys.createOrReplaceTempView("tpolicys")
tcoverages_all.createOrReplaceTempView("tcoverages_all")
tclient_policy_links.createOrReplaceTempView("tclient_policy_links")
tfield_values.createOrReplaceTempView("tfield_values")
#vn_plan_code_map.createOrReplaceTempView("vn_plan_code_map")
nbv_margin_histories.createOrReplaceTempView("nbv_margin_histories")

tpolidm_mthend.createOrReplaceTempView("tpolidm_mthend")
tagtdm_mthend.createOrReplaceTempView("tagtdm_mthend")
tcustdm_mthend.createOrReplaceTempView("tcustdm_mthend")

# Scoring tables
existing_score.createOrReplaceTempView("existing_score")
new_score.createOrReplaceTempView("new_score")
early_lapse.createOrReplaceTempView("early_lapse")

# MOVE and CWS tables
muser_flat.createOrReplaceTempView("muser_flat")
manulifemember_flat.createOrReplaceTempView("manulifemember_flat")
movekey_flat.createOrReplaceTempView("movekey_flat")
userstate_flat.createOrReplaceTempView("userstate_flat")
hit_data.createOrReplaceTempView("hit_data")
sf_account.createOrReplaceTempView("sf_account")
cws_hit_data.createOrReplaceTempView("cws_hit_data")


# COMMAND ----------

# MAGIC %md
# MAGIC Policy and Coverage Base

# COMMAND ----------


policy_base = spark.sql(f"""
    select cov.pol_num
    ,cov.cvg_eff_dt
    ,cov.dscnt_prem
    ,cov.plan_code
    ,cov.vers_num
    ,cast(cov.prem_dur as int) prem_dur
    ,cov.prem_dur_pre
    ,cast(cov.prem_dur-cov.prem_dur_pre as int) as prem_dur_post
    ,pol.dist_chnl_cd
    ,pol.pol_stat_cd
    ,case when pol.pol_stat_cd in ('1','2','3','5') then 1 else 0 end as f_inforce_ind
    ,case when pol.pol_stat_cd in ('B') then 1 else 0 end as f_lapse_ind
    ,case when pol.pol_stat_cd in ('E') then 1 else 0 end as f_surr_ind
    ,case when pol.pol_stat_cd in ('F','H','D','M','T') then 1 else 0 end as f_mature_ind
    ,case when pol.pol_stat_cd in ('A') then 1 else 0 end as f_nottaken_ind
    ,case when pol.pol_stat_cd in ('C','L','N','R','X') then 1 else 0 end f_ter_ind
    ,case when pol.pol_stat_cd in ('4','7','9') then 1 else 0 end as f_paid_ind
    ,pol.pmt_mode
    ,pol.bill_mthd
    ,case when pol.agt_code = pol.wa_cd_1 then 1 else 0 end as f_same_agent
    ,tfield.fld_valu as ins_typ
    ,tfield.fld_valu_desc_eng as ins_typ_desc
    ,tclient.cli_num as po_num
    ,contact.mobl_phon_num as cli_mobile
    ,contact.email_addr as cli_email
    ,agt.loc_cd
    ,case when agt.comp_prvd_num not in ('04','05','34','36','97','98') then datediff(coalesce(agt.agt_term_dt, '{last_mthend}'), agt.agt_join_dt)/365.25 else null end as agt_tenure_yrs
    ,agt_scr.agent_group
    ,agt_scr.agent_tier
    ,agt_scr.agent_cluster
    ,agt_scr.agent_taskforce
    ,agt_scr.mdrt_flag
    ,agt_scr.mdrt_tot_flag
    ,agt_scr.mdrt_cot_flag
    ,agt_scr.active_1m_flag
    ,agt_scr.active_3m_flag
    ,ctyp.cli_typ
    ,bnk.office_cd
    ,case when pdm.lst_pmt_mthd = 'Séc' then 1 else 0 end as f_cheque
    ,case when pdm.lst_pmt_mthd = 'Tiền mặt' then 1 else 0 end as f_cash
    ,case when pdm.lst_pmt_mthd = 'Chuyển khoản' then 1 else 0 end as f_auto
    ,pol.inforce_year
    from tpolicys pol
    left join tcoverages_all cov on pol.pol_num = cov.pol_num
    left join tpolidm_mthend pdm on pol.pol_num = pdm.pol_num
    left join tfield_values tfield on pol.ins_typ_base = tfield.fld_valu
    left join tclient_policy_links tclient on pol.pol_num =tclient.pol_num
    left join cli_contact contact on tclient.cli_num = contact.cli_num
    left join tagtdm agt on pol.agt_code = agt.agt_code
    left join agent_scorecard agt_scr on pol.agt_code = agt_scr.agt_code
    left join banca_banks bnk on agt.loc_cd = bnk.loc_cd
    left join cli_type ctyp on tclient.cli_num =ctyp.cli_num
--Add tpos
""")
#policy_base.where(F.col('f_inforce_ind')==1).select('pol_num', 'plan_code','prem_dur', 'prem_dur_pre', 'prem_dur_post').show(20)

#Combine Scores
propensity_scores = spark.sql("""
                       with existing as 
                       (select cli_num as po_num , least(decile_inv
                                                    ,decile_ci
                                                    ,decile_lp
                                                    ,decile_lt
                                                    ,decile_acc
                                                    ,decile_med) as min_decile from existing_score),
                       new as 
                      (select po_num, least(ci_decile
                                        ,invst_decile
                                        ,lt_decile) as min_decile from new_score)
                      select po_num, min_decile from existing 
                      union all 
                      select po_num, min_decile from new
""")



move_info = spark.sql("""
                             with move_acc_mthend as
                             (
                                 select
                                mu.`_id` muser_id
                                ,mk.`value` movekey
                                ,from_unixtime(unix_timestamp(mk.activationdate,"yyyy-MM-dd")) activation_date
                                ,from_unixtime(unix_timestamp(urt.lastdatasync,"yyyy-MM-dd")) last_data_sync_date
                                                from
                                muser_flat mu
                                inner join manulifemember_flat me on (mu.`_id` = me.userid)
                                left join movekey_flat mk on (me.keyid = mk.`_id`)
                                left join userstate_flat urt on (mu.`_id` = urt.userid)
                            where
                                mk.activated = 1
                             ),
                             move_login_transactions as (
                                    select
                                        post_evar1 muser_id
                                        ,from_unixtime(unix_timestamp(date_time,"yyyy-MM-dd")) login_dt
                                        ,post_pagename
                                        ,concat(post_visid_high, post_visid_low) visitor_id
                                        ,concat(post_visid_high,post_visid_low,visit_num,visit_start_time_gmt) visit_id
                                        ,visit_page_num
                                        ,visit_num
                                        ,date_time
                                        ,case
                                            when post_mobileosversion like '%Android%' then 'Android'
                                            else 'iOS'
                                        end os
                                        ,row_number() over(partition by post_evar1 order by date_time asc) rw_num
                                    from
                                        hit_data
                                    where
                                        exclude_hit = 0
                                        and hit_source not in ('5', '7', '8', '9')
                                        and concat(post_visid_high, post_visid_low) is not null
                                ),
                                move_login as (
                                select
                                    muser_id
                                    ,max(login_dt) lst_login_dt
                                from
                                    move_login_transactions
                                where
                                    rw_num <> 1
                                    and login_dt <= last_day(add_months(current_date,-1))
                                group by
                                    muser_id
                                ),
                                move_info as (
                                select
                                    substr(a.movekey,2,length(a.movekey)-1) cli_num
                                    ,a.activation_date
                                    ,b.lst_login_dt
                                from
                                    move_acc_mthend a
                                    left join move_login b on (a.muser_id = b.muser_id)
                            )select * from move_info where lst_login_dt <= '2023-03-31' and activation_date <= '2023-03-31'
                         
                        """)

move_info.createOrReplaceTempView("move_info")

move_information_mthend = spark.sql("""
                                    with rs_dis as (
                                    select
                                        cli_num
                                        ,activation_date
                                        ,lst_login_dt
                                        ,row_number() over(partition by cli_num order by activation_date desc,lst_login_dt desc) rw_num
                                    from
                                        move_info
                                )
                                select * from rs_dis where rw_num = 1
                                """)


cws_information_mthend = spark.sql("""
                                   with cws_acc as (
                                    select
                                        external_id__c cli_num
                                        ,mcf_user_id__pc acc_id
                                    from
                                        sf_account
                                    where
                                        mcf_user_id__pc is not null
                                )
                                ,cws_login_transactions as(
                                    select
                                        hd.post_evar37 as login_id
                                        ,concat(hd.post_visid_high, hd.post_visid_low, hd.visit_num) as visit_id
                                        ,hd.date_time as login_date_time
                                        ,row_number() over(partition by hd.post_evar37 order by hd.date_time asc) rw_num
                                    from
                                        cws_hit_data hd	
                                    where
                                        1=1
                                        and hd.exclude_hit = 0
                                        and hd.hit_source not in ('5', '7', '8', '9')
                                        and concat(hd.post_visid_high, hd.post_visid_low) is not null
                                        and hd.post_evar37 <> ''
                                        and hd.post_evar19 = '/portfolio/policies'
                                        and hd.user_server in ('hopdongcuatoi.manulife.com.vn','hopdong.manulife.com.vn')
                                )
                                ,cws_reg as (
                                    select
                                        login_id
                                        ,login_date_time reg_dt
                                    from
                                        cws_login_transactions
                                    where
                                        rw_num = 1
                                )
                                ,cws_login as (
                                    select
                                        login_id
                                        ,max(login_date_time) lst_login_dt
                                    from
                                        cws_login_transactions
                                    where
                                        rw_num > 1
                                        and login_date_time <= last_day(add_months(current_date,-1))
                                    group by
                                        login_id
                                )
                                ,cws_infor as (
                                    select
                                        a.cli_num
                                        ,b.reg_dt cws_joint_dt
                                        ,c.lst_login_dt
                                    from
                                        cws_acc a
                                        left join cws_reg b on (a.acc_id = b.login_id)
                                        left join cws_login c on (a.acc_id = c.login_id)
                                ) select * from cws_infor where lst_login_dt <= '2023-03-31' and cws_joint_dt <= '2023-03-31'
                                   
                                   """)



# COMMAND ----------

#TPOS and TPolicys

#1 Since there is no indicator which payment refers to due date, 

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate NBV per Coverage

# COMMAND ----------

    
all_coverage_nbv = policy_base.withColumn('effective_qtr', F.when(F.month('cvg_eff_dt')<=3, F.concat(F.year('cvg_eff_dt')-1, F.lit(' Q3')) )
                                            .when(F.month('cvg_eff_dt')<=6, F.concat(F.year('cvg_eff_dt')-1, F.lit(' Q4')))
                                            .when(F.month('cvg_eff_dt')<=9, F.concat(F.year('cvg_eff_dt'), F.lit(' Q1')))
                                            .when(F.month('cvg_eff_dt')<=12, F.concat(F.year('cvg_eff_dt'), F.lit(' Q2'))))\
                                            .join(vn_plan_code_map.select('plan_code',
                                                 'nbv_margin_agency_affinity',
                                                 'nbv_margin_agency',
                                                 'nbv_margin_dmtm',
                                                 'nbv_margin_other_channel_affinity',
                                                 'nbv_margin_other_channel',
                                                 'nbv_margin_banca_other_banks',
                                                 'nbv_margin_banca_scb',
                                                 'nbv_margin_banca_tcb'),
                                                on='plan_code', how='left')\
                                            .join(nbv_margin_histories.select('plan_code','effective_qtr',
                                                 F.col('nbv_margin_agency_affinity').alias('nbv_margin_agency_affinity2'),
                                                 F.col('nbv_margin_agency').alias('nbv_margin_agency2'),
                                                 F.col('nbv_margin_dmtm').alias('nbv_margin_dmtm2'),
                                                 F.col('nbv_margin_other_channel_affinity').alias('nbv_margin_other_channel_affinity2'),
                                                 F.col('nbv_margin_other_channel').alias('nbv_margin_other_channel2'),
                                                 F.col('nbv_margin_banca_other_banks').alias('nbv_margin_banca_other_banks2'),
                                                 F.col('nbv_margin_banca_scb').alias('nbv_margin_banca_scb2'),
                                                 F.col('nbv_margin_banca_tcb').alias('nbv_margin_banca_tcb2')), on=['plan_code', 'effective_qtr'], how='left')\
                                            .withColumn('nbv_margin', F.when(F.col('loc_cd').isNull(),
                                                    F.when(F.col('dist_chnl_cd').isin(['03','10','14','16','17','18','19','22','23','24','25','29','30','31','32','33','39','41','44','47','49','51','52','53']), F.coalesce(F.col('nbv_margin_banca_other_banks'), F.col('nbv_margin_banca_other_banks2')))
                                                     .when(F.col('dist_chnl_cd').isin(['48']), F.coalesce(F.col('nbv_margin_other_channel_affinity'), F.col('nbv_margin_other_channel_affinity2')))
                                                     .when(F.col('dist_chnl_cd').isin(['01', '02', '08', '50', '*']), F.coalesce(F.col('nbv_margin_agency'), F.col('nbv_margin_agency2')))
                                                     .when(F.col('dist_chnl_cd').isin(['05','06','07','34','36']), F.coalesce(F.col('nbv_margin_dmtm'), F.col('nbv_margin_dmtm2')))
                                                     .when(F.col('dist_chnl_cd').isin(['09']), F.lit(-1.34041044648343))
                                                     .otherwise(F.coalesce(F.col('nbv_margin_other_channel'), F.col('nbv_margin_other_channel2'))))
                                            .when(F.col('dist_chnl_cd').isin(['*']), F.coalesce(F.col('nbv_margin_agency'), F.col('nbv_margin_agency2')))
                                            .when(F.col('loc_cd').like('TCB%'), F.coalesce(F.col('nbv_margin_banca_tcb'), F.col('nbv_margin_banca_tcb2')))
                                            .when(F.col('loc_cd').like('SAG%'), F.coalesce(F.col('nbv_margin_banca_scb'), F.col('nbv_margin_banca_scb2')))
                                            .otherwise(F.coalesce(F.col('nbv_margin_other_channel'), F.col('nbv_margin_other_channel2'))))\
                                            .withColumn('plan_nbv', F.col('dscnt_prem')/23.145*F.col('nbv_margin'))\
                                            .withColumn('coverage_ape', F.col('dscnt_prem')*12/(F.col('pmt_mode'))/23.145)\
                                            .withColumn('annual_flag', F.when(F.col('pmt_mode')=='12', 1).otherwise(0))\
                                            .withColumn('valid_email', F.when(F.col('cli_email').isNotNull(), 1).otherwise(0))\
                                            .withColumn('valid_mobile', F.when(F.col('cli_mobile').isNotNull(), 1).otherwise(0))\
                                            .withColumn('channel', F.when(F.col('dist_chnl_cd').isin(['01', '02', '08', '50', '*']), 'Agency')
                                                                    .when(F.col('dist_chnl_cd').isin(['05','06','07','34','36']), 'DMTM')
                                                                    .otherwise(F.col('office_cd')))

all_coverage_nbv = all_coverage_nbv.filter(~F.col('pol_stat_cd').isin(['6','8']))                                           
inforce_coverage_nbv = all_coverage_nbv.filter(F.col('f_inforce_ind')==1)

#inforce_coverage_nbv.createOrReplaceTempView("inforce_coverage_nbv")
print("all_coverage_nbv:", all_coverage_nbv.count())
#print("inforce_coverage_nbv:", inforce_coverage_nbv.count())


# COMMAND ----------

#print(inforce_coverage_nbv.count())
#policy_base.filter(F.col('pol_stat_cd').isin(['1','2','3','5','7'])==True).count()

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Channel Determination</strong>

# COMMAND ----------

cli_channel = all_coverage_nbv.select('po_num', 'cvg_eff_dt', 'channel')\
    .withColumn('rn', F.row_number().over(Window.partitionBy('po_num').orderBy('cvg_eff_dt')))\
    .filter(F.col('rn')==1)\
    .select('po_num', 'channel')

cli_channel2 = all_coverage_nbv.withColumn('channel_agency_flag', F.when(F.col('channel')=='Agency', 1).otherwise(0))\
    .withColumn('channel_TCB_flag', F.when(F.col('channel')=='TCB', 1).otherwise(0))\
    .withColumn('channel_SAG_flag', F.when(F.col('channel')=='SAG', 1).otherwise(0))\
    .groupby('po_num').agg(F.when(F.sum('channel_agency_flag')>=1, 'Agency')
                                .when(F.sum('channel_TCB_flag')>=1, 'TCB')
                                .when(F.sum('channel_SAG_flag')>=1, 'SAG')
                                .otherwise('tbd').alias('channel_final'))\
    .join(cli_channel, on='po_num', how='left')\
    .withColumn('channel_final', F.when(F.col('channel_final')=='tbd', F.col('channel')).otherwise(F.col('channel_final')))


# COMMAND ----------

# MAGIC %md
# MAGIC <span style="font-size: 96px;"><strong>Customer Marketing Segmentation</strong></span>

# COMMAND ----------

all_cli_mkt_seg = all_coverage_nbv.groupBy('po_num')\
                    .agg(F.sum(F.col('coverage_ape')).alias('total_ape'),
                         F.countDistinct('pol_num').alias('pol_cnt'),
                         F.countDistinct(F.when(F.col('inforce_year') >= 10, F.col('pol_num'))).alias('10yr_pol_cnt'))
                    

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Policy and Client LTV Calculation</strong>

# COMMAND ----------


# get basic ltv
all_pol_ltv= all_coverage_nbv.groupby('pol_num')\
                            .agg(
                            F.sum('coverage_ape').alias('coverage_ape'),
                            F.sum('plan_nbv').alias('plan_nbv'),
                            (F.sum('plan_nbv')/F.sum('coverage_ape')).alias('cli_nbv_margin'),
                            ((F.sum('plan_nbv'))*(F.min('prem_dur'))).alias('pol_ltv'), # useprem_duration 
                            ((F.sum('plan_nbv'))*(F.min('prem_dur_pre'))).alias('pol_ltv_pre'),
                            ((F.sum('plan_nbv'))*(F.min('prem_dur_post'))).alias('pol_ltv_post')
                        )\
                        .join(tpolidm_mthend, on='pol_num', how='left')\
                        .select('pol_num', 'po_num', 'coverage_ape', 'plan_nbv' , 'cli_nbv_margin', 'pol_ltv', 'pol_ltv_pre', 'pol_ltv_post')\
                        .join(early_lapse.select('pol_num', 'decile', F.col('decile').alias('early_lapse_decile'), 'p_1') , on = 'pol_num', how ='left')\
                        .join(lapse_score_next_due.select('pol_num', 'decile', F.col('decile').alias('next_due_lapse_decile'), 'lapse_score'), on = 'pol_num', how ='left')
    

# add more stats on Pol ltv
# add lapse score to calculation logic
# new ltv = basic_ltv*(1-lapse_score)
all_pol_ltv = all_pol_ltv.withColumn('pol_ltv', (F.col('pol_ltv')*(1-F.coalesce('lapse_score', 'p_1', F.lit(0)))).alias('pol_ltv'))\
    .withColumn('pol_ltv_pre', (F.col('pol_ltv_pre')*(1-F.coalesce('lapse_score', 'p_1', F.lit(0)))).alias('pol_ltv_pre'))\
    .withColumn('pol_ltv_post', (F.col('pol_ltv_post')*(1-F.coalesce('lapse_score', 'p_1', F.lit(0)))).alias('pol_ltv_post'))

#all_pol_ltv.display(10)

# COMMAND ----------

all_pol_ltv.createOrReplaceTempView('all_pol_ltv')

all_pol_ltv_qtl = spark.sql("""select a.* , ntile(10) over (order by a.pol_ltv desc)  as poldecile from all_pol_ltv a """)

all_pol_ltv_qtl.groupBy('poldecile')\
    .agg(F.count('pol_num').alias('pol_count'),
    F.mean('pol_ltv').alias('mean_ltv'),
    F.mean('pol_ltv_pre').alias('mean_ltv_pre'),
    F.mean('pol_ltv_post').alias('mean_ltv_post'))\
    .show(10)

# COMMAND ----------

all_cli_ltv = all_pol_ltv.groupBy('po_num')\
                        .agg(
                             F.sum('coverage_ape').alias('coverage_ape'),
                            F.sum('plan_nbv').alias('plan_nbv'),
                            (F.sum('plan_nbv')/F.sum('coverage_ape')).alias('cli_nbv_margin'),
                            F.sum('pol_ltv').alias('cli_ltv'),
                            F.sum('pol_ltv_pre').alias('cli_ltv_pre'),
                            F.sum('pol_ltv_post').alias('cli_ltv_post'),
                            F.min('early_lapse_decile').alias('early_lapse_decile'),
                            F.min('next_due_lapse_decile').cast('int').alias('next_due_lapse_decile')

                        )\
                        .join(cli_channel2, on='po_num', how='left')\
                        .filter(F.col('channel_final')=='Agency')


all_pol_ltv.createOrReplaceTempView("all_pol_ltv")
all_pol_ltv.count()

all_cli_ltv.createOrReplaceTempView("all_cli_ltv")
all_cli_ltv.count()

# COMMAND ----------

all_cli_ltv.createOrReplaceTempView('all_cli_ltv')

all_cli_ltv_qtl = spark.sql("""select a.* , ntile(10) over (order by a.cli_ltv desc)  as poldecile from all_cli_ltv a """)

all_cli_ltv_qtl.groupBy('poldecile')\
    .agg(F.count('po_num').alias('customer_count'),
    F.mean('cli_ltv').alias('mean_ltv'),
    F.mean('cli_ltv_pre').alias('mean_ltv_pre'),
    F.mean('cli_ltv_post').alias('mean_ltv_post'))\
    .show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Client Level Data</strong>

# COMMAND ----------


#Client Level Data
all_client_coverage = all_coverage_nbv.groupby('po_num')\
    .agg(
        F.min('cvg_eff_dt').alias('first_pol_eff_dt'),
       (F.datediff(F.lit(last_mthend), F.min('cvg_eff_dt'))/365.25).alias('client_tenure'),
        F.countDistinct('pol_num').alias('pol_count'),
        F.countDistinct('ins_typ_desc').alias('ins_typ_count'),
        F.countDistinct(F.when(F.col("ins_typ_desc") == "Term Life" ,F.col("pol_num"))).alias('term_pol'),
        F.countDistinct(F.when(F.col("ins_typ_desc") == "Endowment" ,F.col("pol_num"))).alias('endow_pol'),
        F.countDistinct(F.when(F.col("ins_typ_desc") == "Health Indemnity" ,F.col("pol_num"))).alias('health_indem_pol'),
        F.countDistinct(F.when(F.col("ins_typ_desc") == "Whole Life" ,F.col("pol_num"))).alias('whole_pol'),
        F.countDistinct(F.when(F.col("ins_typ_desc") == "Investment" ,F.col("pol_num"))).alias('investment_pol'),
        F.countDistinct(F.when(F.col("f_inforce_ind") == 1 ,F.col("pol_num"))).alias('inforce_pol'),
        F.countDistinct(F.when(F.col("f_lapse_ind") == 1 ,F.col("pol_num"))).alias('lapsed_pol'),
        F.countDistinct(F.when(F.col("f_surr_ind") == 1 ,F.col("pol_num"))).alias('surrendered_pol'),
        F.countDistinct(F.when(F.col("f_mature_ind") == 1 ,F.col("pol_num"))).alias('matured_pol'),
        F.countDistinct(F.when(F.col("f_nottaken_ind") == 1 ,F.col("pol_num"))).alias('nottaken_pol'),
        F.countDistinct(F.when(F.col("f_ter_ind") == 1 ,F.col("pol_num"))).alias('terminated_pol'),
        F.countDistinct(F.when(F.col("f_paid_ind") == 1 ,F.col("pol_num"))).alias('oth_paid_pol'),
        F.max('f_same_agent').alias('f_same_agent'),
        F.sum('f_cash').alias('f_cash'),
        F.sum('f_cheque').alias('f_cheque'),
        F.sum('f_auto').alias('f_auto'),
        F.mean('agt_tenure_yrs').alias('agt_tenure_yrs'),
        (F.when(F.sum('mdrt_flag')>=1, 1).otherwise(0)).alias('mdrt_flag'),
        (F.when(F.sum('mdrt_tot_flag')>=1, 1).otherwise(0)).alias('mdrt_tot_flag'),
        (F.when(F.sum('mdrt_cot_flag')>=1, 1).otherwise(0)).alias('mdrt_cot_flag'),
        (F.when(F.sum('active_1m_flag')>=1, 1).otherwise(0)).alias('active_1m_flag'),
        (F.when(F.countDistinct('pol_num')>=2, 1).otherwise(0)).alias('multi_prod'),
        (F.when(F.sum('annual_flag')>=1, 1).otherwise(0)).alias('annual_flag'),
        (F.when(F.sum('valid_email')>=1, 1).otherwise(0)).alias('valid_email'),
        (F.when(F.sum('valid_mobile')>=1, 1).otherwise(0)).alias('valid_mobile'),
        F.sum('coverage_ape').alias('coverage_ape'),
        F.sum('plan_nbv').alias('plan_nbv')                
)
all_client_level = all_client_coverage\
                    .join(all_cli_ltv ,on='po_num', how='inner')\
                    .join(propensity_scores, on='po_num', how='left')\
                    .join(tcustdm_mthend, on=all_coverage_nbv['po_num']==tcustdm_mthend['cli_num'], how='left')\
                    .withColumn('f_HCM', F.when(F.col('CITY')=='Hồ Chí Minh',1).otherwise(0))\
                    .withColumn('f_HN', F.when(F.col('CITY')=='Hà Nội',1).otherwise(0))\
                    .withColumn('f_DN', F.when(F.col('CITY')=='Đà Nẵng',1).otherwise(0))\
                    .withColumn('f_oth_city',F.when(F.col('CITY').isin(['Hồ Chí Minh','Hà Nội','Đà Nẵng'])==False,1).otherwise(0))\
                    .join(lifestage, on=inforce_coverage_nbv['po_num']==lifestage['client_number'], how='left')\
                    .withColumn('f_adult_self_insured', F.when(F.col('customer_segment')=='Adult Self Insured',1).otherwise(0))\
                    .withColumn('f_family', F.when(F.col('customer_segment')=='Family',1).otherwise(0))\
                    .withColumn('f_family_wkids', F.when(F.col('customer_segment')=='Family with Kids',1).otherwise(0))\
                    .withColumn('f_empty_nest', F.when(F.col('customer_segment')=='Empty Nester',1).otherwise(0))\
                    .withColumn('f_undefined_segment', F.when(F.col('customer_segment')=='Undefined Segmentation',1).otherwise(0))\
                    .withColumn('f_male', F.when(F.col('cus_gender')=='Male',1).otherwise(0))\
                    .join(cus_rfm, on='po_num', how='left')\
                    .join(move_information_mthend, on=all_coverage_nbv['po_num'] == move_information_mthend['cli_num'], how='left')\
                    .withColumn('move_tenure_days',  (F.datediff(F.lit(last_mthend), F.col('activation_date'))))\
                    .withColumn('move_last_log_days',  (F.datediff(F.lit(last_mthend), move_information_mthend['lst_login_dt'])))\
                    .join(cws_information_mthend, on=all_coverage_nbv['po_num'] == cws_information_mthend['cli_num'], how='left')\
                    .withColumn('cws_tenure_days',  (F.datediff(F.lit(last_mthend), F.col('cws_joint_dt'))))\
                    .withColumn('cws_last_log_days',  (F.datediff(F.lit(last_mthend), cws_information_mthend['lst_login_dt'])))\
                    .join(all_cli_mkt_seg, on=all_coverage_nbv['po_num']==all_cli_mkt_seg['po_num'], how='left')\
                    .withColumn('f_vip_elite', F.when(F.col('total_ape')>=12961.76,1).otherwise(0))\
                    .withColumn('f_vip_plat', F.when((F.col('total_ape')>6480.88) & (F.col('total_ape')<12961.76),1).otherwise(0))\
                    .withColumn('f_vip_gold', F.when((F.col('total_ape')>=2808.38) & (F.col('total_ape')<6480.88),1).otherwise(0))\
                    .withColumn('f_vip_silver', F.when((F.col('total_ape')>=864.12) & (F.col('total_ape')<2808.38) & (F.col('10yr_pol_cnt')>=1),1).otherwise(0))\
                    .join(tpoli_first, on='po_num', how='left')\
                    .withColumn('f_1st_term', F.when(F.col("ins_typ_desc_1") == "Term Life",1).otherwise(0))\
                    .withColumn('f_1st_endow', F.when(F.col("ins_typ_desc_1") == "Endowment",1).otherwise(0))\
                    .withColumn('f_1st_health_indem', F.when(F.col("ins_typ_desc_1") == "Health Indemnity",1).otherwise(0))\
                    .withColumn('f_1st_whole', F.when(F.col("ins_typ_desc_1") == "Whole Life",1).otherwise(0))\
                    .withColumn('f_1st_invest', F.when(F.col("ins_typ_desc_1") == "Investment",1).otherwise(0))\
                    .withColumn('f_2nd_term', F.when(F.col("ins_typ_desc_2") == "Term Life",1).otherwise(0))\
                    .withColumn('f_2nd_endow', F.when(F.col("ins_typ_desc_2") == "Endowment",1).otherwise(0))\
                    .withColumn('f_2nd_health_indem', F.when(F.col("ins_typ_desc_2") == "Health Indemnity",1).otherwise(0))\
                    .withColumn('f_2nd_whole', F.when(F.col("ins_typ_desc_2") == "Whole Life",1).otherwise(0))\
                    .withColumn('f_2nd_invest', F.when(F.col("ins_typ_desc_2") == "Investment",1).otherwise(0))\
                    .select(all_client_coverage['po_num'], 
                            'first_pol_eff_dt',
                            'client_tenure',
                            'pol_count',
                            'ins_typ_count',
                            'term_pol',
                            'endow_pol',
                            'health_indem_pol',
                            'whole_pol',
                            'investment_pol',
                            'inforce_pol',
                            'lapsed_pol',
                            'surrendered_pol',
                            'matured_pol',
                            'nottaken_pol',
                            'terminated_pol',
                            'oth_paid_pol',
                            'f_same_agent',
                            'f_cash',
                            'f_cheque',
                            'f_auto',
                            'agt_tenure_yrs',
                            'mdrt_flag',
                            'mdrt_tot_flag',
                            'mdrt_cot_flag',
                            'active_1m_flag',
                            'multi_prod',
                            'annual_flag',
                            'valid_email',
                            'valid_mobile',
                            all_client_coverage['coverage_ape'],
                            all_client_coverage['plan_nbv'],
                            'cli_nbv_margin',
                            'cli_ltv',
                            'cli_ltv_pre',
                            'cli_ltv_post',
                            F.least('next_due_lapse_decile','early_lapse_decile').alias('lapse_decile'),
                            'channel_final',
                            'channel',
                            'min_decile',
                            'cur_age',
                            'FRST_ISS_AGE',
                            'CITY',
                            (F.col('MTHLY_INCM')/23.145).alias('MTHLY_INCM'),
                            cus_rfm['NO_DPND'],
                            'f_HCM',
                            'f_HN',
                            'f_DN',
                            'f_oth_city',
                            'customer_segment',
                            'cus_age_band',
                            'dependent_age_band',
                            'cus_gender',
                            'f_adult_self_insured',
                            'f_family',
                            'f_family_wkids',
                            'f_empty_nest',
                            'f_undefined_segment',
                            'f_male',
                            'k_inf_cvg_acc',
                            'k_inf_cvg_ci',
                            'k_inf_cvg_inv',
                            'k_inf_cvg_lp',
                            'k_inf_cvg_lts',
                            'k_inf_cvg_med',
                            'f_addrchg_1m',
                            'f_addrchg_3m',
                            'f_addrchg_6m',
                            'f_addrchg_12m',
                            'f_occpchg_1m',
                            'f_occpchg_3m',
                            'f_occpchg_6m',
                            'f_occpchg_12m',
                            'activation_date',
                            'move_tenure_days',
                            'move_last_log_days',
                            'cws_tenure_days',
                            'cws_last_log_days',
                            'pol_cnt',
                            'total_ape',
                            '10yr_pol_cnt',
                            'f_vip_elite',
                            'f_vip_plat',
                            'f_vip_gold',
                            'f_vip_silver',
                            'f_1st_term',
                            'f_1st_endow',
                            'f_1st_health_indem',
                            'f_1st_whole',
                            'f_1st_invest',
                            'f_2nd_term',
                            'f_2nd_endow',
                            'f_2nd_health_indem',
                            'f_2nd_whole',
                            'f_2nd_invest',
                            'yr_2nd_prod',
                            F.when(F.col('inforce_pol')>0, 1).otherwise(0).alias('inforce_ind'),
    )
                    
all_client_level.display(20)

# COMMAND ----------

all_client_level = all_client_level\
.select('po_num', 'first_pol_eff_dt', 'client_tenure', 'pol_count', 'inforce_pol', 'lapsed_pol', 'surrendered_pol', 'matured_pol', 'nottaken_pol', 'terminated_pol', 'oth_paid_pol', 
        'ins_typ_count', 'term_pol', 'endow_pol',	'health_indem_pol',	'whole_pol', 'investment_pol',	'f_same_agent',	'f_cash', 'f_cheque',	'f_auto',	'agt_tenure_yrs',	'mdrt_flag', 'mdrt_tot_flag', 'mdrt_cot_flag', 'active_1m_flag', 'multi_prod',	'annual_flag',	'valid_email',	'valid_mobile',	'coverage_ape',	'plan_nbv',	'cli_nbv_margin',	'cli_ltv', 'cli_ltv_pre', 'cli_ltv_post',	'lapse_decile',	'channel_final',	'channel',	'min_decile',	'cur_age',	'FRST_ISS_AGE',	'CITY',	'MTHLY_INCM', 'NO_DPND',	'f_HCM',	'f_HN',	'f_DN',	'f_oth_city',	'customer_segment',	'cus_age_band',	'dependent_age_band',	'cus_gender',	'f_adult_self_insured',	'f_family',	'f_family_wkids', 'f_empty_nest',	'f_undefined_segment',	'f_male',	'k_inf_cvg_acc',	'k_inf_cvg_ci',	'k_inf_cvg_inv', 'k_inf_cvg_lp', 'k_inf_cvg_lts', 'k_inf_cvg_med',	'f_addrchg_1m',	'f_addrchg_3m',	'f_addrchg_6m',	'f_addrchg_12m',	'f_occpchg_1m',	'f_occpchg_3m',	'f_occpchg_6m',	'f_occpchg_12m',	'activation_date',	'move_tenure_days', 'move_last_log_days', 'cws_tenure_days', 'cws_last_log_days', 'f_vip_elite', 'f_vip_plat', 'f_vip_gold', 'f_vip_silver', 'f_1st_term', 'f_1st_endow', 'f_1st_health_indem', 'f_1st_whole', 'f_1st_invest', 'f_2nd_term', 'f_2nd_endow', 'f_2nd_health_indem', 'f_2nd_whole', 'f_2nd_invest', 'yr_2nd_prod',
        'inforce_ind'
) #add other columns : premiums and ePOS

all_client_level.write.mode("overwrite").parquet('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/cseg_all')

# COMMAND ----------

#Read Saved Pyspark data
all_client_level = spark.read.format("parquet").load('abfss://lab@abcmfcadovnedl01psea.dfs.core.windows.net/vn/project/scratch/cseg_all', header = True)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Summary Stats : LTV</strong>

# COMMAND ----------

all_client_level.createOrReplaceTempView('all_client_level')

all_client_level_qtl = spark.sql("""select a.* , ntile(10) over (order by a.cli_ltv desc)  as decile from all_client_level a """)

all_client_level_qtl_summary = all_client_level_qtl.groupBy('decile')\
    .agg(F.count('po_num').alias('customer_count'),
    F.mean('cli_ltv').alias('mean_ltv'),
    F.mean('cli_ltv_pre').alias('mean_ltv_pre'),
    F.mean('cli_ltv_post').alias('mean_ltv_post'),
    F.mean('pol_count').alias('mean_pol'),
    F.mean('client_tenure').alias('mean_client_tenure'),
    F.mean('f_vip_elite').alias('mean_platinum_elite'),
    F.mean('f_vip_plat').alias('mean_platinum'),
    F.mean('f_vip_gold').alias('mean_gold'),
    F.mean('f_vip_silver').alias('mean_silver'),
    F.mean('cur_age').alias('mean_age'),
    F.mean('MTHLY_INCM').alias('mean_MTHLY_INCM'),
    F.mean('NO_DPND').alias('mean_NO_DPND'),
    F.mean('f_HCM').alias('mean_f_HCM'),
    F.mean('f_HN').alias('mean_f_HN'),
    F.mean('f_DN').alias('mean_f_DN'),
    F.mean('f_oth_city').alias('mean_f_oth_city'),
    F.mean('inforce_pol').alias('mean_inf_pol'),
    F.mean('lapsed_pol').alias('mean_lps_pol'),
    F.mean('surrendered_pol').alias('mean_srd_pol'),
    F.mean('matured_pol').alias('mean_mat_pol'),
    F.mean('nottaken_pol').alias('mean_ntk_pol'),
    F.mean('terminated_pol').alias('mean_ter_pol'),
    F.mean('lapse_decile').alias('mean_lapse_decile'),
    F.mean('ins_typ_count').alias('mean_ins_type'),
    F.mean('term_pol').alias('mean_term_pol'),
    F.mean('endow_pol').alias('mean_endow_pol'),
    F.mean('health_indem_pol').alias('mean_health_pol'),
    F.mean('whole_pol').alias('mean_whole_pol'),
    F.mean('investment_pol').alias('mean_inv_pol'),
    F.mean('f_1st_term').alias('mean_1st_term'),
    F.mean('f_1st_endow').alias('mean_1st_endow'),
    F.mean('f_1st_health_indem').alias('mean_1st_health_indem'),
    F.mean('f_1st_whole').alias('mean_1st_whole'),
    F.mean('f_1st_invest').alias('mean_1st_invest'),
    F.mean('f_2nd_term').alias('mean_2nd_term'),
    F.mean('f_2nd_endow').alias('mean_2nd_endow'),
    F.mean('f_2nd_health_indem').alias('mean_2nd_health_indem'),
    F.mean('f_2nd_whole').alias('mean_2nd_whole'),
    F.mean('f_2nd_invest').alias('mean_2nd_invest'),
    F.mean('yr_2nd_prod').alias('mean_yr_2nd_prod'),
    F.mean('agt_tenure_yrs',).alias('mean_agt_tenure'),
    F.mean('mdrt_flag',).alias('mdrt%'),
    F.mean('mdrt_tot_flag').alias('mdrt_tot%'),
    F.mean('mdrt_cot_flag').alias('mdrt_cot%'),
    F.mean('active_1m_flag').alias('1mA%'),
    F.mean('multi_prod').alias('multi_prod%'),
    F.mean('coverage_ape').alias('mean_ape'),
    F.mean('inforce_ind').alias('mean_inforce_ind')
    )
    
all_client_level_qtl_summary.limit(10).toPandas()


# COMMAND ----------

#Random Sample 
all_sample = all_client_level.sample(0.3, seed = 123)

# COMMAND ----------

# MAGIC %md
# MAGIC Preparation for K-Means Clustering

# COMMAND ----------

    client_df = all_sample.toPandas()

# COMMAND ----------

#client_df.set_index("po_num", inplace = True)
#client_df.head()
#client_df = client_df.reset_index()

#set index as po_num


# COMMAND ----------

# MAGIC %md
# MAGIC Select Columns

# COMMAND ----------

relevant_cols = ['po_num', 'client_tenure',	'pol_count', 'ins_typ_count', 'inforce_pol', 'lapsed_pol', 'surrendered_pol', 'matured_pol', 'nottaken_pol', 'terminated_pol', 'oth_paid_pol',
                 'term_pol', 'endow_pol', 'health_indem_pol', 'whole_pol', 'investment_pol', 'f_same_agent', 'f_cash', 'f_cheque', 'f_auto', 'agt_tenure_yrs', 'mdrt_flag', 'mdrt_tot_flag','mdrt_cot_flag', 'active_1m_flag',	'multi_prod',	'annual_flag',	'valid_email', 'valid_mobile', 'coverage_ape', 'plan_nbv', 'cli_nbv_margin', 'cli_ltv', 'cli_ltv_pre', 'cli_ltv_post', 'early_lapse_decile', 'next_due_lapse_decile', 'min_decile', 'cur_age', 'FRST_ISS_AGE', 'MTHLY_INCM', 'NO_DPND',	'f_HCM', 'f_HN',	'f_DN',	'f_oth_city', 'f_adult_self_insured',	'f_family',	'f_family_wkids',	'f_empty_nest',	'f_undefined_segment',	'f_male',	'k_inf_cvg_acc',	'k_inf_cvg_ci',	'k_inf_cvg_inv', 'k_inf_cvg_lp',	'k_inf_cvg_lts',	'k_inf_cvg_med',	'f_addrchg_1m',	'f_addrchg_3m',	'f_addrchg_6m',	'f_addrchg_12m',	'f_occpchg_1m',	'f_occpchg_3m',	'f_occpchg_6m',	'f_occpchg_12m',	'move_tenure_days',	'move_last_log_days','cws_tenure_days',	'cws_last_log_days', 'f_vip_elite', 'f_vip_plat', 'f_vip_gold', 'f_vip_silver', 'f_1st_term', 'f_1st_endow', 'f_1st_health_indem', 'f_1st_whole', 'f_1st_invest', 'f_2nd_term', 'f_2nd_endow', 'f_2nd_health_indem', 'f_2nd_whole', 'f_2nd_invest', 'yr_2nd_prod'                 
                 ]

client_df_cols = client_df[relevant_cols]
client_df_cols['agt_tenure_yrs'] = client_df_cols['agt_tenure_yrs'].fillna('99')
client_df_cols['min_decile'] = client_df_cols['min_decile'].fillna('99')
client_df_cols['early_lapse_decile'] = client_df_cols['early_lapse_decile'].fillna('99')
client_df_cols['next_due_lapse_decile'] = client_df_cols['next_due_lapse_decile'].fillna('99')
client_df_cols['yr_2nd_prod'] = client_df_cols['yr_2nd_prod'].fillna('99')

client_df_cols['MTHLY_INCM'] = client_df_cols['MTHLY_INCM'].astype(float)
client_df_cols['pol_count'] = client_df_cols['pol_count'].astype(float)
client_df_cols['ins_typ_count'] = client_df_cols['ins_typ_count'].astype(float)
client_df_cols['inforce_pol'] = client_df_cols['inforce_pol'].astype(float)
client_df_cols['lapsed_pol'] = client_df_cols['lapsed_pol'].astype(float)
client_df_cols['surrendered_pol'] = client_df_cols['surrendered_pol'].astype(float)
client_df_cols['matured_pol'] = client_df_cols['matured_pol'].astype(float)
client_df_cols['nottaken_pol'] = client_df_cols['nottaken_pol'].astype(float)
client_df_cols['terminated_pol'] = client_df_cols['terminated_pol'].astype(float)
client_df_cols['oth_paid_pol'] = client_df_cols['oth_paid_pol'].astype(float)

#client_df_na = client_df_cols.fillna(0)

# COMMAND ----------

client_df_cols.head(2)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

categorical_columns = list(client_df_cols.select_dtypes(include=["object", "int32","int64", "uint8", "category"]).columns)
numerical_columns = list(client_df_cols.select_dtypes(exclude=["object", "int32","int64", "uint8", "category"]).columns)


categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value = 0)),
        ]
    )


numerical_pipe = Pipeline(
                [   
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
                )


preprocessing_pipe = ColumnTransformer(
        [
            ("categorical", categorical_pipe, categorical_columns),
            ("numeric", numerical_pipe, numerical_columns),
        ],
        n_jobs=-1,
    )

features_list = categorical_columns + numerical_columns 

#scaler = StandardScaler()

preprocessing_pipe.fit(client_df_cols) #save scaler object

client_df_cols = preprocessing_pipe.transform(client_df_cols)
client_df_cols = pd.DataFrame(data=client_df_cols, columns=features_list)

client_df_cols = client_df_cols.set_index(client_df_cols['po_num'])
client_df_cols = client_df_cols.drop('po_num', axis='columns')

#Give more weight to cli_ltv
client_df_cols['cli_ltv']=client_df_cols['cli_ltv']*10


#Fill None as 0 -- in case imputer does not work (for checking)

client_df_na = client_df_cols.fillna(0)

client_df_na_dbscan = copy.deepcopy(client_df_na)
client_df_na_agg = copy.deepcopy(client_df_na)

# COMMAND ----------

# MAGIC %md
# MAGIC Save preprocessing Pipe

# COMMAND ----------


joblib.dump(preprocessing_pipe, '/dbfs/FileStore/tables/vn_cseg_preprocessing_pipe.model')
preprocessing_pipe = joblib.load('/dbfs/FileStore/tables/vn_cseg_preprocessing_pipe.model')


# COMMAND ----------

def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)

    
    return clusters_centers, k_values

# COMMAND ----------

def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# COMMAND ----------

clusters_centers, k_values = find_best_clusters(client_df_na, 10)

generate_elbow_plot(clusters_centers, k_values)

# COMMAND ----------

client_df_na.columns

# COMMAND ----------

kmeans_model = KMeans(n_clusters = 5,random_state = 123)

kmeans_clustering_model = kmeans_model.fit(client_df_na)

# COMMAND ----------

# MAGIC %md
# MAGIC Save KMeans Model

# COMMAND ----------

joblib.dump(kmeans_clustering_model, '/dbfs/FileStore/tables/kmeans_clustering_model.model')

kmeans_clustering_model = joblib.load('/dbfs/FileStore/tables/kmeans_clustering_model.model')

# COMMAND ----------



# COMMAND ----------

#client_df_na.head(2)

# COMMAND ----------

client_df_na["clusters"] = kmeans_clustering_model.labels_

client_df_na.head()

# COMMAND ----------

client_df_na.groupby(['clusters']).count()

#df.groupby(['revenue','session','user_id'])['user_id'].count()

# COMMAND ----------

#client_df.head(1)
#client_df_ft = client_df.set_index(client_df['po_num'])

# COMMAND ----------

#Apply MOdel to Full Data

#1 preprocess .transform
#2 kmeans_clustering_model.predict 

#Loop
#all_client_level.groupby('customer_segment').count().show(truncate=False)

#Use as loop
segment = ['Retired Self Insured' ,'Empty Nester' ,'Adult with Dependent' , 'Adult Self Insured' , 'Family', 'Family with Kids' , 'Undefined Segmentation']


from pyspark.sql import SparkSession
from pyspark.sql.types import *

emp_RDD = spark.sparkContext.emptyRDD()
 
# Create empty schema
columns = StructType([StructField("po_num", StringType(), True)
                                             ,StructField("clusters", IntegerType(), True)])
 
# Create an empty RDD with empty schema
data_emp = spark.createDataFrame(data = emp_RDD,
                             schema = columns)

for seg in segment: 
    data =all_client_level.filter(all_client_level.customer_segment ==seg)
    data =data.toPandas()

    data = data[relevant_cols]
    data['agt_tenure_yrs'] = data['agt_tenure_yrs'].fillna('99')
    data['min_decile'] = data['min_decile'].fillna('99')
    data['early_lapse_decile'] = data['early_lapse_decile'].fillna('99')
    data['next_due_lapse_decile'] = data['next_due_lapse_decile'].fillna('99')
    data['yr_2nd_prod'] = data['yr_2nd_prod'].fillna('99')

    data['MTHLY_INCM'] = data['MTHLY_INCM'].astype(float)
    data['pol_count'] = data['pol_count'].astype(float)
    data['ins_typ_count'] = data['ins_typ_count'].astype(float)
    data['inforce_pol'] = data['inforce_pol'].astype(float)
    data['lapsed_pol'] = data['lapsed_pol'].astype(float)
    data['surrendered_pol'] = data['surrendered_pol'].astype(float)
    data['matured_pol'] = data['matured_pol'].astype(float)
    data['nottaken_pol'] = data['nottaken_pol'].astype(float)
    data['terminated_pol'] = data['terminated_pol'].astype(float)
    data['oth_paid_pol'] = data['oth_paid_pol'].astype(float)

    data = preprocessing_pipe.transform(data)
    data = pd.DataFrame(data=data, columns=features_list)
    data['cli_ltv']=data['cli_ltv']*10
    data['cli_ltv_pre']=data['cli_ltv_pre']*10  #change this to whatever weight deemed fit
    data['cli_ltv_post']=data['cli_ltv_post']*10 #change this to whatever weight deemed fit

    data = data.fillna(0)
    data.set_index('po_num', inplace =True)
    data['clusters'] = kmeans_clustering_model.predict(data) #change this to other algorithm

    data = data.reset_index()
    data = data[['po_num', 'clusters']]

    data_spark = spark.createDataFrame(data)

    data_emp = data_emp.union(data_spark)


# COMMAND ----------

#len(data.columns)

# COMMAND ----------

#Join Score to Original Data

kmeans_data = all_client_level.join(data_emp, on ='po_num', how='left')
kmeans_data.limit(5).toPandas()

# COMMAND ----------

#Aggregate 
#mean cli_ltv
#mean pol
#mean tenure

kmeans_summary = kmeans_data.groupBy('clusters')\
    .agg(F.count('po_num').alias('customer_count'),
    F.mean('cli_ltv').alias('mean_ltv'),
    F.mean('cli_ltv_pre').alias('mean_ltv_pre'),
    F.mean('cli_ltv_post').alias('mean_ltv_post'),
    F.mean('pol_count').alias('mean_pol'),
    F.mean('client_tenure').alias('mean_client_tenure'),
    F.mean('f_vip_elite').alias('mean_platinum_elite'),
    F.mean('f_vip_plat').alias('mean_platinum'),
    F.mean('f_vip_gold').alias('mean_gold'),
    F.mean('f_vip_silver').alias('mean_silver'),
    F.mean('cur_age').alias('mean_age'),
    F.mean('MTHLY_INCM').alias('mean_MTHLY_INCM'),
    F.mean('NO_DPND').alias('mean_NO_DPND'),
    F.mean('f_HCM').alias('mean_f_HCM'),
    F.mean('f_HN').alias('mean_f_HN'),
    F.mean('f_DN').alias('mean_f_DN'),
    F.mean('f_oth_city').alias('mean_f_oth_city'),
    F.mean('inforce_pol').alias('mean_inf_pol'),
    F.mean('lapsed_pol').alias('mean_lps_pol'),
    F.mean('surrendered_pol').alias('mean_srd_pol'),
    F.mean('matured_pol').alias('mean_mat_pol'),
    F.mean('nottaken_pol').alias('mean_ntk_pol'),
    F.mean('terminated_pol').alias('mean_ter_pol'),
    F.mean('oth_paid_pol').alias('mean_oth_pol'),
    F.mean('ins_typ_count').alias('mean_ins_type'),
    F.mean('term_pol').alias('mean_term_pol'),
    F.mean('endow_pol').alias('mean_endow_pol'),
    F.mean('health_indem_pol').alias('mean_health_pol'),
    F.mean('whole_pol').alias('mean_whole_pol'),
    F.mean('investment_pol').alias('mean_inv_pol'),
    F.mean('f_1st_term').alias('mean_1st_term'),
    F.mean('f_1st_endow').alias('mean_1st_endow'),
    F.mean('f_1st_health_indem').alias('mean_1st_health_indem'),
    F.mean('f_1st_whole').alias('mean_1st_whole'),
    F.mean('f_1st_invest').alias('mean_1st_invest'),
    F.mean('f_2nd_term').alias('mean_2nd_term'),
    F.mean('f_2nd_endow').alias('mean_2nd_endow'),
    F.mean('f_2nd_health_indem').alias('mean_2nd_health_indem'),
    F.mean('f_2nd_whole').alias('mean_2nd_whole'),
    F.mean('f_2nd_invest').alias('mean_2nd_invest'),
    F.mean('yr_2nd_prod').alias('mean_yr_2nd_prod'),
    F.mean('agt_tenure_yrs',).alias('mean_agt_tenure'),
    F.mean('mdrt_flag',).alias('mdrt%'),
    F.mean('mdrt_tot_flag').alias('mdrt_tot%'),
    F.mean('mdrt_cot_flag').alias('mdrt_cot%'),
    F.mean('active_1m_flag').alias('1mA%'),
    F.mean('multi_prod').alias('multi_prod%'),
    F.mean('coverage_ape').alias('mean_ape')
    )

kmeans_summary.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC DBSCAN

# COMMAND ----------

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Quartiles

# COMMAND ----------

all_client_level.createOrReplaceTempView('all_client_level')

all_client_level_qtl = spark.sql("""select a.* , ntile(5) over (order by a.cli_ltv desc)  as quartile from all_client_level a """)

# COMMAND ----------

all_client_level_qtl.groupBy('quartile')\
    .agg(F.count('po_num').alias('customer_count'),
    F.mean('cli_ltv').alias('mean_ltv'),
    F.mean('cli_ltv_pre').alias('mean_ltv_pre'),
    F.mean('cli_ltv_post').alias('mean_ltv_post'),
    F.mean('pol_count').alias('mean_pol'),
    F.mean('client_tenure').alias('mean_client_tenure'),
    F.mean('f_vip_elite').alias('mean_platinum_elite'),
    F.mean('f_vip_plat').alias('mean_platinum'),
    F.mean('f_vip_gold').alias('mean_gold'),
    F.mean('f_vip_silver').alias('mean_silver'),
    F.mean('cur_age').alias('mean_age'),
    F.mean('MTHLY_INCM').alias('mean_MTHLY_INCM'),
    F.mean('NO_DPND').alias('mean_NO_DPND'),
    F.mean('f_HCM').alias('mean_f_HCM'),
    F.mean('f_HN').alias('mean_f_HN'),
    F.mean('f_DN').alias('mean_f_DN'),
    F.mean('f_oth_city').alias('mean_f_oth_city'),
    F.mean('inforce_pol').alias('mean_inf_pol'),
    F.mean('lapsed_pol').alias('mean_lps_pol'),
    F.mean('surrendered_pol').alias('mean_srd_pol'),
    F.mean('matured_pol').alias('mean_mat_pol'),
    F.mean('nottaken_pol').alias('mean_ntk_pol'),
    F.mean('terminated_pol').alias('mean_ter_pol'),
    F.mean('oth_paid_pol').alias('mean_oth_pol'),
    F.mean('ins_typ_count').alias('mean_ins_type'),
    F.mean('term_pol').alias('mean_term_pol'),
    F.mean('endow_pol').alias('mean_endow_pol'),
    F.mean('health_indem_pol').alias('mean_health_pol'),
    F.mean('whole_pol').alias('mean_whole_pol'),
    F.mean('investment_pol').alias('mean_inv_pol'),
    F.mean('agt_tenure_yrs',).alias('mean_agt_tenure'),
    F.mean('mdrt_flag',).alias('mdrt%'),
    F.mean('mdrt_tot_flag').alias('mdrt_tot%'),
    F.mean('mdrt_cot_flag').alias('mdrt_cot%'),
    F.mean('active_1m_flag').alias('1mA%'),
    F.mean('multi_prod').alias('multi_prod%'),
    F.mean('coverage_ape').alias('mean_ape'))limit(100).toPandas()


# COMMAND ----------



# COMMAND ----------

relevant_cols

# COMMAND ----------



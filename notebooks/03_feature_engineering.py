#!/usr/bin/env python
# coding: utf-8

# # 03_feature_engineering.ipynb
# 
# # Feature Engineering  
# Home Credit Default Risk Competition – building and testing candidate features  
# 
# ---
# 
# 
# ## 1. Setup & Data Loading
# 
# Import libraries, set plotting options, and load the four key tables.  
# All paths assume this notebook lives in `notebooks/` and `data/` is one level up.
# 

# In[ ]:


# Path
import sys, os
proj_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Standard imports
import pandas as pd
import numpy as np

# Load data
app = pd.read_csv(os.path.join(proj_root, 'data', 'application_train.csv'))
inst = pd.read_csv(os.path.join(proj_root, 'data', 'installments_payments.csv'))
bureau = pd.read_csv(os.path.join(proj_root, 'data', 'bureau.csv'))
prev = pd.read_csv(os.path.join(proj_root, 'data', 'previous_application.csv'))
bureau_balance = pd.read_csv(os.path.join(proj_root, 'data', 'bureau_balance.csv'))
pos_cash = pd.read_csv(os.path.join(proj_root, 'data', 'POS_CASH_balance.csv'))
cc_balance = pd.read_csv(os.path.join(proj_root, 'data', 'credit_card_balance.csv'))


# ## 2. Preprocessing
# 
# Clean obvious anomalies before feature creation.

# In[15]:


# Treat the sentinel in DAYS_EMPLOYED
app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].replace(365243, np.nan)

# Fill selected missing columns with median
fill_cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
             'FLOORSMAX_MEDI','YEARS_EMPLOYED']  # YEARS_EMPLOYED created below
app['YEARS_EMPLOYED'] = -app['DAYS_EMPLOYED'] / 365
app[fill_cols] = app[fill_cols].fillna(app[fill_cols].median())


# ## 3. Basic Person-Level Features
# 
# Simple transforms: age, employment years, log‐transforms, and category groupings.
# 

# In[16]:


# Age in years
app['AGE'] = (-app['DAYS_BIRTH'] / 365).clip(lower=0)

# Years employed
app['YEARS_EMPLOYED'] = (-app['DAYS_EMPLOYED'] / 365).clip(lower=0)

# Log transforms for skewed money amounts
app['LOG_INCOME'] = np.log1p(app['AMT_INCOME_TOTAL'])
app['LOG_CREDIT'] = np.log1p(app['AMT_CREDIT'])
app['LOG_GOODS_PRICE'] = np.log1p(app['AMT_GOODS_PRICE'])


# ## 4. Ratio & Interaction Features
# 
# Combine basic columns to capture relationships (income vs credit, etc.).

# In[17]:


# Annuity‐to‐income ratio
app['ANNUITY_INCOME_RATIO'] = app['AMT_ANNUITY'] / (app['AMT_INCOME_TOTAL'] + 1)

# Credit minus goods price
app['CREDIT_GOODS_DIFF'] = app['AMT_CREDIT'] - app['AMT_GOODS_PRICE']

# Credit‐to‐annuity ratio
app['CREDIT_ANNUITY_RATIO'] = app['AMT_CREDIT'] / (app['AMT_ANNUITY'] + 1)

# Age * employment years
app['AGE_EMPLOYED_PRODUCT'] = app['AGE'] * app['YEARS_EMPLOYED']


# ## 5. External Score Features
# 
# Leverage the three external scores and their interactions.
# 

# In[18]:


ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

# Mean of external scores
app['EXT_SOURCE_MEAN'] = app[ext].mean(axis=1)

# Interaction: log‐published‐ID age × external mean
app['ID_PUBLISH_SCORE_INTERACTION'] = (
    np.log1p(-app['DAYS_ID_PUBLISH'].abs()) * app['EXT_SOURCE_MEAN']
)

# Credit‐stress: requests per day de‐weighted by score
app['CREDIT_STRESS_SCORE'] = (
    app['AMT_REQ_CREDIT_BUREAU_DAY'] * (1 - app['EXT_SOURCE_MEAN'])
)


# ## 6. Housing & Document Features
# 
# Combine housing info and count submitted documents.

# In[19]:


# Building age × floors
app['BUILDING_AGE_SCORE'] = (
    app['YEARS_BEGINEXPLUATATION_MEDI'] * app['FLOORSMAX_MEDI']
)

# Count of submitted documents
doc_cols = [c for c in app.columns if c.startswith('FLAG_DOCUMENT_')]
app['DOCUMENT_COUNT'] = app[doc_cols].sum(axis=1)


# ## 7. Socio-Demographic Grouping
# 
# Simplify and combine education, income type, and family status into a single category.
# 

# In[20]:


# Simplify levels
app['EDU_SIMPLE'] = app['NAME_EDUCATION_TYPE'].replace({
    'Secondary / secondary special':'Secondary',
    'Higher education':'Higher','Incomplete higher':'Higher',
    'Lower secondary':'Secondary','Academic degree':'Higher'
})
app['INC_SIMPLE'] = app['NAME_INCOME_TYPE'].replace({
    'Working':'Employed','State servant':'Employed',
    'Commercial associate':'Self-employed','Business Entity Type 3':'Self-employed',
    'Pensioner':'Retired','Student':'Student','Unemployed':'Unemployed'
})
app['FAM_SIMPLE'] = app['NAME_FAMILY_STATUS'].replace({
    'Married':'Married','Civil marriage':'Single','Single / not married':'Single',
    'Widow':'Single','Separated':'Single','Unknown':'Unknown'
})

# Combined socio-group
app['SOCIO_GROUP'] = (
    app['EDU_SIMPLE'] + '_' + app['INC_SIMPLE'] + '_' + app['FAM_SIMPLE']
).astype('category')


# ## 8. Installments-Payments Features
# 
# From `installments_payments.csv`, compute lateness, delay, and payment ratios, then aggregate per client.

# In[21]:


# Create raw columns
inst['PAYMENT_LATE'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
inst['PAYMENT_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['DELAY_RATIO'] = inst['PAYMENT_DELAY'] / (inst['DAYS_INSTALMENT'].abs() + 1)
inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1e-5)
inst['ABS_PAY_DIFF'] = (inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']).abs()

# additional raw features
inst['PAYMENT_EARLY'] = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)
inst['IS_UNDERPAYMENT'] = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype(int)
inst['IS_OVERPAYMENT'] = (inst['AMT_PAYMENT'] > inst['AMT_INSTALMENT']).astype(int)
inst['MISSED_PAYMENT'] = (inst['AMT_PAYMENT'] == 0).astype(int)
inst['PAID_NOTHING_BUT_SHOULD'] = inst['MISSED_PAYMENT']
inst['NUM_PREVIOUS_LOANS'] = inst['SK_ID_PREV']  # will count unique later

# Aggregate by customer
agg_inst = inst.groupby('SK_ID_CURR').agg({
    'PAYMENT_DELAY': ['max'],
    'PAYMENT_LATE': ['mean'],
    'PAYMENT_EARLY': ['mean','max'],
    'DELAY_RATIO': ['max'],
    'PAYMENT_RATIO': ['mean','max'],
    'ABS_PAY_DIFF': ['mean'],
    'IS_UNDERPAYMENT': ['mean'],
    'IS_OVERPAYMENT': ['mean'],
    'MISSED_PAYMENT': ['sum'],
    'PAID_NOTHING_BUT_SHOULD': ['sum'],
    'SK_ID_PREV': pd.Series.nunique
})

# flatten columns
agg_inst.columns = [
    'inst_' + 
    ( 'NUM_PREV_LOANS_NUNIQUE' if c[0]=='SK_ID_PREV' 
      else '_'.join(c).upper() )
    for c in agg_inst.columns
]
app = app.merge(agg_inst, how='left', on='SK_ID_CURR')


# ## 9. Bureau Features
# 
# From `bureau.csv`, fill missing, create ratios and flags, then aggregate per client.

# In[22]:


# Preprocess nulls
for col in ['AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE',
            'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM']:
    bureau[col].fillna(0, inplace=True)
bureau['DAYS_CREDIT_ENDDATE'].fillna(bureau['DAYS_CREDIT_ENDDATE'].median(), inplace=True)
bureau['DAYS_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'].notna().astype(int)
bureau['IS_ACTIVE'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
bureau['CREDIT_COUNT'] = 1

# Feature creation
bureau['DEBT_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1e-5)
bureau['DEBT_LIMIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM_LIMIT'] + 1e-5)
bureau['CREDIT_DURATION'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
bureau['MAX_OVERDUE_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / (bureau['AMT_CREDIT_SUM'] + 1e-5)
bureau['PROLONG_RATIO'] = bureau['CNT_CREDIT_PROLONG'] / (bureau['CREDIT_COUNT'] + 1e-5)

# aggregate by customer
agg_bur = bureau.groupby('SK_ID_CURR').agg({
    'CREDIT_COUNT':'sum',
    'IS_ACTIVE':'mean',
    'DEBT_CREDIT_RATIO':['mean','max'],
    'DEBT_LIMIT_RATIO':['mean'],
    'CREDIT_DURATION':['mean','max'],
    'MAX_OVERDUE_RATIO':['mean'],
    'PROLONG_RATIO':['mean']
})
agg_bur.columns = ['bur_' + '_'.join(col).upper() for col in agg_bur.columns]
app = app.merge(agg_bur, how='left', on='SK_ID_CURR')


# ## 10. Previous Application Features
# 
# From `previous_application.csv`, impute, derive per‐row features, then aggregate per client.
# 

# In[23]:


# Preprocess nulls
for c in ['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']:
    prev[c].fillna(0, inplace=True)
prev['DAYS_TERMINATION'].fillna(prev['DAYS_TERMINATION'].median(), inplace=True)

# Row‐level features
prev['APP_AGE'] = prev['DAYS_DECISION'].abs()
prev['APPROVAL_RATIO'] = prev['AMT_CREDIT'] / (prev['AMT_APPLICATION'] + 1e-5)
prev['CREDIT_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / (prev['AMT_ANNUITY'] + 1e-5)
prev['GOODS_CREDIT_DIFF'] = prev['AMT_CREDIT'] - prev['AMT_GOODS_PRICE']
prev['CREDIT_DURATION'] = prev['DAYS_TERMINATION'] - prev['DAYS_DECISION']
prev['WAS_APPROVED'] = (prev['NAME_CONTRACT_STATUS']=='Approved').astype(int)

# Aggregate by customer
agg_prev = prev.groupby('SK_ID_CURR').agg({
    'APP_AGE':['mean','max'],
    'APPROVAL_RATIO':['mean'],
    'CREDIT_ANNUITY_RATIO':['mean'],
    'GOODS_CREDIT_DIFF':['mean'],
    'CREDIT_DURATION':['mean'],
    'WAS_APPROVED':['mean','sum']
})
agg_prev.columns = ['prev_' + '_'.join(col).upper() for col in agg_prev.columns]
app = app.merge(agg_prev, how='left', on='SK_ID_CURR')


# ## 11. Bureau Balance Features
# 
# From `bureau_balance.csv`, join to `bureau` on `SK_ID_BUREAU` to get `SK_ID_CURR`, map status codes to numeric delinquency levels, create a delinquency indicator, then aggregate per customer.

# In[24]:


# 11. Bureau Balance Features
bb = bureau_balance.merge(
    bureau[['SK_ID_BUREAU','SK_ID_CURR']],
    on='SK_ID_BUREAU',
    how='left'
)

# Map STATUS to numeric levels (0 = closed/unknown/no DPD, 1–5 = severity of DPD)
status_map = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
bb['STATUS_NUM'] = bb['STATUS'].map(status_map).fillna(0).astype(int)
bb['DELINQ']     = bb['STATUS'].isin(['1','2','3','4','5']).astype(int)

# Aggregate features by customer
agg_bb = bb.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE': ['min','max','count'],
    'STATUS_NUM':     ['mean','max'],
    'DELINQ':         ['mean']
})

# Rename columns
agg_bb.columns = ['bb_' + '_'.join(col).upper() for col in agg_bb.columns]

# Merge into main dataframe
app = app.merge(agg_bb, how='left', on='SK_ID_CURR')


# ## 12. POS Cash Balance Features
# 
# From `POS_CASH_balance.csv`, derive indicators for delinquency and active status, then compute summary statistics per customer.
# 

# In[25]:


# 12. POS Cash Balance Features
pc = pos_cash.copy()

# Create delinquency and active indicators
pc['DELINQ']      = (pc['SK_DPD'] > 0).astype(int)
pc['DELINQ_DEF']  = (pc['SK_DPD_DEF'] > 0).astype(int)
pc['ACTIVE']      = (pc['NAME_CONTRACT_STATUS'] == 'Active').astype(int)

# Aggregate features by customer
agg_pc = pc.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE':       ['min','max','count'],
    'CNT_INSTALMENT':       ['mean'],
    'CNT_INSTALMENT_FUTURE':['mean'],
    'DELINQ':               ['mean','max'],
    'DELINQ_DEF':           ['mean','max'],
    'ACTIVE':               ['mean']
})

# Rename columns
agg_pc.columns = ['pc_' + '_'.join(col).upper() for col in agg_pc.columns]

# Merge into main dataframe
app = app.merge(agg_pc, how='left', on='SK_ID_CURR')


# ## 13. Credit Card Balance Features
# 
# From `credit_card_balance.csv`, compute ratio features for balances, withdrawals and payments, create delinquency indicators, then aggregate per customer.
# 

# In[ ]:


# 13. Credit Card Balance Features
ccb = cc_balance.copy()

# Ratio features
ccb['LIMIT_BAL_RATIO']   = ccb['AMT_BALANCE'] / (ccb['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
ccb['DRAWINGS_RATIO']    = ccb['AMT_DRAWINGS_CURRENT'] / (ccb['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
ccb['PAYMENT_BAL_RATIO'] = ccb['AMT_PAYMENT_CURRENT'] / (ccb['AMT_BALANCE'] + 1e-5)
ccb['RECV_DIFF_RATIO']   = (
    (ccb['AMT_TOTAL_RECEIVABLE'] - ccb['AMT_RECEIVABLE_PRINCIPAL'])
    / (ccb['AMT_TOTAL_RECEIVABLE'] + 1e-5)
)

# Create delinquency indicators
ccb['DELINQ']     = (ccb['SK_DPD'] > 0).astype(int)
ccb['DELINQ_DEF'] = (ccb['SK_DPD_DEF'] > 0).astype(int)

# Aggregate features by customer
agg_ccb = ccb.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE':    ['min','max','count'],
    'LIMIT_BAL_RATIO':   ['mean','max'],
    'DRAWINGS_RATIO':    ['mean'],
    'PAYMENT_BAL_RATIO': ['mean'],
    'RECV_DIFF_RATIO':   ['mean'],
    'DELINQ':            ['mean','max'],
    'DELINQ_DEF':        ['mean','max']
})

# Rename columns
agg_ccb.columns = ['ccb_' + '_'.join(col).upper() for col in agg_ccb.columns]

# Merge into main dataframe
app = app.merge(agg_ccb, how='left', on='SK_ID_CURR')


# ## 14. Feature Evaluation (Quick Sanity Check)
# 
# In this step we select **only numeric** features (excluding identifiers and the target), then compute their absolute Pearson correlation with `TARGET`.  
# This gives a first indication of which engineered features are most promising and should be prioritized in the subsequent feature selection phase.

# In[27]:


# pick only the numeric columns
numeric_feats = (
    app
    .select_dtypes(include=['number'])
    .columns
    .drop(['SK_ID_CURR', 'TARGET'], errors='ignore')
)

# compute absolute Pearson correlation with TARGET
corrs = (
    app[numeric_feats.tolist() + ['TARGET']]
    .corr()['TARGET']
    .abs()
    .sort_values(ascending=False)
)

# show top 15
print(corrs.head(15))


# ## 15. Export Feature Matrix
# 
# Save the full set of engineered features for the selection notebook.

# In[ ]:


# List of all feature columns
feature_cols = [c for c in app.columns if c not in ['SK_ID_CURR','TARGET']]
pd.Series(feature_cols).to_csv(os.path.join(proj_root,'outputs','00_feature_names_all.csv'), index=False)

# Export full DataFrame
app[['SK_ID_CURR'] + feature_cols + ['TARGET']] \
   .to_csv(os.path.join(proj_root,'outputs','00_train_features_all.csv'), index=False)

print("Exported", len(feature_cols), "features.")


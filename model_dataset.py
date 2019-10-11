# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 23:54:20 2019

@author: Luis Rodriguez
"""

# Model Dataset

import pandas as pd
import numpy as np


# Current Accounts Percentage

def wd(w):
  if w['worst_delinquency'] == 0:
      return 0
    
  else:
      return 1

db['delinquency_accounts'] = db.apply(wd, axis=1)
id_debt['delinquency_accounts']=db.groupby('id').agg({'delinquency_accounts':'sum'})

id_debt['current_accounts']=id_debt['total_credits']-id_debt['delinquency_accounts']
id_debt['current_accounts_percentage']=id_debt['current_accounts']/id_debt['total_credits']*100   
id_debt['delinquency_accounts_percentage']=id_debt['delinquency_accounts']/id_debt['total_credits']*100   

# Worst delinquency & worst delinquency past due balance

id_debt['worst_delinquency']=db.groupby('id').agg({'worst_delinquency':'max'})
id_debt['worst_delinquency_past_due_balance']=db.groupby('id').agg({'worst_delinquency_past_due_balance':'max'})

# Maximim credit limit

id_debt['credit_limit']=db.groupby('id').agg({'credit_limit':'max'})

# Maximum credit payments (in months)

db['maximum_number_of_payments']=db['total_credit_payments']/db['payments_per_month']
id_debt['maximum_number_of_credit_payments']=db.groupby('id').agg({'maximum_number_of_payments':'max'})

# Model Dataset

md=id_debt

md.to_csv('model_dataset.csv',index=False)






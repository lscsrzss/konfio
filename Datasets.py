# -*- coding: utf-8 -*-
"""
Created on Wednesday Oct 9 19:50:28 2019

@author: Luis Rodriguez
"""

import pandas as pd
import numpy as np

# DF's creation
reports=pd.read_csv('credit_reports.csv')
users=pd.read_csv('users.csv')

# DF's general info 
print(reports.info(),'\n'*2)
print(users.info(),'\n'*2)
print(reports.describe(),'\n'*2)
print(users.describe(),'\n'*2)

# DF's heads
print(reports.head(10),'\n'*2)
print(users.head(10),'\n'*2)

# Users Class Mean

print('El',users['class'].mean() * 100,'% de los clientes son catalogados como buenos clientes.','\n')

# Income - Outcome

users['monthly_difference']=users['monthly_income']-users['monthly_outcome']
users=users.reindex(['id','monthly_income','monthly_outcome','monthly_difference','class'], axis=1) 


# Merge dataframes

reports=reports.rename(columns={'user_id':'id'})
db = pd.merge(reports,users, on='id')

# Amount to payt next payment

payments_per_month = {
    'Semanal':4,
    'Quincenal':2,
    'Catorcenal':2,
    'Mensual':1,
    'Una sola exhibición':1, 
    'Pago mínimo para cuentas revolventes':1,
    'Bimestral':1/2, 
    'Trimestral':1/3,
    'Anual':1/12, 
    'Deducción del salario':0,
    np.nan:0
}

db['payments_per_month']=db['payment_frequency'].apply(lambda x: payments_per_month[x])
db['real_monthly_payment']=db['amount_to_pay_next_payment'] * db['payments_per_month']
db['loan_term_months']=db['total_credit_payments']/db['payments_per_month']


# Separating Good and Bad Clients

good_clients=db.loc[db['class'] == 1]
bad_clients=db.loc[db['class'] == 0]

# monthly_difference - real_monthly_payment
'''Group by monthly payment per client, monthly difference, etc..'''
id_debt=pd.DataFrame()
id_debt=db.groupby('id').agg({'real_monthly_payment':'sum'}).reset_index()
id_debt['monthly_difference']=db.groupby('id').agg({'monthly_difference':'sum'})
id_debt['difference_surplus']=id_debt['monthly_difference']-id_debt['real_monthly_payment']
id_debt['class']=users['class']

id_debt_good_clients=id_debt.loc[id_debt['class'] == 1]
id_debt_bad_clients=id_debt.loc[id_debt['class'] == 0]



# Good & Bad Clients Pie chart

import matplotlib.pyplot as plt

labels='Buenos Clientes','Malos Clientes'
sizes=[53.5*1000, (100-53.5)*1000]
explode=(.05,.05)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()

# Hypothesis testing

good_clients['current_balance_minus_credit_limit']=good_clients['current_balance']-good_clients['credit_limit']
bad_clients['current_balance_minus_credit_limit']=bad_clients['current_balance']-bad_clients['credit_limit']

print('En el caso de Buenos Clientes, el promedio de la Diferencia de Ingresos y Egresos es', id_debt_good_clients['monthly_difference'].mean(),'\n'  )
print('En el caso de Malos Clientes, el promedio de la Diferencia de Ingresos y Egresos es', id_debt_bad_clients['monthly_difference'].mean(),'\n'  )
print( 'En el caso de Buenos Clientes, el promedio de la Diferencia de Ingresos y Egresos menos el Pago Real Mesual de cada cliente es', id_debt_good_clients['difference_surplus'].mean(),'\n'  )
print( 'En el caso de Malos Clientes, el promedio de la Diferencia de Ingresos y Egresos menos el Pago Real Mesual de cada cliente es', id_debt_bad_clients['difference_surplus'].mean(),'\n' )
print('En el caso de Buenos Clientes el promedio de su Balance menos su Límite de Crédito es',good_clients['current_balance_minus_credit_limit'].mean(),'\n')
print('En el caso de Malos Clientes el promedio de su Balance menos su Límite de Crédito es',bad_clients['current_balance_minus_credit_limit'].mean(),'\n')
print('En el caso de Buenos Clientes, se tienen', good_clients['number_of_payments_due'].sum(), 'de payments due.','\n')
print('En el caso de Malos Clientes, se tienen', bad_clients['number_of_payments_due'].sum(), 'de payments due.','\n')
print('En el caso de Buenos Clientes, se tienen', good_clients['worst_delinquency'].sum(), 'de worst delinquency.','\n')
print('En el caso de Malos Clientes, se tienen', bad_clients['worst_delinquency'].sum(), 'de worst delinquency.','\n')

# Total Credits per Client

id_debt['total_credits']=db.groupby('id').size()
id_debt_good_clients['total_credits']=good_clients.groupby('id').size()
id_debt_bad_clients['total_credits']=bad_clients.groupby('id').size()

# Ploting in order to see a tendence 

ax=id_debt_good_clients.plot('id','total_credits',kind='scatter',color='red',label='Good Clients',marker='+')
id_debt_bad_clients.plot('id','total_credits',kind='scatter',color='darkblue',ax=ax,label='Bad Clients',marker='^')
plt.show()
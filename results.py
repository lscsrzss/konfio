# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:44:50 2019

@author: Luis Rodriguez
"""

# Selecting Best Clients, Loan Ammount, Loan Term and Anual Interest.

import pandas as pd

data_set=pd.read_csv('model_dataset_vf.csv')

# Best clients will be thw ones who are in Class=1 and the Predicted Class = 1
# And their income minus outcome minus monthly real payment is greater than 0.


data_set['class_prediction']=logistic_regresion_model.predict(X)


data_pred=data_set.loc[data_set['class'] == 1] 
best_clients=data_pred.loc[data_pred['class_prediction'] == 1]


best_clients['free_income']=best_clients['difference_surplus']-best_clients['real_monthly_payment']
best_clients=best_clients.loc[best_clients['free_income']>0]
best_clients=best_clients.loc[best_clients['maximum_number_of_credit_payments']>0]

print('159 clients were classified as Best Clients')   

# Loan ammount
# In business management, the recomended ammount for a company in a regular 
# situation, is to use 50% of their surplus tops, to pay their credits.
# Considering this situation, and encouraging our clients business health,
# our recomende amounts should be considered as 50% of thei free income per month.

best_clients['loan_ammount_monthly_payment']=best_clients['free_income']/2

# Loan term
# Based on the clients maximum credit term in months, offering them an increase
# of 10% in maximum credit terms.

best_clients['recomended_loan_term']=best_clients['maximum_number_of_credit_payments']*1.1

# Credit limit

best_clients['credit_limit']=best_clients['loan_ammount_monthly_payment']*best_clients['recomended_loan_term']
#best_clients['credit_limit']=best_clients.replace([best_clients['credit_limit']>best_clients['credit_limit']*2,best_clients['credit_limit']*2])

# Anual interest to make the loan profitable
# According to CONDUSEF, the anual interest rate is between 40% and up to 78%
# for PYMES, so, we should be in that range, or even offer a better rate to 
# help and attract more clients, also considering, that the automation Konfio 
# develops, let them have low operational costs, so a lower rate is possible.

# Considering a 40% rate (3.33% monthly rate)

best_clients['loan_term_in_years']=best_clients['recomended_loan_term']/12
best_clients['minimum_interest_loan_40%']=best_clients['loan_term_in_years']*best_clients['loan_ammount_monthly_payment']*.333

# Considering the 78% rate (6.5% monthly rate)
best_clients['maximum_interest_loan_78%']=best_clients['loan_term_in_years']*best_clients['loan_ammount_monthly_payment']*.65

# To define the interest, we should know more information about the Konfio's 
# operational costs, and probably consider other factors as inflation, to set
# an adequate rate for each client.

results = best_clients[['id', 'free_income','loan_ammount_monthly_payment',
                        'recomended_loan_term','loan_term_in_years','credit_limit',
                        'minimum_interest_loan_40%','maximum_interest_loan_78%',
                        ]].copy()

#results['credit_limit']=results.replace([results['credit_limit']>(results['credit_limit']*2),results['credit_limit']*2])


results.to_csv('results.csv',index=False)


import matplotlib.pyplot as plt

plt.title('')
plt.plot(results['id'],results['free_income'],label='Free Income')
plt.plot(results['id'],results['loan_ammount_monthly_payment'],label='Loan Monthly Payment')
plt.plot(results['id'],results['credit_limit'],label='Credit Limit')
plt.plot(results['id'],results['minimum_interest_loan_40%'],label='Min Interest Loan')
plt.plot(results['id'],results['maximum_interest_loan_78%'],label='Max Interest Loan')
plt.legend()
plt.xlabel('Id')
plt.ylabel('$')
plt.show()














# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 10:59:56 2015

@author: dleonard
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 16:36:26 2015

@author: dleonard
"""
#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, math

""" Read in training data csv file """

df = pd.read_csv('C:/Users/dleonard/Documents/Kaggle/Taxi/taxi_train.csv', index_col=[0])


""" Data munging """

# Create time variable: The total travel time of the trip (the prediction target)
df['TIME'] = ((((df['POLYLINE'].str.count(',') + 1) / 2 ) - 1) * 15).clip(0)

# Create variables based on timestamp
sec_from_jul = (df['TIMESTAMP'] % (60 * 60 * 24 * 15887))
df['START_HOUR'] = np.fix((sec_from_jul / (60 * 60)) % 24)
df['START_DAY_OF_WEEK'] = np.fix((sec_from_jul / (60 * 60 * 24)) / 7 % 7) + 1

# Caclulate median trip times grouped by feature categories
df['TIME_BY_HOUR'] = df['START_HOUR'].map(df['TIME'].groupby(df['START_HOUR']).median())
df['TIME_BY_DAY'] = df['START_DAY_OF_WEEK'].map(df['TIME'].groupby(df['START_DAY_OF_WEEK']).median())
df['TIME_BY_CALL_TYPE'] = df['CALL_TYPE'].map(df['TIME'].groupby(df['CALL_TYPE']).median())
df['TIME_BY_STAND'] = df['ORIGIN_STAND'].map(df['TIME'].groupby(df['ORIGIN_STAND']).median())
df['TIME_BY_STAND'] = df['TIME_BY_STAND'].fillna(df['TIME'].median())


""" Create charts """

df['START_HOUR'].hist(bins=24)
df['START_DAY_OF_WEEK'].hist(bins=7)

plt.scatter(df['START_HOUR'], df['TIME'])
plt.scatter(df['START_DAY_OF_WEEK'], df['TIME'])

df['TIME'].hist(bins=100, range = (0, 5000))
df[df['CALL_TYPE'] == 'A']['TIME'].hist(bins=100, range = (0, 3000))
df[df['CALL_TYPE'] == 'B']['TIME'].hist(bins=100, range = (0, 3000))
df[df['CALL_TYPE'] == 'C']['TIME'].hist(bins=100, range = (0, 3000))


""" Summary statistics """

start_hour_grouped = df['TIME'].groupby(df['START_HOUR'])
pd.concat([start_hour_grouped.median(), start_hour_grouped.std(), start_hour_grouped.count() / len(df)], axis=1)


""" RMSLE on training data """

RMSLE = math.sqrt((1.0 / len(df['TIME'])) * sum((np.log(df['TIME'] + 1.0) - np.log(df['TIME'].median() + 1.0))**2.0))
RMSLE_call_type = math.sqrt((1.0 / len(df['TIME'])) * sum((np.log(df['TIME'] + 1.0) - np.log(df['TIME_BY_CALL_TYPE'] + 1.0))**2.0))
RMSLE_origin_stand = math.sqrt((1.0 / len(df['TIME'])) * sum((np.log(df['TIME'] + 1.0) - np.log(df['TIME_BY_STAND'] + 1.0))**2.0))
RMSLE_start_hour = math.sqrt((1.0 / len(df['TIME'])) * sum((np.log(df['TIME'] + 1.0) - np.log(df['TIME_BY_HOUR'] + 1.0))**2.0))
RMSLE_start_day_of_week = math.sqrt((1.0 / len(df['TIME'])) * sum((np.log(df['TIME'] + 1.0) - np.log(df['TIME_BY_DAY'] + 1.0))**2.0))
print 'RMSLE Avg time: %f' % RMSLE
print 'RMSLE Avg time by call type: %f' % RMSLE_call_type
print 'RMSLE Avg time by origin stand: %f' % RMSLE_origin_stand
print 'RMSLE Avg time by start hour: %f' % RMSLE_start_hour
print 'RMSLE Avg time by start day of week: %f' % RMSLE_start_day_of_week


""" Read in test data csv file """

df_test = pd.read_csv('C:/Users/dleonard/Documents/Kaggle/Taxi/taxi_test.csv', index_col=[0])
df_test['TIME'] = ((((df_test['POLYLINE'].str.count(',') + 1) / 2 ) - 1) * 15).clip(0)
sec_from_jul = (df_test['TIMESTAMP'] % (60 * 60 * 24 * 15887))
df_test['START_HOUR'] = np.fix((sec_from_jul / (60 * 60)) % 24)

""" Survival analysis on median trip times grouped by start hour """

survivor_averages = []
for j in range(0, 23):
    for i in range(0, 2100, 10):
        survivor_averages.append([j, i, df[(df['START_HOUR'] == j) & (df['TIME'] > i)]['TIME'].median()])
survivor_averages_df = pd.DataFrame(survivor_averages, columns = ['START_HOUR', 'TIME', 'MEDIAN'])

hour_survival = []

for index, row in df_test.iterrows():
    for e in range(0, len(survivor_averages_df['TIME'])):
        if row['TIME'] >= 2000:
            hour_survival.append(row['TIME'])
            break
        elif row['START_HOUR'] != survivor_averages_df['START_HOUR'][e]:
            continue
        elif row['TIME'] >= survivor_averages_df['TIME'][e]:
            continue
        else:
            hour_survival.append(survivor_averages_df['MEDIAN'][e])
            break

df_test['TRAVEL_TIME'] = hour_survival
submission_survival = pd.DataFrame(df_test['TRAVEL_TIME'])
submission_survival.to_csv('C:/Users/dleonard/Documents/Kaggle/Taxi/taxi_survival_medians_starthour.csv')
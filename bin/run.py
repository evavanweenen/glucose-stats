import pandas as pd
from cyclingstats.stats import calc_hr_zones, calc_power_zones, agg_zones, agg_power 
from cyclingstats.stats import chronic_training_load, acute_training_load, training_stress_balance

# read time series of power and/or heart rate
df = pd.read_csv("PATH_TO_YOUR_HEARTRATE_AND_POWER_DATA")
df['date'] = pd.to_datetime(df['timestamp'].dt.date)
# perform any other preprocessing steps here

# ---------- zones
# define LTHR and FTP to calculate custom Coggan heart rate and power zones
LTHR = # TODO: fill in a number for the lactate threshold heart rate [bpm]
FTP = # TODO: fill in a number for the functional threshold power [W]

hr_zones = calc_hr_zones(LTHR)
power_zones = calc_power_zones(FTP)

# calculate hr and power zones
df_zones = df.groupby('date').apply(agg_zones, hr_zones=hr_zones, power_zones=power_zones)

# ---------- power
df = df.set_index('timestamp')

# calculate power statistics
df_power = df.groupby('date').apply(agg_power, FTP=FTP)

# fill up dates for which we don't have an entry to get exponential weighted mean (ewm)
dates = df_power.index
df_power = df_power.reindex(date_range)

# calculate ctl, atl and tsb
df_power['chronic_training_load'] = chronic_training_load(df_power['training_stress_score'])
df_power['acute_training_load'] = acute_training_load(df_power['training_stress_score'])
df_power['training_stress_balance'] = training_stress_balance(df_power['chronic_training_load'], df_power['acute_training_load'])

# get back to indices for which there is a training session
df_power = df_power.loc[dates]
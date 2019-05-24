from libs import trading_lib as tl

import numpy as np
import pandas as pd

start_sum = 1_000


def calc_cum_multiplication(series: pd.Series, start: float = start_sum) -> np.array:
    series = np.cumprod(series) * start
    series[0] = start
    return series


# Load data
df_deposit = pd.read_excel('database/Казахстан. Депозиты с 1996.xlsx', converters={'Date': pd.to_datetime})
df_infla = pd.read_csv('database/Казахстан. Инфляция с 1998.csv', engine='python', converters={'Date': pd.to_datetime})
df_kase = pd.read_csv('database/Казахстан. Касе с 2000.csv', engine='python', converters={'Date': pd.to_datetime})
df_meokam = pd.read_excel('database/Казахстан. МЕОКАМ с 1998.xlsx', converters={'Date': pd.to_datetime})
df_reit = pd.read_excel('database/Казахстан. Недвижимость и аренда с 2000.xlsx', converters={'Date': pd.to_datetime})
df_gold = pd.read_csv('database/Gold Daily since 1968.csv', converters={'Date': pd.to_datetime})
df_usdkzt = pd.read_csv('database/Казахстан. Доллар-тенге с 1995.csv', engine='python', converters={'Date': pd.to_datetime})

# Preparation data
df_infla['Percent'] = [float(value.replace("%", '')) / 100 + 1 for value in df_infla["KAZAKHSTANINFRATMOM Adj. Close"]]
df_kase['Percent'] = np.true_divide(df_kase.Close[1:], df_kase.Close[:-1])
df_gold['Close'] = df_gold.Close.replace('.', float('NaN')).fillna(method='ffill')
df_gold['Close'] = df_gold.Close.astype('float')
df_gold['Percent'] = np.true_divide(df_gold.Close[1:], df_gold.Close[:-1])

print(df_kase)

all_frames = [df_deposit, df_infla, df_kase, df_meokam, df_reit, df_gold, df_usdkzt]

# Find_oldest_newest_dates
oldest_date = []
newest_date = []
for frame in all_frames:
    oldest_date.append(frame['Date'][0])
    newest_date.append(frame['Date'].iloc[-1])
start_date, end_date = max(oldest_date), min(newest_date)

# Cut_data_by_dates
print(f"Cutting data by start {start_date} and end {end_date}")
for i in range(len(all_frames)):
    all_frames[i] = all_frames[i].loc[(all_frames[i]['Date'] >= start_date) & (all_frames[i]['Date'] <= end_date)
                        ].reset_index(drop=True)

# Create percent gain
perc_deposit_kzt = all_frames[0]["Deposits from 1 to 5 years (KZT)"] / 12 / 100 + 1
perc_deposit_usd = all_frames[0]["Deposits from 1 to 5 years (USD)"] / 12 / 100 + 1
perc_meokam = all_frames[3]["MEОKAM (<5 years)"] / 12 / 100 + 1
perc_property = all_frames[4]["Old property, M.cng"] / 100 + 1
perc_rent = all_frames[4]["Rent, M.cng"] / 100 + 1
perc_infla = all_frames[1]["Percent"]
perc_kase = all_frames[2]["Percent"]
perc_gold = all_frames[5]["Percent"]
usdkzt = all_frames[6]["Ratio"]

# Create capital gain
capital_deposit_kzt = calc_cum_multiplication(perc_deposit_kzt)
capital_deposit_usd = np.array(calc_cum_multiplication(perc_deposit_usd, start_sum/usdkzt[0])) * np.array(usdkzt)
capital_meokam = calc_cum_multiplication(perc_meokam)
capital_property = calc_cum_multiplication(perc_property)
capital_property_rent = np.cumsum(calc_cum_multiplication(perc_rent, 21.73684)) + capital_property
capital_infla = calc_cum_multiplication(perc_infla)
capital_kase = calc_cum_multiplication(perc_kase)
capital_gold = np.array(calc_cum_multiplication(perc_gold, start_sum/usdkzt[0])) * np.array(usdkzt)

print(capital_property)
print(capital_property_rent)

from libs import trading_lib as tl
from datetime import datetime as dt
from plotly.offline import plot
from plotly import graph_objs as go

import json
import numpy as np
import pandas as pd

start_sum = 1_000


def create_kztk_df(cut_data: dt) -> pd.DataFrame:
    # Take KZTKp prices
    kztk = json.load(open('database/KZTKp.json'))
    kztk_df = pd.DataFrame({
        'Date': pd.Series(kztk['t']).apply(dt.fromtimestamp),
        'Close': pd.Series(kztk['c'])
    })
    kztk_df['Date'] = kztk_df['Date'].apply(lambda x: x.replace(hour=00, minute=00))
    kztk_df['Dividend'] = [0] * len(kztk_df)
    # tl.save_csv('database', 'KZTKp prices', kztk_df)

    # Take KZTKp dividends
    kztk_div = pd.read_excel('database/KZTKp div.xlsx')
    kztk_div['Date'] = pd.to_datetime(pd.Series(kztk_div['Date']))

    # Add dividends to kztk_df frame
    for i in range(len(kztk_div)):
        div_date = kztk_div['Date'][i]
        index = kztk_df['Date'].loc[kztk_df['Date'] >= div_date].first_valid_index()
        kztk_df.loc[index, 'Dividend'] = kztk_div['Dividend'][i]

    kztk_df = kztk_df[kztk_df['Date'] >= cut_data]
    kztk_df['Cls_Div'] = kztk_df['Close'] + np.cumsum(kztk_df['Dividend'])

    # Create final kztk frame
    kztk_final = pd.DataFrame({'Date': all_frames[0]['Date']})
    kztk_final['Close'] = [0] * len(kztk_final)
    for i in range(len(kztk_final)):
        max_date = kztk_final['Date'][i]
        kztk_final.loc[i, 'Close'] = kztk_df['Cls_Div'][kztk_df['Date'] <= max_date].iloc[-1]

    return kztk_final


def calc_cum_multiplication(series: pd.Series, start: float = start_sum) -> np.array:
    series = np.cumprod(series) * start
    series[0] = start
    return series


def create_plotly(deposit_kzt, deposit_usd, meokam, kase, gold, property, property_reit, infla, kztk, chart_name: str,
                  file_name: str):
    # Calc CAGR and volatility
    cagr_deposit_kzt = tl.cagr(list(all_frames[0]['Date']), list(deposit_kzt))
    cagr_deposit_usd = tl.cagr(list(all_frames[0]['Date']), list(deposit_usd))
    cagr_meokam = tl.cagr(list(all_frames[0]['Date']), list(meokam))
    cagr_kase = tl.cagr(list(all_frames[0]['Date']), list(kase))
    cagr_gold = tl.cagr(list(all_frames[0]['Date']), list(gold))
    cagr_property = tl.cagr(list(all_frames[0]['Date']), list(property))
    cagr_property_reit = tl.cagr(list(all_frames[0]['Date']), list(property_reit))
    cagr_kztk = tl.cagr(list(all_frames[0]['Date']), list(kztk))
    cagr_infla = tl.cagr(list(all_frames[0]['Date']), list(infla))
    cagr_infla = cagr_infla if -0.1 > cagr_infla or cagr_infla > 0.1 else 0.0

    vol_deposit_kzt = round(tl.st_dev(list(deposit_kzt), 12) * 100, 2)
    vol_deposit_usd = round(tl.st_dev(list(deposit_usd), 12) * 100, 2)
    vol_meokam = round(tl.st_dev(list(meokam), 12) * 100, 2)
    vol_kase = round(tl.st_dev(list(kase), 12) * 100, 2)
    vol_gold = round(tl.st_dev(list(gold), 12) * 100, 2)
    vol_property = round(tl.st_dev(list(property), 12) * 100, 2)
    vol_property_reit = round(tl.st_dev(list(property_reit), 12) * 100, 2)
    vol_kztk = round(tl.st_dev(list(kztk), 12) * 100, 2)
    vol_infla = round(tl.st_dev(list(infla), 12) * 100, 2)
    vol_infla = vol_infla if -0.1 > vol_infla or vol_infla < 0.1 else 0.0

    chart_table = pd.DataFrame({
        ' ': ['<b>Среднегодовая доходность</b>', '<b>Изменчивость цен</b>'],
        'Депозит KZT': [cagr_deposit_kzt, vol_deposit_kzt],
        'Депозит USD': [cagr_deposit_usd, vol_deposit_usd],
        'МЕОКАМ': [cagr_meokam, vol_meokam],
        'KASE': [cagr_kase, vol_kase],
        'Золото': [cagr_gold, vol_gold],
        'Недвижимость': [cagr_property, vol_property],
        'Недвижимость + аренда': [cagr_property_reit, vol_property_reit],
        'KZTKp': [cagr_kztk, vol_kztk],
        'Инфляция': [cagr_infla, 0.0]
    })
    for column in range(1, len(chart_table.columns)):
        chart_table.iloc[:, column] = chart_table.iloc[:, column].astype(str) + '%'

    # Make chart
    trace1 = go.Scatter(
        x=all_frames[0]["Date"],
        y=deposit_kzt,
        mode='lines',
        # line=dict(color='#0033ff'),
        name='Депозит KZT'
    )
    trace2 = go.Scatter(
        x=all_frames[0]["Date"],
        y=deposit_usd,
        mode='lines',
        line=dict(color='#ff9999'),
        name='Депозит USD'
    )
    trace3 = go.Scatter(
        x=all_frames[0]["Date"],
        y=meokam,
        mode='lines',
        # line=dict(color='#1f77b4'),
        name='МЕОКАМ (до 5 лет)'
    )
    trace4 = go.Scatter(
        x=all_frames[0]["Date"],
        y=kase,
        mode='lines',
        # line=dict(color='#ff0033'),
        name='KASE'
    )
    trace5 = go.Scatter(
        x=all_frames[0]["Date"],
        y=gold,
        mode='lines',
        line=dict(color='#ffcc00'),
        name='Золото (в долларах)'
    )
    trace6 = go.Scatter(
        x=all_frames[0]["Date"],
        y=property,
        mode='lines',
        line=dict(color='#00ccff'),
        name='Недвижимость'
    )
    trace7 = go.Scatter(
        x=all_frames[0]["Date"],
        y=property_reit,
        mode='lines',
        line=dict(color='#00cccc'),
        name='Недвижимость + аренда'
    )
    trace8 = go.Scatter(
        x=all_frames[0]["Date"],
        y=kztk,
        mode='lines',
        line=dict(color='#763ac2'),
        name='Казахтелеком-префы (вкл. дивиденды)'
    )
    trace9 = go.Scatter(
        x=all_frames[0]["Date"],
        y=infla,
        mode='lines',
        line=dict(color='#999999'),
        name='Инфляция тенге'
    )
    trace10 = go.Table(
        domain=dict(
            x=[0.0, 1.0],
            y=[0, 0.2]
        ),
        header=dict(
            values=[f"<b>{key}</b>" for key, _ in chart_table.items()],
            line=dict(color='#506784'),
            align=['center'],
            fill=dict(color='#1f7b4d'),
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[chart_table[key] for key, _ in chart_table.items()],
            line=dict(color='#506784'),
            align=['center'],
            font=dict(color='#506784', size=13)
        )

    )

    plt_data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]

    plt_layout = go.Layout(
        title=chart_name,
        yaxis=dict(
            title='Капитал',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            type='log',
            autorange=True,
            domain=[0.28, 1.0]
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Aleksei Ashikhmin<br>(alex.ash-iv@list.ru)',
                font=dict(
                    # family='Courier New, monospace',
                    size=10,
                    color='gray'
                )
            )
        )
    )

    fig = go.Figure(
        data=plt_data,
        layout=plt_layout
    )
    plot(fig, show_link=False, filename=file_name + '.html')


def create_bars(deposit_kzt, deposit_usd, meokam, kase, gold, property, property_reit, infla, kztk, chart_name: str,
                  file_name: str, method: bool):
    years_slice = ['2001-2003', '2004-2006', '2007-2009', '2010-2012', '2013-2015', '2016-2018'] if method == 0 else\
        ['2001-2018', '2004-2018', '2007-2018', '2010-2018', '2013-2018', '2016-2018']
    assets_dict = {
        'KZT депо': [deposit_kzt, []], 'USD депо': [deposit_usd, []], 'МЕОКАМ': [meokam, []], 'KASE': [kase, []],
        'Золото': [gold, []], 'Недвиж.': [property, []], 'Недвиж. + аренда': [property_reit, []], 'KZTKp': [kztk, []],
        'Инфляция': [infla, []]
    }

    start_year = 2001
    end_year = 2004 if method == 0 else 2019
    while start_year <= 2016:
        slice = all_frames[0]['Date'][
            (all_frames[0]['Date'] >= dt(start_year, 1, 1)) & (all_frames[0]['Date'] < dt(end_year, 1, 1))]
        first_idx = slice.first_valid_index()
        last_idx = slice.last_valid_index()

        for asset in assets_dict:
            data = assets_dict[asset][0]
            assets_dict[asset][1].append(tl.cagr(list(slice), list(data[first_idx:last_idx])))

        start_year += 3
        end_year = end_year + 3 if method == 0 else end_year

    plt_data = []
    colors = ['#1F77B4', '#FF9999', '#2CA02C', '#D62728', '#FFCC00', '#00CCFF', '#00CCCC', '#763AC2']
    for i in range(8):
        plt_data.append(go.Bar(
            x=years_slice,
            y=assets_dict[[*assets_dict][i]][1],
            name=[*assets_dict][i],
            marker=dict(color=colors[i])
        ))

    plt_layout = go.Layout(
        title=chart_name,
        barmode='group',
        yaxis=dict(
            title='Доходность в %',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            range=[-25, 100] if method == 0 else [-10, 50]
        ),
        xaxis = go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Aleksei Ashikhmin<br>(alex.ash-iv@list.ru)',
                font=dict(
                    # family='Courier New, monospace',
                    size=10,
                    color='gray'
                )
            )
        )
    )
  
    fig = go.Figure(
        data=plt_data,
        layout=plt_layout
    )
    plot(fig, show_link=False, filename=file_name + '.html')

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
capital_deposit_kzt = calc_cum_multiplication(perc_deposit_kzt).astype(int)
capital_deposit_usd = (calc_cum_multiplication(perc_deposit_usd, start_sum/usdkzt[0]) * usdkzt).astype(int)
capital_meokam = calc_cum_multiplication(perc_meokam).astype(int)
capital_property = calc_cum_multiplication(perc_property).astype(int)
capital_property_rent = (np.cumsum(calc_cum_multiplication(perc_rent, 21.73684)) + capital_property).astype(int)
capital_infla = calc_cum_multiplication(perc_infla).astype(int)
capital_kase = calc_cum_multiplication(perc_kase).astype(int)
capital_gold = (calc_cum_multiplication(perc_gold, start_sum/usdkzt[0]) * usdkzt).astype(int)

# Adding KazakhTelecom
df_kztk = create_kztk_df(dt(2000, 10, 31))
df_kztk['Percent'] = np.true_divide(df_kztk.Close[1:], df_kztk.Close[:-1])
capital_kztk = calc_cum_multiplication(df_kztk['Percent']).astype(int)

create_plotly(
    capital_deposit_kzt, capital_deposit_usd, capital_meokam, capital_kase, capital_gold, capital_property,
    capital_property_rent, capital_infla, capital_kztk,
    'Вложение 1 000 тенге в различные активы',
    'Kazakhstan Investing'
)

# Create capital gain with inflation
revers_perc_infla = 1 / perc_infla
revers_perc_infla = np.cumprod(revers_perc_infla)

capital_deposit_kzt = (capital_deposit_kzt * revers_perc_infla).astype(int)
capital_deposit_usd = (capital_deposit_usd * revers_perc_infla).astype(int)
capital_meokam = (capital_meokam * revers_perc_infla).astype(int)
capital_property = (capital_property * revers_perc_infla).astype(int)
capital_property_rent = (capital_property_rent * revers_perc_infla).astype(int)
capital_infla = (capital_infla * revers_perc_infla).astype(int)
capital_kase = (capital_kase * revers_perc_infla).astype(int)
capital_gold = (capital_gold * revers_perc_infla).astype(int)
capital_kztk = (capital_kztk * revers_perc_infla).astype(int)

create_plotly(
    capital_deposit_kzt, capital_deposit_usd, capital_meokam, capital_kase, capital_gold, capital_property,
    capital_property_rent, capital_infla, capital_kztk,
    'Вложение 1 000 тенге в различные активы с учётом инфляции.<br>Чистый прирост покупательской способности.',
    'Kazakhstan Investing + Inflation'
)

create_bars(
    capital_deposit_kzt, capital_deposit_usd, capital_meokam, capital_kase, capital_gold, capital_property,
    capital_property_rent, capital_infla, capital_kztk,
    'Чистая среднегодовая доходность.<br>Если начать и завершать инвестировать в разные года',
    'Kazakhstan Investing Slices',
    method=False
)
create_bars(
    capital_deposit_kzt, capital_deposit_usd, capital_meokam, capital_kase, capital_gold, capital_property,
    capital_property_rent, capital_infla, capital_kztk,
    'Чистая среднегодовая доходность.<br>Если начать инвестировать в разные года, но заканчивать в 2018',
    'Kazakhstan Investing TotalSlices',
    method=True
)

import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

dir_path = os.path.dirname(os.path.realpath(__file__))


def save_to_file(climate_df, month, year):
    if not os.path.exists("{0}/Data/html/{1}".format(dir_path, year)):
        os.makedirs("{0}/Data/html/{1}".format(dir_path, year))
    climate_df.to_csv("{0}/Data/html/{1}/{2}.csv".format(dir_path, year, month))


def read_and_save_table(text, month, year):
    soup = BeautifulSoup(text, "lxml")
    table_data = []
    for table in soup.findAll('table', {'class': "medias mensuales numspan"}):
        for tbody in table:
            temp_table_data = []
            for tr in tbody:
                if tr.get_text() == '-':
                    temp_table_data.append(None)
                else:
                    temp_table_data.append(tr.get_text())
            table_data.append(temp_table_data)
    table_data.pop(-1)
    table_data.pop(-1)
    headers = table_data.pop(0)
    climate_data = pd.DataFrame(np.row_stack(table_data), columns=headers).set_index('Day')
    save_to_file(climate_data, month, year)


def get_pm25_data():
    day = 1
    for year in range(2013, 2019):
        url = "{0}/Data/AQI-Data/aqi{1}.csv".format(dir_path, year)
        avg_data = []
        for rows in pd.read_csv(url, chunksize=24):
            rows = rows[~rows["PM2.5"].isin(["NoData", "---", "InVld", "PwrFail"])]
            rows['PM2.5'] = pd.to_numeric(rows["PM2.5"], downcast="float")
            avg_value = rows["PM2.5"].mean()
            avg_data.append({"day": day, "PM2.5": avg_value})
            day = day + 1
        new_df = pd.DataFrame(avg_data)
        new_df.dropna(inplace=True)
        new_url = "{0}/Data/AQI-Data/aqi{1}_formatted.csv".format(dir_path, year)
        new_df.to_csv(new_url)


def merge_data():
    final_AQI_data = pd.DataFrame()
    for year in range(2013, 2019):
        year_df = pd.read_csv("{0}/Data/html/{1}/1.csv".format(dir_path, year, 1))
        for month in range(2, 13):
            url = "{0}/Data/html/{1}/{2}.csv".format(dir_path, year, month)
            climate_data = pd.read_csv(url)
            year_df = year_df.append(climate_data, ignore_index=True)
        aqi_data = pd.read_csv("{0}/Data/AQI-Data/aqi{1}_formatted.csv".format(dir_path, year))
        pm25_data = aqi_data["PM2.5"]
        df_concat = pd.concat([year_df, pm25_data], axis=1)
        df_concat.drop(["Day"], axis=1)
        df_concat["year"] = year
        if len(final_AQI_data) > 0:
            final_AQI_data = final_AQI_data.append(df_concat, ignore_index=True)
        else:
            final_AQI_data = df_concat
    if not os.path.exists("{0}/Data/realData".format(dir_path)):
        os.makedirs("{0}/Data/realData".format(dir_path))
    final_AQI_data.drop(["Day", "VG", "RA", "SN", "TS", "FG"], axis=1, inplace=True)
    final_AQI_data.dropna(inplace=True)
    final_AQI_data.to_csv("{0}/Data/realData/final_aqi_data.csv".format(dir_path))


def get_html():
    for year in range(2013, 2019):
        for month in range(1, 13):
            if month < 10:
                change = "0{0}-{1}".format(month, year)
            else:
                change = "{0}-{1}".format(month, year)

            url = "https://en.tutiempo.net/climate/{0}/ws-421820.html".format(change)
            texts = requests.get(url)
            text_utf = texts.text.encode('utf=8')
            read_and_save_table(text_utf, month, year)


def fetch_and_save_data():
    #get_html()
    get_pm25_data()
    merge_data()

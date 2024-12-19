import asyncio
from datetime import datetime

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

OPENWEATHERMAP_WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_GEO_API_URL = "http://api.openweathermap.org/geo/1.0/direct"
MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


async def get_current_temperature_async(city, api_key):
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(OPENWEATHERMAP_WEATHER_API_URL, params=params) as response:
            if response.status != 200:
                st.write(await response.json())
                return None
            data = await response.json()
            return data["main"]["temp"]


def is_temperature_normal(data, current_city, current_city_temp, season):
    seasonal_mean = data[(data['city'] == current_city) & (data['season'] == season)]['seasonal_mean'].values[0]
    seasonal_std = data[(data['city'] == current_city) & (data['season'] == season)]['seasonal_std'].values[0]
    return seasonal_mean - 2 * seasonal_std <= current_city_temp <= seasonal_mean + 2 * seasonal_std


def process_data(df):
    seasonal_stats = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std'])
    seasonal_stats = seasonal_stats.rename(columns={'mean': 'seasonal_mean', 'std': 'seasonal_std'})
    df = df.merge(seasonal_stats, on=['city', 'season'], how='left')
    df["temp_m_avg"] = df.groupby("city")["temperature"].transform(
        lambda x: x.rolling(window=30, center=True).apply(np.nanmedian)
    )
    df["temp_m_std"] = df.groupby("city")["temperature"].transform(
        lambda x: x.rolling(window=30, center=True).apply(np.nanstd)
    )
    df["anomaly"] = ((df["temperature"] < df["temp_m_avg"] - 2 * df["temp_m_std"])
                     | (df["temperature"] > df["temp_m_avg"] + 2 * df["temp_m_std"]))
    return df


async def main():
    uploaded_file = st.file_uploader("Choose a CSV file with temperature data")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        selected_city = st.selectbox("Выберите город", data["city"].unique())
        seasons = data["season"].unique()
        city_data = data[data["city"] == selected_city]
        city_data = process_data(city_data)

        st.header(f"Статистика по {selected_city}")
        st.write(city_data.describe())

        st.subheader("Температурный ряд")
        plt.figure(figsize=(10, 6))
        plt.plot(city_data["timestamp"], city_data["temperature"], marker="o")
        plt.title("Температурный ряд")
        plt.xlabel("Дата")
        plt.ylabel("Температура (°C)")
        plt.grid(True)
        st.pyplot(plt)

        st.subheader("Аномалии")
        anomalies = city_data[city_data["anomaly"]]
        plt.figure(figsize=(10, 6))
        plt.plot(city_data["timestamp"], city_data["temperature"], label="Температура")
        plt.scatter(
            anomalies["timestamp"], anomalies["temperature"], color="red", label="Аномалии"
        )
        plt.xlabel("Дата")
        plt.ylabel("Температура, °C")
        plt.title("Аномалии")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        st.subheader("Сезонные профили")
        season_profile = (
            city_data.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
        )
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            season_profile["season"],
            season_profile["mean"],
            yerr=season_profile["std"],
            marker='o',
            mfc='red',
            mec='green',
            ms=10,
            mew=4,
            label="Средняя температура",
        )
        plt.xticks(ticks=range(len(seasons)), labels=season_profile["season"])
        plt.xlabel("Время года")
        plt.ylabel("Температура, °C")
        plt.title("Сезонные профили температуры")
        plt.grid(True)
        for i, row in season_profile.iterrows():
            plt.text(
                row["season"],
                row["mean"] + 1.2,
                f"         {round(row['mean'], 2)}",
                ha="center",
                va="bottom",
            )
        st.pyplot(plt)

        st.header("Работа с OpenWeatherMap")
        api_key = st.text_input("Введите API-ключ OpenWeatherMap")

        if api_key:
            current_temp = await get_current_temperature_async(selected_city, api_key)
            if not current_temp:
                return

            st.write(f"Текущая температура в городе {selected_city}: {current_temp} °C")
            now = datetime.now()
            st.write(f"Текущий сезон: {MONTH_TO_SEASON[now.month]} ({now})")

            st.subheader("Классификация: нормальная / не нормальная")
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                season_profile["season"],
                season_profile["mean"],
                yerr=season_profile["std"],
                marker='o',
                label="Средняя температура",
            )
            plt.scatter(MONTH_TO_SEASON[now.month], current_temp, color="red", label="Текущая температура")
            plt.text(
                MONTH_TO_SEASON[now.month],
                current_temp + 1.2,
                f"         {round(current_temp, 2)}",
                ha="center",
                va="bottom",
            )
            plt.xticks(ticks=range(len(seasons)), labels=season_profile["season"])
            plt.xlabel("Время года")
            plt.ylabel("Температура, °C")
            plt.title("Сезонные профили температуры")
            plt.grid(True)
            for i, row in season_profile.iterrows():
                plt.text(
                    row["season"],
                    row["mean"] + 1.5,
                    f"         {round(row['mean'], 2)}",
                    ha="center",
                    va="bottom",
                )
            plt.legend()

            normal = is_temperature_normal(city_data, selected_city, current_temp, MONTH_TO_SEASON[now.month])
            if normal:
                st.write('Температура классифицирована как нормальная')
            else:
                st.write('Температура классифицирована как не нормальная')

            st.pyplot(plt)


asyncio.run(main())

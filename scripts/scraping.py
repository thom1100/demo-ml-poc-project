import aiohttp
import asyncio
import pandas as pd

async def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "start_date": "2021-01-01",
        "end_date": "2021-01-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Europe/Paris"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temperature_mean": data["daily"]["temperature_2m_mean"],
        "precipitation_sum": data["daily"]["precipitation_sum"]
    })

    return df

weather_df = asyncio.run(fetch_weather())
print(weather_df.head())

weather_df.to_csv("../../data/raw/weather_data.csv")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#PRIMARILY VIBECODED! Just testing library's abilities 

# Пример данных (в тыс. литров, ручной парсинг)
data = {
    'year': [2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023],
    'month': [3, 4, 5, 3, 4, 5, 3, 4, 5],  # март-май
    'sales': [85, 78, 120, 90, 82, 125, 95, 88, 130],  # продажи
    'promo': [0, 0, 1, 0, 1, 0, 1, 0, 1],  # были ли акции
    'dacha_season': [0, 0, 1, 0, 0, 1, 0, 0, 1]  # дачный сезон
}

df = pd.DataFrame(data)

# Прогноз на май 2024
def calculate_may_forecast(df):
    # Подготовка данных для модели
    x = df[['year', 'month', 'promo', 'dacha_season']]
    y = df['sales']
    
    # Обучаем модель
    model = LinearRegression()
    model.fit(x, y)
    
    # Прогноз на май 2024
    may_2024 = pd.DataFrame({
        'year': [2024],
        'month': [5],
        'promo': [1],  # акция запланирована
        'dacha_season': [1]  # дачный сезон
    })
    
    base_forecast = model.predict(may_2024)[0]
    
    # Корректировки на специфические факторы
    promo_effect = base_forecast * 0.3  # акция дает +30% к спросу
    dacha_effect = base_forecast * 0.15  # дачный сезон +15%. Проценты выдуманы, предварительного расчета не было 
    final_forecast = base_forecast + promo_effect + dacha_effect
    
    # Распределение по сетям (исходя из исторических долей). Процент деления выдумал гпт
    pyaterochka_share = 0.4  # 40%
    magnit_share = 0.6       # 60%
    
    return {
        'total_forecast': round(final_forecast, 1),
        'pyaterochka': round(final_forecast * pyaterochka_share, 1),
        'magnit': round(final_forecast * magnit_share, 1)
    }

result = calculate_may_forecast(df)
print(f"Общий прогноз на май: {result['total_forecast']} тыс. литров")
print(f"Пятёрочка: {result['pyaterochka']} тыс. литров") 
print(f"Магнит: {result['magnit']} тыс. литров")
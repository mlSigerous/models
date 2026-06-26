import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Данные
sales_data = {
    "year": [2020, 2021, 2022, 2023, 2024],
    "sales": [11.1, 11.3, 11.3, 11.7, 12.4],
}

df = pd.DataFrame(sales_data)


def smart_sales_forecast(df):

    recent_data = df[df["year"] >= 2022].copy()

    # Сдвигаем годы для лучшей стабильности модели
    X = recent_data[["year"]].values - 2022  # 2022=0, 2023=1, 2024=2
    y = recent_data["sales"].values

    # Обучаем линейную регрессию на ускоренном тренде
    model = LinearRegression()
    model.fit(X, y)

    # Прогноз на 2025 (год = 2025-2022 = 3)
    next_year = np.array([[3]])
    prediction_2025 = model.predict(next_year)[0]

    # Расчет точности на исторических данных
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return prediction_2025, model, r2, recent_data


# Запускаем прогноз
prediction, model, r2, recent_data = smart_sales_forecast(df)

# Визуализация
plt.figure(figsize=(12, 8))

# Все исторические данные
plt.plot(
    df["year"],
    df["sales"],
    "bo-",
    linewidth=2,
    markersize=8,
    label="Исторические данные",
    alpha=0.7,
)

# Данные, используемые для прогноза (выделяем цветом)
plt.plot(
    recent_data["year"],
    recent_data["sales"],
    "go-",
    linewidth=3,
    markersize=10,
    label="Данные для прогноза (2022-2024)",
)

# Прогноз
future_year = 2025
plt.plot(
    future_year,
    prediction,
    "r*",
    markersize=20,
    label=f"Прогноз 2025: {prediction:.2f}",
)

# Линия тренда (только для периода прогноза)
years_trend = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1) - 2022
trend_line = model.predict(years_trend)
plt.plot([2022, 2023, 2024, 2025], trend_line, "r--", label="Тренд ускоренного роста")

plt.xlabel("Год", fontsize=12)
plt.ylabel("Объем продаж", fontsize=12)
plt.title("Прогноз продаж с учетом смены тренда", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df["year"].tolist() + [2025])

# Добавляем аннотации
plt.annotate(
    f"Прогноз 2025: {prediction:.2f}",
    xy=(2025, prediction),
    xytext=(2023.5, 12.8),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=12,
    color="red",
)

plt.tight_layout()
plt.show()

# Детальная информация
print("\n" + "=" * 50)
print("ДЕТАЛИ ПРОГНОЗА:")
print("=" * 50)
print(f"Модель построена на данных: {recent_data['year'].tolist()}")
print(f"Соответствующие продажи: {recent_data['sales'].tolist()}")
print(f"Тренд: +{model.coef_[0]:.3f} единиц в год")
print(f"Качество модели (R²): {r2:.4f}")
print(f"ПРОГНОЗ НА 2025: {prediction:.2f}")

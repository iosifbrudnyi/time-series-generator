import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from datetime import timedelta
import numpy as np

# Загрузка данных
df = pd.read_csv('test_data/test1.csv')

# Преобразование timestamp в pandas datetime
df['time'] = pd.to_datetime(df['time'])

# Сортировка по времени (на всякий случай)
df = df.sort_values('time')

# Вычисление интервалов между временными метками
df['interval'] = df['time'].diff().dt.total_seconds()
df = df.dropna()  # Удаление первого NaN значения

# Визуализация распределения интервалов
plt.figure(figsize=(10, 6))
plt.hist(df['interval'], bins=50, density=True, alpha=0.6, color='g')
plt.title('Distribution of Time Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Density')
plt.show()

# Использование GMM для моделирования распределения интервалов
gmm = GaussianMixture(n_components=3, random_state=0)  # Количество компонентов можно настроить
gmm.fit(df['interval'].values.reshape(-1, 1))

# Генерация новых интервалов
n_intervals = 1000  # Количество генерируемых интервалов
new_intervals = gmm.sample(n_intervals)[0].flatten()

# Преобразование интервалов в временные метки
start_time = df['time'].iloc[0]  # Начальное время для нового датасета
new_times = [start_time]

for interval in new_intervals:
    new_times.append(new_times[-1] + timedelta(seconds=interval))

# Создание нового DataFrame для сгенерированных данных
new_df = pd.DataFrame(new_times, columns=['time'])

# Визуализация новых данных
plt.figure(figsize=(10, 6))
plt.plot(new_df['time'], np.zeros(len(new_df)), '|')
plt.title('Generated Time Series Data')
plt.xlabel('Time')
plt.ylabel('Events')
plt.show()

# Сохранение нового датасета
new_df.to_csv('gen_data/gen1.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Загрузка данных
original_df = pd.read_csv('test_data/test1.csv')
generated_df = pd.read_csv('gen_data/gen1.csv')

# Преобразование timestamp в pandas datetime
original_df['time'] = pd.to_datetime(original_df['time'])
generated_df['time'] = pd.to_datetime(generated_df['time'])

# Вычисление интервалов между временными метками
original_df['interval'] = original_df['time'].diff().dt.total_seconds().dropna()
generated_df['interval'] = generated_df['time'].diff().dt.total_seconds().dropna()

# Визуальное сравнение распределений
plt.figure(figsize=(10, 6))
plt.hist(original_df['interval'], bins=50, density=True, alpha=0.6, color='g', label='Original')
plt.hist(generated_df['interval'], bins=50, density=True, alpha=0.6, color='b', label='Generated')
plt.title('Histogram of Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Ядерные оценки плотности (KDE)
plt.figure(figsize=(10, 6))
original_df['interval'].plot(kind='kde', label='Original', color='g')
generated_df['interval'].plot(kind='kde', label='Generated', color='b')
plt.title('KDE of Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Статистическое сравнение
ks_stat, ks_p_value = ks_2samp(original_df['interval'], generated_df['interval'])
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")

# Проверка среднего и стандартного отклонения
original_mean = original_df['interval'].mean()
original_std = original_df['interval'].std()
generated_mean = generated_df['interval'].mean()
generated_std = generated_df['interval'].std()

print(f"Original Mean: {original_mean}, Original Std: {original_std}")
print(f"Generated Mean: {generated_mean}, Generated Std: {generated_std}")

# Проверка автокорреляции
original_autocorr = [original_df['interval'].autocorr(lag) for lag in range(1, 11)]
generated_autocorr = [generated_df['interval'].autocorr(lag) for lag in range(1, 11)]

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), original_autocorr, marker='o', label='Original')
plt.plot(range(1, 11), generated_autocorr, marker='o', label='Generated')
plt.title('Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.legend()
plt.show()

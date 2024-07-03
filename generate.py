import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from datetime import timedelta

# Загрузка данных
df = pd.read_csv('test_data/test1.csv')

# Преобразование timestamp в pandas datetime
df['time'] = pd.to_datetime(df['time'])

# Вычисление интервалов между временными метками
df['interval'] = df['time'].diff().dt.total_seconds()
df = df.dropna()  # Удаление первого NaN значения

# Подготовка данных для моделирования
X = df['interval'].values.reshape(-1, 1)

# Определение диапазона гиперпараметров
n_components_range = np.arange(1, 10)
covariance_types = ['full', 'tied', 'diag', 'spherical']

# Функция для вычисления среднего значения BIC с использованием кросс-валидации
def compute_bic_with_cv(X, n_components, covariance_type):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    bics = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0)
        gmm.fit(X_train)
        bic = gmm.bic(X_test)
        bics.append(bic)
    return np.mean(bics)

# Перебор различных параметров для GMM и выбор наилучшей модели по BIC
lowest_bic = np.inf
best_gmm = None
best_params = None

for n_components in n_components_range:
    for covariance_type in covariance_types:
        bic = compute_bic_with_cv(X, n_components, covariance_type)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0).fit(X)
            best_params = (n_components, covariance_type)

print(f"Best model parameters: n_components={best_params[0]}, covariance_type={best_params[1]}")

# Генерация новых интервалов
n_intervals = 500  # Количество генерируемых интервалов
new_intervals = best_gmm.sample(n_intervals)[0].flatten()

# Преобразование интервалов в временные метки
start_time = df['time'].iloc[0]  # Начальное время для нового датасета
new_times = [start_time]

for interval in new_intervals:
    new_times.append(new_times[-1] + timedelta(seconds=interval))

# Создание нового DataFrame для сгенерированных данных
new_df = pd.DataFrame(new_times, columns=['time'])

# Визуализация оригинальных данных
plt.figure(figsize=(10, 6))
plt.hist(df['interval'], bins=50, density=True, alpha=0.6, color='g')
plt.title('Distribution of Time Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Density')

# Визуализация временного ряда оригинальных данных
plt.figure(figsize=(10, 6))
plt.plot(df['time'], np.zeros(len(df)), '|')
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Events')

# Визуализация временного ряда сгенерированных данных
plt.figure(figsize=(10, 6))
plt.plot(new_df['time'], np.zeros(len(new_df)), '|')
plt.title('Generated Time Series Data')
plt.xlabel('Time')
plt.ylabel('Events')
plt.show()

# Сохранение нового датасета
new_df.to_csv('gen_data/gen1.csv', index=False)

import sys
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from datetime import timedelta

if len(sys.argv) < 3:
    print("Usage example: python generate.py <data_path> <n_intervals>")
    sys.exit()

data_path = sys.argv[1]
n_intervals = int(sys.argv[2])

# Загрузка данных
df = pd.read_csv(data_path)

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
# n_intervals = 500  # Количество генерируемых интервалов
new_intervals = best_gmm.sample(n_intervals)[0].flatten()

# Преобразование интервалов в временные метки
start_time = df['time'].iloc[0]  # Начальное время для нового датасета
new_times = [start_time]

for interval in new_intervals:
    new_times.append(new_times[-1] + timedelta(seconds=interval))

# Создание нового DataFrame для сгенерированных данных
new_df = pd.DataFrame(new_times, columns=['time'])

out_file_name = sys.argv[1].split("/")[-1]
out_file_name = out_file_name.split(".")[0]
out_file_name += "_gen.csv"

# Сохранение нового датасета
new_df.to_csv("gen_data/" + out_file_name, index=False)

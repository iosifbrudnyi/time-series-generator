import numpy as np
import pandas as pd

def read_data(file_path):
    """
    Считывает данные из CSV файла и возвращает DataFrame
    """
    df = pd.read_csv(file_path)
    return df

def analyze_data(df):
    """
    Анализирует данные и возвращает статистические характеристики
    """
    # Вычисляем интервалы между событиями
    intervals = df['time'].diff().dropna()
    
    # Вычисляем среднее время между событиями
    mean_interval = intervals.mean()
    
    # Вычисляем дисперсию времени между событиями 
    var_interval = intervals.var()
    
    return mean_interval, var_interval

def generate_data(mean_interval, var_interval, num_events):
    """
    Генерирует новые данные на основе статистических характеристик
    """
    # Генерируем интервалы между событиями из экспоненциального распределения
    intervals = np.random.exponential(mean_interval, num_events)
   
    # Вычисляем время каждого события
    times = np.cumsum(intervals)

    # Создаем DataFrame с сгенерированными данными
    df = pd.DataFrame({'time': times, 'event': range(1, num_events+1)})

    return df

# Пример использования


if __name__ == "__main__":
    file_path = 'test_data/test1.csv'
    df = read_data(file_path)

    mean_interval, var_interval = analyze_data(df)
    print(f"Среднее время между событиями: {mean_interval:.2f}")
    print(f"Дисперсия времени между событиями: {var_interval:.2f}")

    gen_df = generate_data(mean_interval, var_interval, 1000)
    gen_df.to_csv('gen_data/gen1.csv', index=False)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Загрузка данных
df = pd.read_csv("fish_train.csv")

# Отбор числовых и категориальных колонок
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[np.object]).columns

# Определение признаков (X) и целевой переменной (y)
X_numeric = df[numeric_cols].drop("Weight", axis=1)
X_categorical = df[categorical_cols]
y = df["Weight"]

# Разбиение данных на обучающую и тестовую выборки с учетом стратификации
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=21, stratify=df['Species']
)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания для тестового набора данных
y_pred = model.predict(X_test)

# Оценка модели при помощи метрики r2_score
r2 = r2_score(y_test, y_pred)

# Вывод результата
print("R²-статистика:", r2)

# < ENTER YOUR CODE HERE > 
# Построение матрицы корреляций для тренировочного набора данных
correlation_matrix_train = X_train.corr()

# Построение тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_train, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляций для тренировочного набора данных')
plt.show()


# Нахождение тройки наиболее коррелированных признаков
correlation_matrix_flatten = correlation_matrix_train.abs().unstack()
correlation_matrix_flatten = correlation_matrix_flatten[correlation_matrix_flatten < 1]  # Исключаем диагональные элементы
correlation_matrix_flatten = correlation_matrix_flatten.reset_index()
correlation_matrix_flatten.columns = ['Feature 1', 'Feature 2', 'Correlation']
correlation_matrix_flatten['Feature Pair'] = correlation_matrix_flatten.apply(lambda row: '-'.join(sorted([row['Feature 1'], row['Feature 2']])), axis=1)
correlation_matrix_flatten = correlation_matrix_flatten.drop_duplicates(subset='Correlation', keep='first')
top_correlations_train = correlation_matrix_flatten.nlargest(3, 'Correlation')[['Feature 1', 'Feature 2', 'Correlation']]

# correlations_train = correlation_matrix_train.unstack().sort_values(kind="quicksort", ascending=False)
# top_correlations_train = correlations_train[correlations_train < 1].head(3).index.tolist()

# Вывод тройки наиболее коррелированных признаков
print("Тройка наиболее коррелированных признаков:")
print(top_correlations_train)


####### ВТОРОЕ ЗАДАНИЕ - ПРО PCA МОДЕЛЬ

pca = PCA(n_components=3, svd_solver='full')
# Доля объясненной дисперсии для первой главной компоненты
print("Доля объясненной дисперсии для первой главной компоненты:", pca.explained_variance_ratio_[0])

##################### ДЛЯ TRAIN ##############################
X_pca = pca.fit_transform(X_train[["Length1", "Length2", "Length3"]])

# Получение значений счетов первой главной компоненты
lengths_pca = X_pca[:, 0]
# Удаление трех наиболее коррелированных признаков
X_train = X_train.drop(top_correlations_train[['Feature 1', 'Feature 2']].values.flatten(), axis=1)
# Добавление нового признака Lengths
X_train['Lengths'] = lengths_pca
##############################################################


##################### ДЛЯ TEST ###############################
X_pca = pca.transform(X_test[["Length1", "Length2", "Length3"]])

# Получение значений счетов первой главной компоненты
lengths_pca = X_pca[:, 0]
# Удаление трех наиболее коррелированных признаков
X_test = X_test.drop(top_correlations_train[['Feature 1', 'Feature 2']].values.flatten(), axis=1)
# Добавление нового признака Lengths
X_test['Lengths'] = lengths_pca
##############################################################


model = LinearRegression();
model.fit(X_train, y_train)

# Предсказания для тестового набора данных
y_pred = model.predict(X_test)

# Оценка модели при помощи метрики r2_score
r2 = r2_score(y_test, y_pred)

# Вывод результата
print("R²-статистика:", r2)



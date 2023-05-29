import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Читаем csv файл
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Проверяем столбцы с пустыми значениями
data.isnull().sum()
# Удаляем столбцы где много пропущенных значений
data.drop('Cabin', axis=1, inplace=True)
# Удаляем данные с пропущенными значениями
data.dropna(inplace=True)

# Можно увидеть первые 5 строк
data.head()
# Переводим в числовые значения
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# X - наши критерии, y - что мы хотим предсказать
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
#X = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# Тренировка дерева решений
dt = DecisionTreeClassifier(max_depth=3)
dt_model = dt.fit(x_train, y_train)


fig = plt.figure(figsize=(12, 10))
tree.plot_tree(dt_model, feature_names=list(X.columns), class_names=['Not survived', 'Survived'])
plt.show()

# Оценка эффективности модели
print("Accuracy score on Testing set: ", dt_model.score(x_test, y_test))
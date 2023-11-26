import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
raw_data = pd.read_csv(r'train.csv', encoding='unicode_escape')
raw_data.drop(["date", "station"], axis=1, inplace=True)
raw_data[raw_data == 'NR'] = 0
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
raw_data.set_index(raw_data.iloc[:, 0], inplace=True)
raw_data.drop('testmaterial', axis=1, inplace=True)
data = pd.DataFrame(np.zeros((5760, 18)))
data.set_axis(raw_data.index.tolist()[:18], axis='columns', inplace=True)
for row in range(raw_data.shape[0]):
    for col in range(raw_data.shape[1]):
        data.iloc[col + row//18 * 24][row % 18] = raw_data.iloc[row][col]
data.to_csv('清洗后数据.csv')
data = pd.read_csv('清洗后数据.csv')
data.drop(data.columns[0], axis=1, inplace=True)
cor = data.corr()
sns.heatmap(cor, cmap='YlGnBu', square=True, annot=True)
plt.title('Heatmap of Indicators')
plt.show()
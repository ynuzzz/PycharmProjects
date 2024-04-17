import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_excel('jobs_in_data.xltx')
df.head()


#değişkenleri seçtik
df = df[['job_title' , 'job_category' , 'salary_in_usd' , 'employee_residence' , 'experience_level' , 'employment_type' , 'company_location' , 'company_size']]
df = df[df['employment_type'] == 'Full-time']
df.drop_duplicates(inplace=True)


#az görünen değerleri grupladık
df['company_location'].value_counts()
def shorten_category(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map =shorten_category(df.company_location.value_counts() , 60)
df['company_location'] = df['company_location'].map(country_map)
df['company_location'].value_counts()

#outlierları veri setinden çıkardık
df.boxplot('salary_in_usd' , 'company_location')
plt.title('Before Handling the Outliers')
plt.xticks(rotation = 50)
plt.show()


def find_outliers(df, column_name, location_value):
    subset = df[df['company_location'] == location_value][column_name]

    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df['company_location'] == location_value) & (
                (df[column_name] < Q1 - 1.5 * IQR) | (df[column_name] > Q3 + 1.5 * IQR))]

    return outliers

outliers_us_salary = find_outliers(df, 'salary_in_usd', 'United States')
df.drop(outliers_us_salary.index, inplace=True)
outliers_us_salary = find_outliers(df, 'salary_in_usd', 'Other')
df.drop(outliers_us_salary.index, inplace=True)
outliers_us_salary = find_outliers(df, 'salary_in_usd', 'United Kingdom')
df.drop(outliers_us_salary.index, inplace=True)

df.boxplot('salary_in_usd' , 'company_location')
plt.title('After Handling the Outliers')
plt.xticks(rotation = 50)
plt.show()

#encoding
from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['job_title'] = le_education.fit_transform(df['job_title'])
df["job_title"].unique()

le_education = LabelEncoder()
df['job_category'] = le_education.fit_transform(df['job_category'])
df["job_category"].unique()

le_education = LabelEncoder()
df['employee_residence'] = le_education.fit_transform(df['employee_residence'])
df["employee_residence"].unique()

le_education = LabelEncoder()
df['experience_level'] = le_education.fit_transform(df['experience_level'])
df["experience_level"].unique()

le_education = LabelEncoder()
df['employment_type'] = le_education.fit_transform(df['employment_type'])
df["employment_type"].unique()

le_education = LabelEncoder()
df['company_location'] = le_education.fit_transform(df['company_location'])
df["company_location"].unique()

le_education = LabelEncoder()
df['company_size'] = le_education.fit_transform(df['company_size'])
df["company_size"].unique()

#bağımlı değikenimizi çıkarıyoruz
X = df.drop(['salary_in_usd' , 'job_title'] , axis = 1)
Y = df['salary_in_usd']

#doğrusal modele sokuyoruz
linear_model = LinearRegression()
linear_model.fit(X,Y.values)

from sklearn.metrics import mean_squared_error, mean_absolute_error
y_pred = linear_model.predict(X)
error = np.sqrt(mean_squared_error(Y, y_pred))
print("${:,.02f}".format(error))
#mse: $50,844.82
#benzersiz mse: $55,931.91

#karar ağaçı
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, Y.values)

y_pred = dec_tree_reg.predict(X)
y_pred = dec_tree_reg.predict(X)
error = np.sqrt(mean_squared_error(Y, y_pred))
print("${:,.02f}".format(error))

#mse:$44,686.53
#benzersiz mse:$47,026.92

#tahmin modeline örnek alma

X = np.array(['Data Engineering', 'Germany', 'Mid-level', 'Full-time', 'United States', 'M'])

le_education = LabelEncoder()
all_unique_values = np.unique(X)
le_education.fit(all_unique_values)

X = le_education.transform(X)
X = X.astype(float)
X

y_pred = dec_tree_reg.predict(X.reshape(1, -1))
tahmin = y_pred[0]

print("${:,.02f}".format(tahmin))

#tahmin :$80,000.00














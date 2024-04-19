import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# Veriyi yükleyip ilk işlemleri yap
def load_and_preprocess_data():
    df = pd.read_excel('jobs_in_data.xltx')
    df = df[['job_title', 'job_category', 'salary_in_usd', 'employee_residence', 'experience_level', 'employment_type',
             'company_location', 'company_size']]
    df = df[df['employment_type'] == 'Full-time']  # Sadece tam zamanlı işleri al
    df.drop_duplicates(inplace=True)  # Çift girdileri sil

    # Az görünen değerleri gruplama
    def shorten_category(categories, cutoff):
        categorical_map = {}
        for i in range(len(categories)):
            if categories.values[i] >= cutoff:
                categorical_map[categories.index[i]] = categories.index[i]
            else:
                categorical_map[categories.index[i]] = 'Other'
        return categorical_map

    country_map = shorten_category(df.company_location.value_counts(), 60)
    df['company_location'] = df['company_location'].map(country_map)

    # Outlier'ları temizleme fonksiyonu
    def remove_outliers(df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Maaş için outlier'ları kaldır
    df = remove_outliers(df, 'salary_in_usd')

    # Kategorik ve sürekli değişkenleri ayır
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numeric_features = df.select_dtypes(exclude=['object']).columns.drop('salary_in_usd').tolist()

    # Kategorik değişkenler için OneHotEncoder, sürekli değişkenler için StandardScaler kullan
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer ile ön işleme adımlarını birleştir
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return df, preprocessor


# Modeli eğit ve kaydet
def train_and_save_model(df, preprocessor):
    X = df.drop('salary_in_usd', axis=1)
    y = df['salary_in_usd']

    # Pipeline oluştur
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Modeli eğit
    model_pipeline.fit(X, y)

    # Modeli dosya olarak kaydet
    joblib.dump(model_pipeline, 'model_pipeline.pkl')


# Main kısmı
if __name__ == "__main__":
    df, preprocessor = load_and_preprocess_data()
    train_and_save_model(df, preprocessor)


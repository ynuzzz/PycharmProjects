#### Streamlit ###

import streamlit as st
import pandas as pd


#### Streamlit ###



import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import plotly.express as px
from plotly.offline import iplot , plot
from plotly.subplots import make_subplots

# Veri yükleme ve ön işleme fonksiyonu
def get_data():
    dataframe = pd.read_excel('/Users/yunusseyrek/PycharmProjects/Final_Projesi_Miuul/jobs_in_data.xltx')
    return dataframe

# Veriyi yükle
data = get_data()

# Model ve veri ön işleme pipeline'ı
categorical_features = ['job_title', 'job_category', 'employee_residence', 'experience_level', 'employment_type', 'company_location', 'company_size']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)
])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modeli eğit
X_train = data.drop(columns=['salary'])  # 'salary' sütununun adını doğru şekilde ayarlayın
y_train = data['salary']  # 'salary' sütununun adını doğru şekilde ayarlayın
model_pipeline.fit(X_train, y_train)

# Streamlit uygulaması

st.set_page_config(layout= "wide", page_title= "Izmir Data")
st.title("İzmir Data")
st.write("Veri Bilimi Yolculuğunuza Hoşgeldiniz!")

main_tab, us_tab, gra_tab, salary_tab = st.tabs(["Home", "Hakkımızda", "Analiz", "Maaş Tahmini"])

## 1) Anasayfa
left_col, right_col = main_tab.columns(2)
left_col.write(" * Siz de bir veri profesyoneli olmak mı istiyorsunuz?"
         "\n * Peki ne kadar kazanacağınızı bilmiyor musunuz?"
         "\n * O zaman doğru yerdesiniz.")
right_col.image("Saat.webp", width=400)

## 2) Hakkımızda
left_col, right_col = us_tab.columns(2)
left_col.write(" * Can - Elektrik Elektrik Mühendisi"
         "\n * Saitcan - Elektrik Elektrik Mühendisi"
         "\n * Özge - Endüstri Mühendisi"
         "\n * Rana - Endüstri Mühendisi"
         "\n * Yunus - İnşaat Mühendisi"
         "\n * Dilem - Jeoloji Mühendisi")
right_col.image("Saat.webp", width=400)


## 3) Güncel Veriler ##

import plotly.express as px
import pandas as pd
import streamlit as st
import plotly.express as px
import pycountry
import plotly.express as px

#Birinci grafik
with gra_tab:
    # Ülke adlarını ISO kodlarına çevirme fonksiyonu
    def get_country_code(country_name):
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except:
            return None  # Eğer ülke bulunamazsa None döndür


    # 'company_location' sütunundaki ülke adlarını ISO kodlarına çevir
    data['iso_alpha'] = data['company_location'].apply(get_country_code)

    # Veri setini kontrol edin
    print(data[['company_location', 'iso_alpha']].head())

    # Ülkeleri gruplayın ve ortalama maaşı hesaplayın
    country_avg_salaries = data.groupby('iso_alpha')['salary_in_usd'].mean().reset_index()

    # Harita oluşturma
    fig = px.choropleth(country_avg_salaries,
                        locations='iso_alpha',  # ISO kodlarını içeren sütun
                        color='salary_in_usd',  # Renklendirme için maaş sütunu
                        hover_name='iso_alpha',  # Fareyle üzerine gelindiğinde gösterilecek ISO kodu
                        color_continuous_scale=px.colors.sequential.Plasma,  # Renk skalası
                        title='Global Average Salary Distribution by Country')
    fig.update_geos(projection_type="natural earth")  # Harita stilini 'natural earth' olarak güncelle

   # Streamlit'te göster
    st.plotly_chart(fig, use_container_width=True)


  #İkinci grafik

   # 'work_setting' sütunundaki verileri kullanarak bir pie chart oluşturun
    df_work_setting = data['work_setting'].value_counts()
    fig = px.pie(values=df_work_setting.values,
             names=df_work_setting.index,  # Kategori isimlerini doğrudan veri setinden alın
             title='Type of Work Setting')
    fig.update_traces(textinfo='percent+label')

   # Streamlit'te göster
    st.plotly_chart(fig, use_container_width=True)

   #Üçüncü Grafik
   # 'company_size' sütunundaki verileri kullanarak bir pie chart oluşturun
    df_company_size = data['company_size'].value_counts()
    fig = px.pie(values=df_company_size.values,
             names=df_company_size.index,  # 'M', 'L', 'S' yerine gerçek kategori isimlerini kullanın
             title='Size of Company')
    fig.update_traces(textinfo='percent+label')

  # Streamlit'te göster
    st.plotly_chart(fig, use_container_width=True)


   ## Dördüncü Grafik
  # Veri setinden elde edilen verilere dayanarak bir pie chart oluşturun
    data2=data[data['job_title'] == 'Data Scientist']
    df_experience_level = data2['experience_level'].value_counts()
    fig = px.pie(values=df_experience_level.values,
             names=df_experience_level.index,  # Gerçek kategori isimlerini kullanın
             title='Experience Level for Data Scientist')
    fig.update_traces(textinfo='percent+label')

   # Streamlit'te göster
    st.plotly_chart(fig, use_container_width=True)

    ## Beşinci Grafik

    # Meslek başlıklarına göre gruplandırma ve ortalama maaşların hesaplanması
    ortalama_maaslar = data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

    # İlk 20 mesleği göstermek için ortalama maaşların sıralanması ve sınırlanması
    top_ortalama_maaslar = ortalama_maaslar.head(20)

    # Sütun grafiği çizdirme
    fig = px.bar(
        x=top_ortalama_maaslar.values,
        y=top_ortalama_maaslar.index,
        labels={'x': 'Ortalama Maaş (USD)', 'y': 'Meslek'},
        orientation='h',
        title='Top 20 Jobs',
        color=top_ortalama_maaslar.values,  # Maaş değerlerine göre renklendirme
        color_continuous_scale=px.colors.sequential.Viridis  # Viridis renk skalasını kullanma
    )

    # Grafiğin görsel ayarlarını yapılandırma
    fig.update_layout(
        xaxis_title='Ortalama Maaş (USD)',
        yaxis_title='Meslek',
        title_font_size=16,
        xaxis_tickangle=45,
        xaxis_gridcolor='gray',
        yaxis=dict(categoryorder='total ascending'),
        coloraxis_colorbar=dict(
            title='Maaş USD'
        )
    )

    # Streamlit'te grafiği göster
    st.plotly_chart(fig, use_container_width=True)

## 4) Maaş Tahmini
with salary_tab:
    st.header("Maaş Tahmini Yap")
    job_title = st.selectbox('İş Unvanı', options=np.unique(data['job_title']))
    job_category = st.selectbox('İş Kategorisi', options=np.unique(data['job_category']))
    employee_residence = st.selectbox('Çalışanın Yerleşim Yeri', options=np.unique(data['employee_residence']))
    experience_level = st.selectbox('Deneyim Seviyesi', options=np.unique(data['experience_level']))
    employment_type = st.selectbox('İstihdam Türü', options=np.unique(data['employment_type']))
    company_location = st.selectbox('Şirketin Konumu', options=np.unique(data['company_location']))
    company_size = st.selectbox('Şirket Büyüklüğü', options=np.unique(data['company_size']))

    # Tahmin yapma butonu
    if st.button('Maaş Tahmini Yap'):
        # Girdileri dataframe'e dönüştür
        input_data = pd.DataFrame([[job_title, job_category, employee_residence, experience_level, employment_type, company_location, company_size]],
                                  columns=categorical_features)
        # Tahmin yap
        prediction = model_pipeline.predict(input_data)
        st.write(f"Tahmini Maaş: ${prediction[0]:,.2f}")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import pycountry

# Modeli yükle
model_pipeline = joblib.load('model_pipeline.pkl')

# Veri yükleme ve işleme
def get_data():
    dataframe = pd.read_excel('jobs_in_data.xltx')
    return dataframe

data = get_data()

st.set_page_config(layout="wide", page_title="Izmir Data")
st.title("İzmir Data")
st.write("Veri Bilimi Yolculuğunuza Hoşgeldiniz!")

main_tab, us_tab, gra_tab, salary_tab = st.tabs(["Home", "Hakkımızda", "Analiz", "Maaş Tahmini"])

with main_tab:
    left_col, right_col = st.columns(2)
    left_col.write("""
        * Siz de bir veri profesyoneli olmak mı istiyorsunuz?
        * Peki ne kadar kazanacağınızı bilmiyor musunuz?
        * O zaman doğru yerdesiniz.
    """)
    right_col.image("Saat.webp", width=400)

with us_tab:
    left_col, right_col = st.columns(2)
    left_col.write("""
        * Can - Elektrik Elektrik Mühendisi
        * Saitcan - Elektrik Elektrik Mühendisi
        * Özge - Endüstri Mühendisi
        * Rana - Endüstri Mühendisi
        * Yunus - İnşaat Mühendisi
        * Dilem - Jeoloji Mühendisi
    """)
    right_col.image("Saat.webp", width=400)
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
        input_data = np.array([[job_title, job_category, employee_residence, experience_level, employment_type, company_location, company_size]])
        input_df = pd.DataFrame(input_data, columns=['job_title', 'job_category', 'employee_residence', 'experience_level', 'employment_type', 'company_location', 'company_size'])

        # Tahmini yap
        prediction = model_pipeline.predict(input_df)
        st.success(f'Tahmini Maaş: ${prediction[0]:,.2f}')





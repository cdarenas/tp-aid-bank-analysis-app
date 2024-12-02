#  █████╗ ██╗██████╗       ████████╗██████╗      
# ██╔══██╗██║██╔══██╗      ╚══██╔══╝██╔══██╗
# ███████║██║██║  ██║         ██║   ██████╔╝
# ██╔══██║██║██║  ██║         ██║   ██╔═══╝ 
# ██║  ██║██║██████╔╝         ██║   ██║     
# ╚═╝  ╚═╝╚═╝╚═════╝          ╚═╝   ╚═╝        
# Descripción: Script para AID - TP2.
# Este script realiza análisis de datos avanzados con visualización y estadísticas.
# Autor: Cristian D. Arenas
# Fecha: 07/11/2024


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

PAGE_CONFIG = {"page_title": "Fantasy Bank", "page_icon": ":bank:",
               "layout": "wide", "initial_sidebar_state": "collapsed"}

st.set_page_config(**PAGE_CONFIG)


def show_init_page():
    data_file = st.file_uploader("Subir CSV", type="csv")
    if data_file is not None:
        file_details = {"filename": data_file.name,
                        "filetype": data_file.type, "filesize": data_file.size}
        st.write(file_details)
        st.session_state.data_file = data_file
        df = pd.read_csv(data_file)

        st.subheader("Exploración del dataset")
        #st.dataframe(df)
        st.write('Primeras 10 filas del DataFrame:')
        st.dataframe(df.head(10), use_container_width=True)

        # Creao un nuevo dataframe con nombres de columnas y tipos de datos
        df_info = pd.DataFrame({
            'Columnas': df.columns}).reset_index(drop=True)

        # Creo una nueva columna para una breve descripción de cada campo
        df_info['Description'] = ["ID del cliente", "Apellido", "Nivel Crediticio", "País", "Sexo", "Edad", "Antiguedad como cliente",
                                  "Saldo", "Cantidad de productos en uso", "Tiene tarjeta de crédito", "Es miembro activo", "Salario estimado", "Abandonó"]
        df_info['Type'] = ["Discreta", "Cualitativa", "Cuantitativa Ordinal", "Categórica", "Categórica",
                           "Discreta", "Discreta", "Continua", "Discreta", "Binaria", "Binaria", "Continua", "Binaria"]

        st.subheader("Nombres de columnas, descripción y tipos de variables:")
        st.dataframe(df_info, use_container_width=True)

        # Calcular los datos faltantes por columna
        missing_data = df.isnull().sum()

        # Creao un DataFrame para visualizarlo mejor
        missing_data_df = pd.DataFrame(missing_data, columns=['Datos Faltantes'])
        # Filtro solo columnas con datos faltantes
        missing_data_df = missing_data_df[missing_data_df['Datos Faltantes'] > 0]

        # Mouestro el número de datos faltantes por columna
        st.subheader("Cantidad de datos faltantes por columna: ")
        st.dataframe(missing_data_df)

        # Selección de la estrategia de imputación
        method = st.selectbox("Selecciona el método para manejar los datos faltantes:",
                          ("Eliminar filas", "Media", "Mediana", "Moda", "Valor fijo"))

        if method == "Eliminar filas":
            # Elimina filas con cualquier valor faltante
            df = df.dropna()
            st.write("Filas con valores faltantes eliminadas.")
        elif method == "Media":
            # Imputación con la media
            # Solo columnas numéricas
            for col in df.select_dtypes(include='number').columns:
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
            st.write("Valores faltantes reemplazados por la media de cada columna numérica.")
        elif method == "Mediana":
            # Imputación con la mediana
            # Solo columnas numéricas
            for col in df.select_dtypes(include='number').columns:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
            st.write("Valores faltantes reemplazados por la mediana de cada columna numérica.")
        elif method == "Moda":
            # Imputación con la moda
            for col in df.columns:
                mode_value = df[col].mode()[0]  # Obtener la moda de cada columna
                df[col] = df[col].fillna(mode_value)
            st.write("Valores faltantes reemplazados por la moda de cada columna.")
        elif method == "Valor fijo":
            # Imputación con un valor fijo, ingresado por el usuario
            fixed_value = st.text_input("Ingresa el valor fijo para reemplazar los valores faltantes:")
            if fixed_value:
                df = df.fillna(fixed_value)
                st.write(f"Valores faltantes reemplazados por el valor fijo: {fixed_value}")

        # Permito eliminar del dataframe los registros de países que no me interesen para el análisis
        # Obtener un vector con los países sin repetir
        countries = df['Geography'].unique()
        countries_to_drop = st.multiselect("Selecciona los países que quieras excluir del DataFrame (Máx 2):",
                                           countries,
                                           max_selections=2)
        # Mostrar los países seleccionados
        st.write('Países seleccionados:', countries_to_drop)

        # Excluir del dataframe las filas que tengan en la columna Geografía los países seleccionados
        df_filtered = df[~df['Geography'].isin(countries_to_drop)]

        st.session_state.dataframe = df_filtered
    else:
        st.write("Por favor, sube un archivo CSV para comenzar.")


def show_exploration_page():
    df = st.session_state.dataframe

    st.success("Descripción analítica del conjunto de datos:")

    # Descripción analítica del conjunto de datos
    df_describe = df.drop(columns=['CustomerId']).describe()
    st.dataframe(df_describe, use_container_width=True)

    st.markdown("<h2 style='font-size:16px;'>Distribución de clientes según su nivel crediticio</h2>", unsafe_allow_html=True)
    # Calcular los cuartiles del nivel crediticio
    st.write("La mayoría de los clientes tienen un nivel crediticio entre 584 y 718, por lo tanto la distribución de niveles crediticios está concentrada en este rango. Esto sugiere una cartera de clientes con un nivel de crédito relativamente moderado. Un 25% se encuentra por debajo de ese rango, con un puntaje bajo y sólo un 25% de los clientes posee un puntaje alto (718+).")

    # Histograma de la distribución de edades
    fig = px.histogram(df, x='Age', nbins=10, title='Distribución de las edades')
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig)

    st.write("Se puede apreciar que la distribución de edades presenta una asimetría positiva (hacia la derecha), con una mayor concentración de clientes adultos jóvenes y una cola que se extiende hacia edades más avanzadas.")
    st.write("La edad promedio de los clientes es de 39 años y la mediana se ubica en 37 años lo que nos indica que hay una distribución relativamente asimétrica, con un 50% de clientes por debajo de los 37 años y un 50% por encima de los 37 años.")

    # Gráfico de barras de la distribución por país
    fig = px.bar(df, x='Geography', title='Distribución por país')
    st.plotly_chart(fig)

    st.write("Se puede observar que Francia es el país con mayor frecuencia de clientes del banco; aproximadamente el 50% de los clientes totales del banco pertenecen a este país y la otra mitad está concentrada entre España y Alemania.")

    cantidades = df['Gender'].value_counts()
    df_cantidades = cantidades.reset_index()
    df_cantidades.columns = ['Gender', 'Count']

    # Crear el gráfico de sectores (pie chart) con Plotly Express
    fig = px.pie(df_cantidades, 
             names='Gender',  # Columna para las categorías
             values='Count',  # Columna con los valores
             color='Gender',  # Columna para asignar colores
             title="Distribución de Género de los Clientes",
             color_discrete_map={"Male": "royalblue", "Female": "lightpink"},  # Colores personalizados
             hole=0.3)

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

    st.write("Respecto de la distribución del género dentro del conjunto de clientes del banco, el 54.6% son hombres, podemos decir que no hay una diferencia significativa entre hombres y mujeres.")

    # Gráfico de cajas de Saldo por país
    fig = px.box(df, x='Geography', y='Balance', title='Saldo en cuenta por país')
    st.plotly_chart(fig)

    st.write("Respecto de la distribución de saldo en cuenta por país, podemos observar que para los casos de España y Francia, la mediana es de aproximadamente 62.000 Euros y ambos países de acuerdo al Rango Intercuartílico, tienen una variabilidad más alta en los saldos que para el caso de Alemania. Para este último país, la variabilidad es menor y la mayor concentración se da en el rango de 103.000 y 137.000 Euros.")
    st.write("Estas diferencias de medias entre los países puede deberse a niveles de ingreso más elevados en unos países que en otros.")
    st.write("Para entender si los salarios promedio de cada país tienen impacto en los saldos, calculamos saldo y salario estimado promedio por país y graficamos los resultados.")

    # Calcular saldo y salario promedio por país
    saldo_pais = df.groupby("Geography")["Balance"].mean()
    salario_pais = df.groupby("Geography")["EstimatedSalary"].mean()

    comparacion_df = pd.DataFrame({
    "Saldo Promedio": saldo_pais,
    "Salario Promedio": salario_pais
    })

    # Crear gráfico de barras agrupadas
    fig = px.bar(comparacion_df.reset_index(), 
             x="Geography", 
             y=["Saldo Promedio", "Salario Promedio"],
             title="Comparación de Saldo y Salario Promedio por País",
             barmode="group",
             labels={"value": "Monto", "variable": "Métrica"},
             color_discrete_map={"Saldo Promedio": "#1f77b4", "Salario Promedio": "lightgreen"})

    st.plotly_chart(fig)

    st.write("Como podemos apreciar, los salarios promedios en los tres países son similares y sólo el saldo en cuenta promedio en Alemania (120.000) difiere de los otros dos países, los cuales tienen promedios similares cercanos a los 60.000 Euros.")
    st.write("La distribución de frecuencias podría sugerir que en Alemania los clientes podrían inclinarse hacia la acumulación de dinero en sus cuentas para ahorro u otros fines.")
    
    # Análisis gráfico de CreditScore según Gender
    fig = px.box(df, x='Gender', y='CreditScore', title='Nivel crediticio por género')
    st.plotly_chart(fig)

    st.write("En relación a la distribución del puntaje de crédito por género, el gráfico sugiere cierta simetría de los datos, medias similares para ambos grupos, con similar variabilidad de los datos. La mediana para el grupo femenino es de 652 y para el caso de los hombres es de 651.")

    # Crear una nueva columna en el DataFrame para agrupar la edad en intervalos de clase
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70, 80], 
                         labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71-80"])

    # Crear el box plot
    fig = px.box(df, 
             x="Age_Group", 
             y="Balance", 
             title="Distribución de Saldo por Grupos de Edad",
             labels={"Age_Group": "Grupo de Edad", "Balance": "Saldo"},
             color="Age_Group",  # Asigna colores a cada grupo de edad para diferenciarlos
             points=False)  # Incluye todos los puntos de datos para ver posibles valores atípicos

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

    st.write("En el boxplot podemos apreciar que todos los intervalos de clase o grupos de edades poseen una dispersión en sus saldos muy similar. La asimetría negativa de las cajas, nos da la idea de que la mayoría de los clientes mantienen saldos por debajo de los 100.000 Euros y una pequeña parte por encima de los 100.000 Euros. Para los casos de los grupos de 31-40 y de 51-60 años, se aprecia una mayor dispersión de datos, con saldos acumulados más extremos sugeridos por sus rangos de valores.")
    

def show_analysis_page():
    df = st.session_state.dataframe
    if st.session_state.data_file is not None:
        st.success("Análisis del conjunto de datos:")
        # Agrupar por Género y Geografía y calcular la tasa de abandono
        tasa_abandono = df.groupby(['Gender', 'Geography'])['Exited'].mean() * 100
        # Convertir el resultado a un DataFrame
        tasa_abandono_df = tasa_abandono.reset_index()

        # Renombrar las columnas (cabeceras personalizadas)
        tasa_abandono_df.columns = ['Género', 'País', 'Tasa de Abandono (%)']
        # Ordenar las filas por 'Tasa de Abandono (%)' de forma descendente
        tasa_abandono_df = tasa_abandono_df.sort_values(by='Tasa de Abandono (%)', ascending=False)
        st.dataframe(tasa_abandono_df)

        st.write("Como podemos apreciar en la tabla, Alemania tiene las tasas de abandono de clientes más altas, en primer lugar las mujeres con un 37.55% y luego hombres con un 27.81%. Podemos concluir que el banco posee las tasas más altas de abandono concentradas en el género femenino, por lo tanto se podría trabajar en estrategias orientadas a esté grupo. Por otra parte también orientar acciones en el mercado Alemán con el objetivo de reducir la tasa de abandono en general.")

        # 1. Distribución de la Edad por Abandono (Histograma)
        st.subheader("Distribución de Edad por Abandono")

        # Crear un histograma con Plotly Express
        fig_hist = px.histogram(df, x="Age", color="Exited", nbins=20,
                        labels={"Age": "Edad", "Exited": "Abandonó (Exited)"})
        fig_hist.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig_hist)

        st.write("La mayor frecuencia de clientes que abandonan el banco se da en el rango de 39-50 años. De todas formas, se puede apreciar que en los grupos de edades de clientes más jóvenes la proporción de clientes que abandonan la entidad es baja en comparación con los grupos de edades más avanzadas. Se puede interpretar que en clientes más grandes, el nivel de deserción es más alto.")

        st.subheader("Contrastes de Hipótesis")
        st.write("Para corroborar la hipótesis acerca de que los clientes del banco que abandonaron tienen un salario promedio significativamente más bajo que los clientes que permanecen, se realizó una prueba t de Student para las medias de ambos grupos.")
        st.write("H₀: La media del salario de los clientes que abandonaron es igual a la media del salario de los clientes que no abandonaron.")
        st.write("H₁: La media del salario de los clientes que abandonaron es menor que la media del salario de los clientes que no abandonaron.")
        # Filtrar los salarios de los clientes que abandonaron (Exited = 1) y los que no (Exited = 0)
        salarios_abandonaron = df[df['Exited'] == 1]['EstimatedSalary']
        salarios_no_abandonaron = df[df['Exited'] == 0]['EstimatedSalary']

        # Realizar la prueba t de Student para muestras independientes
        t_stat, p_value = stats.ttest_ind(salarios_abandonaron, salarios_no_abandonaron, alternative='less')

        # Mostrar los resultados de la prueba
        st.write("Estadístico t:", t_stat)
        st.write("Valor p:", p_value)

        # Conclusión
        alpha = 0.05  # Nivel de significancia
        if p_value < alpha:
            st.write("Rechazamos la hipótesis nula. Existe evidencia suficiente para afirmar que los clientes que abandonaron tienen un salario medio más bajo.")
        else:
            st.write("No rechazamos la hipótesis nula. No hay evidencia suficiente para afirmar que los clientes que abandonaron la entidad tienen un salario medio más bajo.")

        st.success("Conclusiones finales:")
        st.write("Podemos concluir que podrían existir cuestiones demográficas, culturales, de género y relacionadas con la edad que pueden incidir en los clientes que deciden abandonar el banco. No se encontraron relaciones entre Abandono y variables como Credit Score, Salario Estimado o Saldo acumulado; es decir, no necesariamente los clientes que abandonan a la entidad suelen tener un nivel crediticio bajo, sueldos bajos o saldos acumulados bajos. Ambos grupos (Exited=0 y Exited=1) en la distribución de datos de las variables mencionadas anteriormente, poseen una similitud en cuanto a la concentración de los datos centrales y la variabilidad de los mismos, lo que nos sugiere que probalemente existen otros factores que pueden impactar en la decisión de abandonar el banco.")


def show_logistic_regression():
    df = st.session_state.dataframe
    if st.session_state.data_file is not None:
        # Crear una columna con el índice original
        df['OriginalIndex'] = df.index
        st.subheader("Vista previa del dataset")
        st.write(df.head())
        # Seleccionar características y objetivo
        st.sidebar.subheader("Seleccionar columnas")
        features = st.sidebar.multiselect("Selecciona las columnas de características (X)", df.columns)
        st.success("Para la variable independiente (y) el modelo trabajará con 'Exited' (Abandono)")
        if features:
            X = df[features].values
            y = df.iloc[:, 12].values
            original_indices = df['OriginalIndex'].values  # Índices originales
            # División en conjunto de entrenamiento y prueba
            test_size = st.radio("Selecciona el porcentaje para Test", [0.1, 0.2, 0.3])
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, original_indices, test_size = test_size, random_state = 0)
            # Escalado de variables
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            # Entrenar modelo de regresión logística
            st.sidebar.subheader("Entrenar modelo")
            if st.sidebar.button("Entrenar"):
                classifier = LogisticRegression(random_state = 0)
                classifier.fit(X_train, y_train)
                # Predicción de los resultados con el Conjunto de Testing
                y_pred  = classifier.predict(X_test)
                # Mostramos resultados del modelo
                st.subheader("Resultados del modelo")
                # Matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                st.text("Matriz de Confusión:")
                st.write(cm)
                st.text("Métricas de clasificación:")
                st.text(classification_report(y_test, y_pred))
                # Visualizar la matriz de confusión
                st.subheader("Gráfico de la matriz de confusión:")
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
                fig, ax = plt.subplots(figsize=(5, 5))
                cm_display.plot(cmap="Blues", ax=ax)
                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                 # Cambiar color de las etiquetas a blanco o amarillo
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.tick_params(axis='both', colors='white')  # Color de las etiquetas de los ejes
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('yellow') 
                st.pyplot(fig)
                # Mostrar predicciones con índices originales
                st.subheader("Predicciones")
                # Asociar las predicciones con los índices originales
                results = pd.DataFrame({
                    'OriginalIndex': indices_test,
                    'Predicted': y_pred,
                    'Actual': y_test
                }).sort_values(by='OriginalIndex')
                st.dataframe(results, use_container_width=True)


def main():
    # Inicializo variables globales en la session
    if 'data_file' not in st.session_state:
        st.session_state.data_file = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None

    st.sidebar.success("Menu")
    menu = ["🏠 Inicio", "📝 Descipción de datos", "📊 Análisis", "🧠 Regresión Logística", "Acerca de..."]
    selected_option = st.sidebar.selectbox("Opciones", menu)

    if selected_option == "🏠 Inicio":
        st.header("Análisis Inteligente de Datos")
        st.subheader("MCD - Universidad Austral")
        st.subheader("TP 2 - Clientes del Banco X")
        st.write("Autor: Cristian D. Arenas")
        st.warning("Introducción")
        st.write("El Banco X con sucursales en tres países de Europa y casa central en Alemania, viene experimentando desde hace tres años un incremento en la tasa anual de abandono. Esta tasa se ha ido incrementando en los últimos años y se desea conocer el perfil de los clientes, con el fin de poder diseñar un plan para retenerlos como usuarios de sus productos y servicios.")
        st.write("Alcance del estudio: muestra y objetivos")
        st.write("El estudio tiene como objetivo analizar las características de los clientes del banco para entender el perfil de los mismos.")
        st.write("Para la realización del estudio y análisis se utilizó un dataset que almacena 10.000 filas que contienen datos de clientes de distintas sucursales del banco.")
        st.write("El dataset llamado 'Bank Customer Churn' puede descargarse desde la siguiente URL: https://mavenanalytics.io/data-playground?dataStructure=Single%20table&order=date_added%2Cdesc")
        st.write("Para la presentación del análisis y conclusiones, se decidió desarrollar una web app con Python, utilizando el framework Streamlit")
        st.write("Página web oficial de Streamlit: https://streamlit.io/")

        st.success("Selección y limpieza de los datos")
        show_init_page()
    elif selected_option == "📝 Descipción de datos":
        st.subheader("Descripción analítica y gráfica de los datos")
        show_exploration_page()
    elif selected_option == "📊 Análisis":
        st.subheader("Análisis de Datos - Abandono de Clientes")
        show_analysis_page()
    elif selected_option == "🧠 Regresión Logística":
        st.subheader("Regresión Logística - Entrenamiento para predicción de abandono")
        show_logistic_regression()
    else:
        st.subheader("Acerca de...")
        st.write("Esta aplicación fue creada para presentar como trabajo práctico de la materia Análisis Inteligentes de Datos. Los datos utilizados son ficticios y el dataset fue descargado del sitio https://mavenanalytics.io")
        st.write("Las consignas utilizadas en el trabajo práctico y relacionadas con el caso ficticio del Banco X, no tienen relación con un caso real.")


if __name__ == '__main__':
    main()

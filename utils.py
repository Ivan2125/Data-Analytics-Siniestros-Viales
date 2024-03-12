## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def verifica_duplicados_por_columna(df, columna):
    """
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    """
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"

    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted


def verificar_tipo_variable(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
    """

    mi_dict = {"nombre_campo": [], "tipo_datos": []}

    for columna in df.columns:
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
    df_info = pd.DataFrame(mi_dict)

    return df_info


def convertir_a_time(x):
    """
    Convierte un valor a un objeto de tiempo (time) de Python si es posible.

    Esta función acepta diferentes tipos de entrada y trata de convertirlos en objetos de tiempo (time) de Python.
    Si la conversión no es posible, devuelve None.

    Parameters:
        x (str, datetime, or any): El valor que se desea convertir a un objeto de tiempo (time).

    Returns:
        datetime.time or None: Un objeto de tiempo (time) de Python si la conversión es exitosa,
        o None si no es posible realizar la conversión.
    """
    if isinstance(x, str):
        try:
            return datetime.strptime(x, "%H:%M:%S").time()
        except ValueError:
            return None
    elif isinstance(x, datetime):
        return x.time()
    return x


def imputa_valor_frecuente(df, columna):
    """
    Imputa los valores faltantes en una columna de un DataFrame con el valor más frecuente.

    Esta función reemplaza los valores "SD" con NaN en la columna especificada,
    luego calcula el valor más frecuente en esa columna y utiliza ese valor
    para imputar los valores faltantes (NaN).

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna a ser imputada.
        columna (str): El nombre de la columna en la que se realizará la imputación.

    Returns:
        None
    """
    # Se reemplaza "SD" con NaN en la columna
    df[columna] = df[columna].replace("SD", pd.NA)

    # Se calcula el valor más frecuente en la columna
    valor_mas_frecuente = df[columna].mode().iloc[0]
    print(f"El valor mas frecuente es: {valor_mas_frecuente}")

    # Se imputan los valores NaN con el valor más frecuente
    df[columna].fillna(valor_mas_frecuente, inplace=True)


def imputa_edad_media_segun_sexo(df, col, agr):
    """
    Imputa valores faltantes en la columna 'edad' utilizando la edad promedio según el género.

    Esta función reemplaza los valores "SD" con NaN en la columna 'edad', calcula la edad promedio
    para cada grupo de género (Femenino y Masculino), imprime los promedios calculados y
    luego llena los valores faltantes en la columna 'edad' utilizando el promedio correspondiente
    al género al que pertenece cada fila en el DataFrame.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna 'edad' a ser imputada.

    Returns:
        None
    """

    # Se reemplaza "SD" con NaN en la columna 'edad'
    df[col] = df[col].replace("SD", pd.NA)

    # Se calcula el promedio de edad para cada grupo de género
    promedio_por_genero = df.groupby(agr)[col].mean()
    print(
        f'La edad promedio de Femenino es {round(promedio_por_genero["FEMENINO"])} y de Masculino es {round(promedio_por_genero["MASCULINO"])}'
    )

    # Se llenan los valores NaN en la columna 'edad' utilizando el promedio correspondiente al género
    df[col] = df.apply(
        lambda row: (promedio_por_genero[row[agr]] if pd.isna(row[col]) else row[col]),
        axis=1,
    )
    # Lo convierte a entero
    df[col] = df[col].astype(int)


def verificar_tipo_datos_y_nulos(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'no_nulos_%': Porcentaje de valores no nulos en cada columna.
        - 'nulos_%': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    """

    mi_dict = {
        "nombre_campo": [],
        "tipo_datos": [],
        "no_nulos_%": [],
        "nulos_%": [],
        "nulos": [],
    }

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100 - porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)

    return df_info.sort_values(ascending=False, by="nulos_%")


def accidentes_mensuales(df, col_anio, col_mes):
    """
    Crea gráficos de línea para la cantidad de víctimas de accidentes mensuales por año.

    Esta función toma un DataFrame que contiene datos de accidentes, extrae los años únicos
    presentes en la columna 'Año', y crea gráficos de línea para la cantidad de víctimas por mes
    para cada año. Los gráficos se organizan en una cuadrícula de subgráficos de 2x3.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes, con una columna 'Año'.

    Returns:
        None
    """
    # Se obtiene una lista de años únicos
    años = df[col_anio].unique()

    # Se define el número de filas y columnas para la cuadrícula de subgráficos
    n_filas = 3
    n_columnas = 2

    # Se crea una figura con subgráficos en una cuadrícula de 2x3
    fig, axes = plt.subplots(n_filas, n_columnas, figsize=(14, 8))

    # Se itera a través de los años y crea un gráfico por año
    for i, year in enumerate(años):
        fila = i // n_columnas
        columna = i % n_columnas

        # Se filtran los datos para el año actual y agrupa por mes
        data_mensual = (
            df[df[col_anio] == year].groupby(col_mes).agg({"num_victimas": "sum"})
        )

        # Se configura el subgráfico actual
        ax = axes[fila, columna]
        data_mensual.plot(ax=ax, kind="line")
        ax.set_title("Año " + str(year))
        ax.set_xlabel("Mes")
        ax.set_ylabel("Cantidad de Víctimas")
        ax.legend_ = None

    # Se muestra y acomoda el gráfico
    plt.tight_layout()
    plt.show()


def cantidad_victimas_mensuales(df, col_mes):
    """
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por mes.

    Esta función toma un DataFrame que contiene datos de accidentes, agrupa los datos por mes
    y calcula la cantidad total de víctimas por mes. Luego, crea un gráfico de barras que muestra
    la cantidad de víctimas para cada mes.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'Mes'.

    Returns:
        None
    """
    # Se agrupa por la cantidad de víctimas por mes
    data = df.groupby(col_mes).agg({"num_victimas": "sum"}).reset_index()

    # Se grafica
    plt.figure(figsize=(7, 7))
    ax = sns.barplot(x=col_mes, y="num_victimas", data=data, hue=col_mes)
    ax.set_title("Cantidad de víctimas por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad accidentes")

    # Se imprime resumen
    print(f"El mes con menor cantidad de víctimas tiene {data.min()[1]} víctimas")
    print(f"El mes con mayor cantidad de víctimas tiene {data.max()[1]} víctimas")

    # Se muestra el gráfico
    plt.show()


def cantidad_victimas_por_dia_semana(df):
    """
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por día de la semana.

    Esta función toma un DataFrame que contiene datos de accidentes, convierte la columna 'fecha' a tipo de dato
    datetime si aún no lo es, extrae el día de la semana (0 = lunes, 6 = domingo), mapea el número del día
    de la semana a su nombre, cuenta la cantidad_accidentes por día de la semana y crea un gráfico de barras
    que muestra la cantidad de víctimas para cada día de la semana.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'fecha'.

    Returns:
        None
    """
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df["dia_semana"] = df["fecha"].dt.dayofweek

    # Se mapea el número del día de la semana a su nombre
    dias_semana = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    ]
    df["nombre_dia"] = df["dia_semana"].map(lambda x: dias_semana[x])

    # Se cuenta la cantidad_accidentes por día de la semana
    data = df.groupby("nombre_dia").agg({"num_victimas": "sum"}).reset_index()

    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 3))
    ax = sns.barplot(
        x="nombre_dia", y="num_victimas", data=data, order=dias_semana, hue="nombre_dia"
    )

    ax.set_title("Cantidad accidentes por día de la semana")
    ax.set_xlabel("Día de la Semana")
    ax.set_ylabel("Cantidad accidentes")
    plt.xticks(rotation=45)

    # Se muestran datos resumen
    print(
        f"El día de la semana con menor cantidad de víctimas tiene {data.min()[1]} víctimas"
    )
    print(
        f"El día de la semana con mayor cantidad de víctimas tiene {data.max()[1]} víctimas"
    )
    print(
        f"La diferencia porcentual es de {round((data.max()[1] - data.min()[1]) / data.min()[1] * 100,2)}"
    )

    # Se muestra el gráfico
    plt.show()


def crea_categoria_momento_dia(hora):
    """
    Devuelve la categoría de tiempo correspondiente a la hora proporcionada.

    Parameters:
      hora: La hora a clasificar.

    Returns:
      La categoría de tiempo correspondiente.
    """
    if hora.hour >= 6 and hora.hour <= 10:
        return "Mañana"
    elif hora.hour >= 11 and hora.hour <= 13:
        return "Medio día"
    elif hora.hour >= 14 and hora.hour <= 18:
        return "Tarde"
    elif hora.hour >= 19 and hora.hour <= 23:
        return "Noche"
    else:
        return "Madrugada"


def cantidad_accidentes_por_categoria_tiempo(df):
    """
    Calcula la cantidad_accidentes por categoría de tiempo y muestra un gráfico de barras.

    Esta función toma un DataFrame que contiene una columna 'hora' y utiliza la función
    'crea_categoria_momento_dia' para crear la columna 'categoria_tiempo'. Luego, cuenta
    la cantidad_accidentes por cada categoría de tiempo, calcula los porcentajes y
    genera un gráfico de barras que muestra la distribución de accidentes por categoría de tiempo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la información de los accidentes.

    Returns:
        None
    """
    # Se aplica la función crea_categoria_momento_dia para crear la columna 'categoria_tiempo'
    df["categoria_tiempo"] = df["hora"].apply(crea_categoria_momento_dia)

    # Se cuenta la cantidad_accidentes por categoría de tiempo
    data = df["categoria_tiempo"].value_counts().reset_index()
    data.columns = ["categoria_tiempo", "cantidad_accidentes"]

    # Se calculan los porcentajes
    total_accidentes = data["cantidad_accidentes"].sum()
    data["Porcentaje"] = (data["cantidad_accidentes"] / total_accidentes) * 100

    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x="categoria_tiempo", y="cantidad_accidentes", data=data, hue="categoria_tiempo"
    )

    ax.set_title("Cantidad accidentes por categoría de tiempo")
    ax.set_xlabel("Categoría de Tiempo")
    ax.set_ylabel("Cantidad accidentes")

    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(
            f'{row["cantidad_accidentes"]}',
            (index, row["cantidad_accidentes"]),
            ha="center",
            va="bottom",
        )

    # Se muestra el gráfico
    plt.show()


def cantidad_accidentes_por_horas_del_dia(df):
    """
    Genera un gráfico de barras que muestra la cantidad_accidentes por hora_del_dia.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    """
    # Se extrae la hora_del_dia de la columna 'hora'
    df["hora_del_dia"] = df["hora"].apply(lambda x: x.hour)

    # Se cuenta la cantidad_accidentes por hora_del_dia
    data = df["hora_del_dia"].value_counts().reset_index()
    data.columns = ["hora_del_dia", "cantidad_accidentes"]

    # Se ordena los datos por hora_del_dia
    data = data.sort_values(by="hora_del_dia")

    # Se crea el gráfico de barras
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        x="hora_del_dia", y="cantidad_accidentes", data=data, hue="cantidad_accidentes"
    )

    ax.set_title("Cantidad accidentes por hora del día")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Cantidad accidentes")

    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(
            f'{row["cantidad_accidentes"]}',
            (row["hora_del_dia"], row["cantidad_accidentes"]),
            ha="center",
            va="bottom",
        )

    # Se muestra el gráfico
    plt.show()


def cantidad_accidentes_semana_fin_de_semana(df):
    """
    Genera un gráfico de barras que muestra la cantidad_accidentes por tipo_dia (semana o fin_semana).

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    """
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df["dia_semana"] = df["fecha"].dt.dayofweek

    # Se crea una columna 'tipo_dia' para diferenciar entre semana y fin_semana
    df["tipo_dia"] = df["dia_semana"].apply(
        lambda x: "fin_semana" if x >= 5 else "Semana"
    )

    # Se cuenta la cantidad_accidentes por tipo_dia
    data = df["tipo_dia"].value_counts().reset_index()
    data.columns = ["tipo_dia", "cantidad_accidentes"]

    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="tipo_dia", y="cantidad_accidentes", data=data, hue="tipo_dia")

    ax.set_title("Cantidad accidentes por tipo de día")
    ax.set_xlabel("Tipo de día")
    ax.set_ylabel("Cantidad de accidentes")

    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(
            f'{row["cantidad_accidentes"]}',
            (index, row["cantidad_accidentes"]),
            ha="center",
            va="bottom",
        )

    # Se muestra el gráfico
    plt.show()


def distribucion_edad(df):
    """
    Genera un gráfico con un histograma y un boxplot que muestran la distribución de la edad de los involucrados en los accidentes.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico con un histograma y un boxplot.
    """
    # Se crea una figura con un solo eje x compartido
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Se grafica el histograma de la edad
    sns.histplot(df["edad"], kde=True, ax=ax[0])
    ax[0].set_title("Histograma de edad")
    ax[0].set_ylabel("Frecuencia")

    # Se grafica el boxplot de la edad
    sns.boxplot(x=df["edad"], ax=ax[1])
    ax[1].set_title("Boxplot de edad")
    ax[1].set_xlabel("edad")

    # Se ajusta y muestra el gráfico
    plt.tight_layout()
    plt.show()


def distribucion_edad_por_anio(df):
    """
    Genera un gráfico de boxplot que muestra la distribución de la edad de las víctimas de accidentes por año.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de boxplot.
    """
    # Se crea el gráfico de boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Año", y="edad", data=df)

    plt.title("Boxplot de edades de Víctimas por Año")
    plt.xlabel("Año")
    plt.ylabel("edad de las Víctimas")

    # Se muestra el gráfico
    plt.show()


def cantidades_accidentes_por_anio_y_sexo(df):
    """
    Genera un gráfico de barras que muestra la cantidad_accidentes por año y sexo.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    """
    # Se crea el gráfico de barras
    plt.figure(figsize=(12, 4))
    sns.barplot(
        x="anio",
        y="edad",
        hue="sexo",
        data=df,
    )

    plt.title("Cantidad de accidentes por año y sexo")
    plt.xlabel("Año")
    plt.ylabel("Edad de las víctimas")
    plt.legend(title="Sexo")

    # Se muestra el gráfico
    plt.show()


def cohen(group1, group2):
    """
    Calcula el tamaño del efecto de Cohen d para dos grupos.

    Parameters:
        grupo1: El primer grupo.
        grupo2: El segundo grupo.

    Returns:
        El tamaño del efecto de Cohen d.
    """
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = len(group1), len(group2)
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d


def cohen_por_año(df):
    """
    Calcula el tamaño del efecto de Cohen d para dos grupos para los años del Dataframe.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        El tamaño del efecto de Cohen d.
    """
    # Se obtienen los años del conjunto de datos
    años_unicos = df["Año"].unique()
    # Se crea una lista vacía para guardar los valores de Cohen
    cohen_lista = []
    # Se itera por los años y se guarda Cohen para cada grupo
    for a in años_unicos:
        grupo1 = df[((df["Sexo"] == "MASCULINO") & (df["Año"] == a))]["edad"]
        grupo2 = df[((df["Sexo"] == "FEMENINO") & (df["Año"] == a))]["edad"]
        d = cohen(grupo1, grupo2)
        cohen_lista.append(d)

    # Se crea un Dataframe
    cohen_df = pd.DataFrame()
    cohen_df["Año"] = años_unicos
    cohen_df["Estadistico de Cohen"] = cohen_lista
    cohen_df

    # Se grafica los valores de Cohen para los años
    plt.figure(figsize=(8, 4))
    plt.bar(cohen_df["Año"], cohen_df["Estadistico de Cohen"], color="skyblue")
    plt.xlabel("Año")
    plt.ylabel("Estadístico de Cohen")
    plt.title("Estadístico de Cohen por Año")
    plt.xticks(años_unicos)
    plt.show()


def edad_y_rol_victimas(df):
    """
    Genera un gráfico de la distribución de la edad de las víctimas por rol.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(y="Rol", x="edad", data=df)
    plt.title("edades por Condición")
    plt.show()


def distribucion_edad_por_victima(df):
    """
    Genera un gráfico de la distribución de la edad de las víctimas por tipo de vehículo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se crea el gráfico de boxplot
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="Víctima", y="edad", data=df)

    plt.title("Boxplot de edades de Víctimas por tipo de vehículo que usaba")
    plt.xlabel("Tipo de vehiculo")
    plt.ylabel("edad de las Víctimas")

    plt.show()


def cantidad_accidentes_sexo(df):
    """
    Genera un resumen de la cantidad_accidentes por sexo de los conductores.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad_accidentes por sexo de los conductores en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de accidentes por sexo de los conductores.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df["dia_semana"] = df["fecha"].dt.dayofweek

    # Se crea una columna 'tipo_dia' para diferenciar entre semana y fin_semana
    df["tipo_dia"] = df["dia_semana"].apply(
        lambda x: "fin_semana" if x >= 5 else "Semana"
    )

    # Se cuenta la cantidad_accidentes por tipo_dia
    data = df["tipo_dia"].value_counts().reset_index()
    data.columns = ["tipo_dia", "cantidad_accidentes"]

    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="tipo_dia", y="cantidad_accidentes", data=data)

    ax.set_title("cantidad_accidentes por tipo_dia")
    ax.set_xlabel("tipo_dia")
    ax.set_ylabel("cantidad_accidentes")

    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(
            f'{row["cantidad_accidentes"]}',
            (index, row["cantidad_accidentes"]),
            ha="center",
            va="bottom",
        )

    # Se muestra el gráfico
    plt.show()


def cantidad_victimas_sexo_rol_victima(df):
    """
    Genera un resumen de la cantidad de víctimas por sexo, rol y tipo de vehículo en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Gráficos de barras que muestran la cantidad de víctimas por sexo, rol y tipo de vehículo en orden descendente.
    * DataFrames que muestran la cantidad y el porcentaje de víctimas por sexo, rol y tipo de vehículo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se crea el gráfico
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gráfico 1: Sexo
    sns.countplot(data=df, x="sexo", ax=axes[0], hue="sexo")
    axes[0].set_title("Cantidad de víctimas por sexo")
    axes[0].set_ylabel("Cantidad de víctimas")

    # Se define una paleta de colores personalizada (invierte los colores)
    colores_por_defecto = sns.color_palette()
    colores_invertidos = [colores_por_defecto[1], colores_por_defecto[0]]

    # Gráfico 2: Rol
    df_rol = df.groupby(["rol", "sexo"]).size().unstack(fill_value=0)
    df_rol.plot(kind="bar", stacked=True, ax=axes[1], color=colores_invertidos)
    axes[1].set_title("Cantidad de víctimas por rol")
    axes[1].set_ylabel("Cantidad de víctimas")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend().set_visible(False)

    # Gráfico 3: Tipo de vehículo
    df_victima = df.groupby(["victima", "sexo"]).size().unstack(fill_value=0)
    df_victima.plot(kind="bar", stacked=True, ax=axes[2], color=colores_invertidos)
    axes[2].set_title("Cantidad de víctimas por tipo de vehículo")
    axes[2].set_ylabel("Cantidad de víctimas")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend().set_visible(False)

    # Se muestran los gráficos
    plt.show()

    # # Se calcula la cantidad de víctimas por sexo
    # sexo_counts = df['Sexo'].value_counts().reset_index()
    # sexo_counts.columns = ['Sexo', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por sexo
    # total_victimas_sexo = sexo_counts['Cantidad de víctimas'].sum()
    # sexo_counts['Porcentaje de víctimas'] = (sexo_counts['Cantidad de víctimas'] / total_victimas_sexo) * 100

    # # Se crea el DataFrame para sexo
    # df_sexo = pd.DataFrame(sexo_counts)
    # print('Resumen para Sexo:')
    # print(df_sexo)

    # # Se calcula la cantidad de víctimas por rol y sexo
    # df_rol = df.groupby(['Rol', 'Sexo']).size().unstack(fill_value=0)

    # # Se calcula el porcentaje de víctimas por rol y sexo
    # total_victimas_rol = df_rol.sum(axis=1)
    # df_rol_porcentaje = df_rol.divide(total_victimas_rol, axis=0) * 100

    # # Se renombra las columnas para el DataFrame de porcentaje
    # df_rol_porcentaje.columns = [f"Porcentaje de víctimas {col}" for col in df_rol_porcentaje.columns]

    # # Se combinan los DataFrames de cantidad y porcentaje
    # df_rol = pd.concat([df_rol, df_rol_porcentaje], axis=1)
    # print('Resumen para Rol:')
    # print(df_rol)

    # # Se calcula la cantidad de víctimas por tipo de vehículo
    # tipo_vehiculo_counts = df['Víctima'].value_counts().reset_index()
    # tipo_vehiculo_counts.columns = ['Tipo de Vehículo', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por tipo de vehículo
    # total_victimas = tipo_vehiculo_counts['Cantidad de víctimas'].sum()
    # tipo_vehiculo_counts['Porcentaje de víctimas'] = round((tipo_vehiculo_counts['Cantidad de víctimas'] / total_victimas) * 100,2)

    # # Se crea un DataFrame con la cantidad y porcentaje de víctimas por tipo de vehículo
    # df_tipo_vehiculo = pd.DataFrame(tipo_vehiculo_counts)
    # print('Resumen para Tipo de vehículo:')
    # print(df_tipo_vehiculo)

    # # Se calcula la cantidad de víctimas por tipo de vehículo y sexo
    # tipo_vehiculo_sexo_counts = df.groupby(['Víctima', 'Sexo']).size().unstack(fill_value=0).reset_index()
    # tipo_vehiculo_sexo_counts.columns = ['Tipo de Vehículo', 'Mujeres', 'Hombres']

    # # Se calcula la cantidad total de víctimas
    # total_victimas = tipo_vehiculo_sexo_counts[['Hombres', 'Mujeres']].sum(axis=1)

    # # se agregan las columnas de cantidad total y porcentaje
    # tipo_vehiculo_sexo_counts['Cantidad Total'] = total_victimas
    # tipo_vehiculo_sexo_counts['Porcentaje Hombres'] = (tipo_vehiculo_sexo_counts['Hombres'] / total_victimas) * 100
    # tipo_vehiculo_sexo_counts['Porcentaje Mujeres'] = (tipo_vehiculo_sexo_counts['Mujeres'] / total_victimas) * 100

    # # Se imprimen resumenes
    # print("Resumen de víctimas por tipo de vehículo y sexo:")
    # print(tipo_vehiculo_sexo_counts)


def cantidad_victimas_participantes(df):
    """
    Genera un resumen de la cantidad de víctimas por número de participantes en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de víctimas por número de participantes en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas por número de participantes.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se ordenan los datos por 'Participantes' en orden descendente por cantidad
    ordenado = df["participantes"].value_counts().reset_index()
    ordenado = ordenado.rename(columns={"cantidad": "participantes"})
    ordenado = ordenado.sort_values(by="count", ascending=False)

    plt.figure(figsize=(15, 4))

    # Se crea el gráfico de barras
    ax = sns.barplot(
        data=ordenado.head(15),
        x="participantes",
        y="count",
        order=ordenado["participantes"].head(15),
        hue="participantes",
    )
    ax.set_title("Cantidad de víctimas por participantes")
    ax.set_ylabel("Cantidad de víctimas")
    # Rotar las etiquetas del eje x a 45 grados
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

    # Se muestra el gráfico
    plt.show()

    # # Se calcula la cantidad de víctimas por participantes
    # participantes_counts = df['Participantes'].value_counts().reset_index()
    # participantes_counts.columns = ['Participantes', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por participantes
    # total_victimas = participantes_counts['Cantidad de víctimas'].sum()
    # participantes_counts['Porcentaje de víctimas'] = round((participantes_counts['Cantidad de víctimas'] / total_victimas) * 100,2)

    # # Se ordenan los datos por cantidad de víctimas en orden descendente
    # participantes_counts = participantes_counts.sort_values(by='Cantidad de víctimas', ascending=False)

    # # Se imprimen resumenes
    # print("Resumen de víctimas por participantes:")
    # print(participantes_counts)


def cantidad_acusados(df):
    """
    Genera un resumen de la cantidad de acusados en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de acusados en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de acusados.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se ordenan los datos por 'Participantes' en orden descendente por cantidad
    ordenado = df["acusado"].value_counts().reset_index()
    ordenado = ordenado.rename(columns={"Cantidad": "acusado"})
    ordenado = ordenado.sort_values(by="count", ascending=False)

    plt.figure(figsize=(15, 4))

    # Crear el gráfico de barras
    ax = sns.barplot(
        data=ordenado, x="acusado", y="count", order=ordenado["acusado"], hue="acusado"
    )
    ax.set_title("Cantidad de acusados en los hechos")
    ax.set_ylabel("Cantidad de acusados")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

    # Se muestra el gráfico
    plt.show()

    # # Se calcula la cantidad de acusados
    # acusados_counts = df['acusado'].value_counts().reset_index()
    # acusados_counts.columns = ['acusado', 'Cantidad de acusados']

    # # Se calcula el porcentaje de acusados
    # total_acusados = acusados_counts['Cantidad de acusados'].sum()
    # acusados_counts['Porcentaje de acusados'] = round((acusados_counts['Cantidad de acusados'] / total_acusados) * 100,2)

    # # Se ordenan los datos por cantidad de acusados en orden descendente
    # acusados_counts = acusados_counts.sort_values(by='Cantidad de acusados', ascending=False)
    # # Se imprimen resumen
    # print("Resumen de acusados:")
    # print(acusados_counts)


def accidentes_tipo_de_calle(df):
    """
    Genera un resumen de los accidentes de tráfico por tipo de calle y cruce.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de víctimas por tipo de calle.
    * Un gráfico de barras que muestra la cantidad de víctimas en cruces.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas por tipo de calle.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas en cruces.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se crea el gráfico
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=df, x="tipo_calle", ax=axes[0], hue="tipo_calle")
    axes[0].set_title("Cantidad de víctimas por tipo de calle")
    axes[0].set_ylabel("Cantidad de víctimas")

    sns.countplot(data=df, x="cruce", ax=axes[1], hue="tipo_calle")
    axes[1].set_title("Cantidad de víctimas en cruces")
    axes[1].set_ylabel("Cantidad de víctimas")

    # Mostramos los gráficos
    plt.show()

    # # Se calcula la cantidad de víctimas por tipo de calle
    # tipo_calle_counts = df['Tipo de calle'].value_counts().reset_index()
    # tipo_calle_counts.columns = ['Tipo de calle', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por tipo de calle
    # tipo_calle_counts['Porcentaje de víctimas'] = round((tipo_calle_counts['Cantidad de víctimas'] / tipo_calle_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se calcula la cantidad de víctimas por cruce
    # cruce_counts = df['Cruce'].value_counts().reset_index()
    # cruce_counts.columns = ['Cruce', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por cruce
    # cruce_counts['Porcentaje de víctimas'] = round((cruce_counts['Cantidad de víctimas'] / cruce_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se crean DataFrames para tipo de calle y cruce
    # df_tipo_calle = pd.DataFrame(tipo_calle_counts)
    # df_cruce = pd.DataFrame(cruce_counts)

    # #  Se muestran los DataFrames resultantes
    # print("Resumen por Tipo de Calle:")
    # print(df_tipo_calle)
    # print("\nResumen por Cruce:")
    # print(df_cruce)


def graficos_eda_categoricos(cat):
    """
    Realiza gráficos de barras horizontales para explorar datos categóricos.

    Parámetros:
    - cat (DataFrame): DataFrame que contiene variables categóricas a visualizar.

    Retorna:
    - None: La función solo genera gráficos y no devuelve valores.

    La función toma un DataFrame con variables categóricas y genera gráficos de barras horizontales
    para visualizar la distribución de categorías en cada variable. Los gráficos se organizan en
    filas y columnas para facilitar la visualización.
    """
    # Calculamos el número de filas que necesitamos
    from math import ceil

    filas = ceil(cat.shape[1] / 2)

    # Definimos el gráfico
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))

    # Aplanamos para iterar por el gráfico como si fuera de 1 dimensión en lugar de 2
    ax = ax.flat

    # Creamos el bucle que va añadiendo gráficos
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=12, fontweight="bold")
        ax[cada].tick_params(labelsize=12)


def estadisticos_cont(num):
    """
    Calcula estadísticas descriptivas para variables numéricas.

    Parámetros:
    - num (DataFrame o Series): Datos numéricos para los cuales se desean calcular estadísticas.

    Retorna:
    - DataFrame: Un DataFrame que contiene estadísticas descriptivas, incluyendo la media, la desviación estándar,
      los percentiles, el mínimo, el máximo y la mediana.

    La función toma datos numéricos y calcula estadísticas descriptivas, incluyendo la media, desviación estándar,
    percentiles (25%, 50%, 75%), mínimo, máximo y mediana. Los resultados se presentan en un DataFrame organizado
    para una fácil interpretación.

    Nota:
    - El DataFrame de entrada debe contener solo variables numéricas para obtener resultados significativos.
    """
    # Calculamos describe
    estadisticos = num.describe().T
    # Añadimos la mediana
    estadisticos["median"] = num.median()
    # Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]
    # Lo devolvemos
    return estadisticos

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder
import os

user_dir = "..."

# Cargar múltiples fuentes de datos
cliente = pd.read_csv(os.path.join(user_dir, "1 Informacion Sociodemografica.csv"))
econ = pd.read_csv(os.path.join(user_dir, "2. Información Económica.csv"))
buro = pd.read_excel(os.path.join(user_dir, "3. Diccionario Tabla Información Buro de crédito.xlsx"))
prod = pd.read_csv(os.path.join(user_dir, '4. Informacion de Producto.txt'), delimiter='|')
com = pd.read_csv(os.path.join(user_dir, '5. Informacion Comportamiento.txt'), delimiter='|')
pago = pd.read_csv(os.path.join(user_dir, '6 Informacion de Pagos.csv')) 
ree = pd.read_csv(os.path.join(user_dir, '8. Informacion de Reestructuraciones (Por producto).txt'), delimiter='|')
cas = pd.read_csv(os.path.join(user_dir, '9. Informacion Castigos2.txt'), delimiter='|')
ges = pd.read_csv(os.path.join(user_dir, '10 Gestiones.txt'), delimiter='|')

# Funciones Utilizadas


def convertir_columnas_fecha(df):
    """
    Convierte todas las columnas cuyo nombre comienza con 'FECHA' (insensible a mayúsculas)
    a tipo datetime.
    """
    for col in df.columns:
        if col.upper().startswith("FECHA"):
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                print(f"Columna convertida: {col}")
            except Exception as e:
                print(f"No se pudo convertir {col}: {e}")
    return df


def analizar_serie_temporal(df, fecha_col, valor_col, freq='M'):
    """
    Realiza una descomposición temporal para detectar tendencia y estacionalidad.
    """
    df = df[[fecha_col, valor_col]].dropna()
    df = df.set_index(fecha_col).resample(freq).sum()

    descomposicion = seasonal_decompose(df, model='additive')
    descomposicion.plot()
    plt.suptitle(f'{fecha_col} - Descomposición temporal de {valor_col}')
    plt.tight_layout()
    plt.show()


def correlaciones(df, columnas_numericas):
    """
    Genera un mapa de calor de correlaciones entre variables numéricas.
    """
    corr = df[columnas_numericas].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 8}, fmt=".2f")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Correlación entre variables", fontsize=12)
    plt.tight_layout()
    plt.show()


def convertir_a_numerico_hot_n(df):
    """
    Convierte variables categóricas en numéricas usando One-Hot Encoding.
    """
    columnas_no_numericas = df.select_dtypes(include=['object', 'string', 'category']).columns
    df_codificado = pd.get_dummies(df, columns=columnas_no_numericas, drop_first=True, dtype=int)
    return df_codificado



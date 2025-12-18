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


# Ejemplo de uso

# Información general
cliente.info()

# Corrección de tipo de dato fechas
convertir_columnas_fecha(cliente)

# Cálculo de edad
cliente['Edad'] = pd.to_datetime('today').year - cliente['FECHA NACIMIENTO'].dt.year

# Gráfico de distribución de sexo
sns.countplot(x='SEXO', data=cliente)
plt.title('Distribución por Sexo')
plt.show()

# Histograma de edad
cliente['Edad'].hist(bins=90)
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.xlim(0, 100)
plt.ylabel('Frecuencia')
plt.show()

# Histograma de altura mora
com['ALTURA DE MORA'].hist(bins=50)
plt.title('Distribución de Mora')
plt.xlabel('Mora')
plt.xlim(0, 200)
plt.ylabel('Frecuencia')
plt.show()

#VISUALIZACION DE OUTLIERS
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.suptitle('Boxplots: Outliers en variables sociodemográficas')

sns.boxplot(y=cliente['NEGOCIO'], ax=axes[0,0])
sns.boxplot(y=cliente['DEPENDIENTES ECONOMICOS'], ax=axes[0,1])
sns.boxplot(y=cliente['TIEMPO VIVIENDA ANIOS'], ax=axes[1,0])
sns.boxplot(y=cliente['TIEMPO VIVIENDA MESES'], ax=axes[1,1])

axes[1,0].set_ylabel("TIEMPO VIVIENDA AÑOS")
plt.tight_layout()
plt.show()


# Serie temporal

# Asegurar que las fechas estén en datetime
#cliente['FECHA VINCULACION'] = pd.to_datetime(cliente['FECHA VINCULACION'], dayfirst=True, errors='coerce')

# Crear columna dummy para contar afiliaciones
#cliente['FOLADE'] = 1

# Analizar serie temporal de vinculaciones
analizar_serie_temporal(cliente, 'FECHA VINCULACION', 'FOLADE', freq='M')

# Identificación de patrones con correlación

convertir_columnas_fecha(ree)

# Ver columnas para analizar correlación
columnas_correlacion = ree.columns.tolist()

# Mostrar matriz de correlación
correlaciones(ree, columnas_correlacion)

#buro.head()
buro = buro.drop(['CUENTA', 'Unnamed: 4', 'Unnamed: 5', 'CUENTA.1'], axis=1)

# En la base gestion hay espacios en los nombre de las columnas
#ges['          FECHA CORTE']
ges.columns = ges.columns.str.strip()

print('Clientes \n', cliente.columns)
print('\nEconomía \n', econ.columns)
print('\nBuró \n', buro.columns)
print('\nProducto \n', prod.columns)
print('\nComportamiento \n', com.columns)
print('\nPagos \n', pago.columns)
print('\nReestructuras \n', ree.columns)
print('\nCastigos \n', cas.columns)
print('\nGestiones \n', ges.columns)

Clientes 
 Index(['FOLADE', 'NEGOCIO', 'TIPO DE PERSONA', 'FECHA VINCULACION',
       'ESTADO CLIENTE', 'FECHA NACIMIENTO', 'SEXO', 'PROFESION',
       'ESTADO CIVIL', 'CARGO', 'NIVEL ACADEMICO', 'DEPENDIENTES ECONOMICOS',
       'TIPO VIVIENDA', 'TIEMPO VIVIENDA ANIOS', 'TIEMPO VIVIENDA MESES',
       'ESTADO RESIDENCIA', 'MUNICIPIO RESIDENCIA', 'ULTIMO MOVIMIENTO',
       'Edad'],
      dtype='object')

#Economía 
# Index(['FOLADE', 'NEGOCIO', 'ANTIGUEDAD ANIOS', 'ANTIGUEDAD MESES', 'SALARIO', 'ACTIVIDAD ECONOMICA', 'ULTIMO MOVIMIENTO'], dtype='object')
#Buró 
# Index(['FOLADE', 'NEGOCIO', 'FECHA CASTIGO'], dtype='object')
#Producto 
# Index(['FOLADE', 'NEGOCIO', 'OFICINA', 'ESTADO CLIENTE', 'FECHA AUTORIZACION', 'FECHA CANCELACION', 'FECHA SUSPENSION', 'PLAZO PROMEDIO (MESES)', 'FORMATO DEL NEGOCIO', 'CAPACIDAD DE PAGO', 'TOPE POR LINEA', 'FECHA PRIMER COMPRA', 'CARGOS MENSUAL', 'CARGOS QUINCENAL', 'CARGOS SEMANAL'], dtype='object')
#Comportamiento 
# Index(['FOLADE', 'NEGOCIO', 'FECHA CORTE BASE', 'ESTADO CARTERA', 'ALTURA DE MORA', 'SALDO TOTAL', 'SALDO CAPITAL', 'VALOR CUOTA', 'FECHA PROX VTO', 'FECHA ULTIMO PAGO', 'FECHA REEST', 'FECHA REFIN'], dtype='object')
#Pagos 
# Index(['FOLADE', 'NEGOCIO', 'MES_PAGO', 'PAGO_TOTAL'], dtype='object')
#Reestructuras 
# Index(['FOLADE', 'NEGOCIO', 'TIPO DE REESTRUCTURACION', 'FECHA REESTRUCTURACION'], dtype='object')
#Castigos 
# Index(['FOLADE', 'NEGOCIO', 'FECHA CASTIGO'], dtype='object')
#Gestiones 
# Index(['FOLADE', 'NEGOCIO', 'ID PERSONA', 'FECHA CORTE', 'TIPO GESTION', 'RESULTADO'], dtype='object')


# Idenficifación de Varialbes y Modelo
----------------------------------------------------
# De acuerdo a las reglas del negocio, el indicador Buenos y Malos se convierte en la variable que se va a pronosticar y la que el score pretende explicar.

# * Buenos: Obligaciones que Mantienen o Mejoran su tramo de mora y que no son Reestructurados, Castigos o Adjudicados al mes siguiente.

# * Malos: Obligaciones que Incrementan su tramo de mora o que se Mantienen en mora > a 120 días o que son, Reestructurados, Castigos o Adjudicados al mes siguiente


# Definición de variable objetivo
----------------------------------------------------

def generar_target(comportamiento, reestructuras, castigos):
    # 1. Preparación de fechas
    # -----------------------------
    comportamiento['FECHA CORTE BASE'] = pd.to_datetime(comportamiento['FECHA CORTE BASE'])
    reestructuras['FECHA REESTRUCTURACION'] = pd.to_datetime(reestructuras['FECHA REESTRUCTURACION'])
    castigos['FECHA CASTIGO'] = pd.to_datetime(castigos['FECHA CASTIGO'])

    # ID único
    for df in [comportamiento, reestructuras, castigos]:
        df['ID'] = df['FOLADE'].astype(str) + '_' + df['NEGOCIO'].astype(str)

    # 2. Orden temporal y mora t+1
    # -----------------------------
    comportamiento = comportamiento.sort_values(['ID', 'FECHA CORTE BASE'])

    comportamiento['MORA_T1'] = (comportamiento.groupby('ID')['ALTURA DE MORA'].shift(-1))

    comportamiento['MES_T'] = comportamiento['FECHA CORTE BASE']
    comportamiento['MES_T1'] = comportamiento.groupby('ID')['MES_T'].shift(-1)

    # 3. Target por mora
    # -----------------------------
    comportamiento['target_mora'] = (
        (comportamiento['MORA_T1'] > comportamiento['ALTURA DE MORA']) |
        (comportamiento['MORA_T1'] >= 120)
    ).astype(int)

    # Quitamos último mes (no tiene t+1)
    comportamiento = comportamiento.dropna(subset=['MES_T1'])

    # 4. Eventos en ventana t+1
    # -----------------------------
    # Marcamos evento si ocurre entre MES_T y MES_T1
    def evento_en_t1(eventos, fecha_evento, df_base):
        eventos = eventos[['ID', fecha_evento]].copy()
        eventos['flag'] = 1

        merged = df_base.merge(eventos, on='ID', how='left')

        return (
            (merged[fecha_evento] > merged['MES_T']) & (merged[fecha_evento] <= merged['MES_T1'])
        ).astype(int)

    comportamiento['flag_reest'] = evento_en_t1(reestructuras, 'FECHA REESTRUCTURACION', comportamiento)

    comportamiento['flag_cast'] = evento_en_t1(castigos, 'FECHA CASTIGO', comportamiento)

    # 5. Target final
    # -----------------------------
    comportamiento['target'] = (comportamiento[['target_mora', 'flag_reest', 'flag_cast']].max(axis=1))

    return comportamiento[['ID', 'MES_T', 'target']]


target_final = generar_target(com, ree, cas)


# función para filtrar cualquier base por fecha <= t
def merge_features(features_df, date_col, target_df):
    features_df[date_col] = pd.to_datetime(features_df[date_col])

    return target_df.merge(
        features_df,
        how="left",
        on="ID"
    ).query(f"{date_col} <= MES_T")  # aseguramos temporalidad



# Unificacion de fuentes de datos
----------------------------------------------------

df = target_final.copy()

cliente['ID'] = cliente['FOLADE'].astype(str) + '_' + cliente['NEGOCIO'].astype(str)

# agregar cliente (no temporal)
df = df.merge(cliente.drop(columns=['FOLADE','NEGOCIO']), on='ID', how='left')

prod['ID'] = prod['FOLADE'].astype(str) + '_' + prod['NEGOCIO'].astype(str)
# producto (no cambia con el tiempo)
df = df.merge(prod.drop(columns=['FOLADE','NEGOCIO']), on='ID', how='left')

econ['ID'] = econ['FOLADE'].astype(str) + '_' + econ['NEGOCIO'].astype(str)
econ = econ.drop(columns=['FOLADE','NEGOCIO'])
# variables económicas por fecha
df = df.merge(econ, left_on='ID', right_on='ID', how='left')

buro['ID'] = buro['FOLADE'].astype(str) + '_' + buro['NEGOCIO'].astype(str)
buro = buro.drop(columns=['FOLADE','NEGOCIO'])
# buró – debe contener datos del mes t
df = df.merge(buro, left_on='ID', right_on='ID', how='left')

# pagos filtrados a t
pago['ID'] = pago['FOLADE'].astype(str) + '_' + pago['NEGOCIO'].astype(str)
pago['MES_PAGO'] = pd.to_datetime(pago['MES_PAGO'])
pago_t = pago.groupby(['ID', pago['MES_PAGO'].dt.to_period('M')])['PAGO_TOTAL'].sum().reset_index()
pago_t['MES'] = pago_t['MES_PAGO'].dt.to_timestamp()

df = df.merge(pago_t, left_on=['ID','MES_T'], right_on=['ID','MES'], how='left')

# gestiones filtradas
ges['ID'] = ges['FOLADE'].astype(str) + '_' + ges['NEGOCIO'].astype(str)
ges['FECHA CORTE'] = pd.to_datetime(ges['FECHA CORTE'])
ges_t = ges[ges['FECHA CORTE'] <= df['MES_T'].max()]  # filtrado general

gestiones_agg = ges_t.groupby('ID')['RESULTADO'].agg(lambda x: x.value_counts().idxmax()).reset_index()
df = df.merge(gestiones_agg, on='ID', how='left')

import pandas as pd

# Cargar el archivo Excel
df = pd.read_excel('COE BRAZIL_2023.xlsx')

# Mostrar las primeras 5 filas para verificar los datos
print(df.info())

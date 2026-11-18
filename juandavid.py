# 1. Importamos las herramientas que vamos a usar
import pandas as pd  # Para hacer tablas
import numpy as np  # Para hacer números y cálculos
import matplotlib.pyplot as plt  # Para hacer gráficas
from mpl_toolkits.mplot3d import Axes3D  # Para hacer la gráfica en 3D
from sklearn.model_selection import train_test_split  # Para dividir los datos
from sklearn.linear_model import LinearRegression  # Para crear el modelo
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Para medir qué tan bien predice

# 2. Ponemos una semilla para que siempre salgan los mismos datos
np.random.seed(42)
n = 300  # Vamos a crear 300 datos inventados

# 3. Creamos los datos inventados
publicidad = np.random.uniform(100, 1000, n)  # Gasto en publicidad entre 100 y 1000
precio = np.random.uniform(10, 100, n)  # Precio entre 10 y 100
clima = np.random.uniform(0, 40, n)  # Temperatura entre 0 y 40 grados

# 4. Creamos las ventas con una fórmula
# Si sube la publicidad y el clima, suben las ventas. Si sube el precio, bajan.
ventas = (
    50
    + 0.8 * publicidad
    - 2.5 * precio
    + 3 * clima
    + np.random.normal(0, 50, n)  # Le agregamos un poco de ruido para que se vea real
)

# 5. Hacemos una tabla con todos los datos
df = pd.DataFrame({
    "publicidad": publicidad,
    "precio": precio,
    "clima": clima,
    "ventas": ventas
})

# 6. Separamos lo que va a adivinar y lo que queremos adivinar
X = df[["publicidad", "precio", "clima"]]  # Esto es lo que ya sabemos
y = df["ventas"]  # Esto es lo que queremos predecir

# 7. Dividimos los datos: 80% para enseñar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Creamos el modelo y lo entrenamos
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 9. Hacemos predicciones con los datos de prueba
y_pred = modelo.predict(X_test)

# 10. Medimos qué tan bien predijo
r2 = r2_score(y_test, y_pred)  # Qué tan bien explica las ventas
mae = mean_absolute_error(y_test, y_pred)  # Error promedio
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Otro error, pero con raíz cuadrada

# 11. Mostramos los resultados en la pantalla
print("Intercepto:", modelo.intercept_)
print("Coeficientes:", modelo.coef_)
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# 12. Hacemos una predicción nueva
# ¿Cuánto venderemos si gastamos 500 en publicidad, precio 40 y clima 25?
nuevo = [[500, 40, 25]]
prediccion = modelo.predict(nuevo)
print("Predicción para publicidad=500, precio=40, clima=25:", prediccion)

# 13. Hacemos la gráfica en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ponemos los puntitos: publicidad, precio y ventas, con color según el clima
ax.scatter(df["publicidad"], df["precio"], df["ventas"], c=df["clima"], cmap='viridis')

# 14. Le ponemos nombre a cada eje y título
ax.set_xlabel("Publicidad")
ax.set_ylabel("Precio")
ax.set_zlabel("Ventas")
ax.set_title("Regresión múltiple: Publicidad, Precio y Ventas")

# 15. Mostramos la gráfica
plt.show()
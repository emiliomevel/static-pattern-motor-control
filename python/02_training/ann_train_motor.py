import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
NOMBRE_DATASET = 'dataset_auto_etiquetado.csv'

print("Cargando dataset...")
df = pd.read_csv('dataset_auto_etiquetado.csv')

X = df[['P1', 'P2', 'P3']].values  # 3 Entradas
y = df['Label'].values             # Clases (0 a 4)

# División 90% Entrenamiento / 10% Prueba (Stratified para mantener proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

print(f"Total de muestras: {len(df)}")
print(f"Muestras de Entrenamiento (90%): {len(X_train)}")
print(f"Muestras de Prueba (10%): {len(X_test)}")

# Guardar el 10% de Test en un CSV independiente para validaciones futuras
df_test = pd.DataFrame(X_test, columns=['P1', 'P2', 'P3'])
df_test['Label'] = y_test
df_test.to_csv('dataset_test_10percent.csv', index=False)
print("-> Archivo 'dataset_test_10percent.csv' guardado para pruebas finales.")

# ==========================================
# 2. ENTRENAMIENTO DE LA RED NEURONAL (3-8-5)
# ==========================================
# Archivo de red: 1 capa oculta con 8 neuronas
# Función de activación: Relu o Logistic (Sigmoide). Para micros se prefiere ReLU o Logistic.
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),    # 8 neuronas en la capa oculta
    activation='relu',          # Activación 'relu' (rápida de evaluar en C: max(0, x))
    solver='adam',
    max_iter=1000,
    random_state=42
)

print("\nEntrenando Perceptrón Multicapa 3-8-5...")
mlp.fit(X_train, y_train)

# ==========================================
# 3. EVALUACIÓN Y VERIFICACIÓN DE ACCURACY
# ==========================================
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train) * 100
test_acc = accuracy_score(y_test, y_pred_test) * 100

print("\n" + "="*50)
print(f"RESULTADOS DE PRECISIÓN (ACCURACY):")
print(f"Accuracy Entrenamiento (90%): {train_acc:.2f}%")
print(f"Accuracy Prueba/Test   (10%): {test_acc:.2f}%")
print("="*50)

print("\nReporte detallado sobre el conjunto de prueba (10%):")
print(classification_report(y_test, y_pred_test))

# ==========================================
# 4. EXTRAER PESOS Y GENERAR CÓDIGO C
# ==========================================
# Weights & Biases
W1 = mlp.coefs_[0]       # Matriz (3, 8) -> Entradas a Capa Oculta
b1 = mlp.intercepts_[0]  # Vector (8,)  -> Bias Capa Oculta
W2 = mlp.coefs_[1]       # Matriz (8, 5) -> Capa Oculta a Salidas
b2 = mlp.intercepts_[1]  # Vector (5,)  -> Bias Capa de Salida

def format_c_matrix_2d(arr, name):
    rows, cols = arr.shape
    out = f"const float {name}[{rows}][{cols}] = {{\n"
    for r in range(rows):
        vals = ", ".join([f"{v:.8f}f" for v in arr[r]])
        out += f"    {{ {vals} }},\n"
    out += "};\n"
    return out

def format_c_vector_1d(arr, name):
    vals = ", ".join([f"{v:.8f}f" for v in arr])
    return f"const float {name}[{len(arr)}] = {{ {vals} }};\n"

# Escribir encabezado con parámetros de C para el microcontrolador
with open("nn_weights.h", "w") as f:
    f.write("/* PESOS GENERADOS AUTOMÁTICAMENTE PARA S32K3 (3-8-5 MLP) */\n")
    f.write("#ifndef NN_WEIGHTS_H\n#define NN_WEIGHTS_H\n\n")
    f.write("#define NN_NUM_INPUTS   3\n")
    f.write("#define NN_NUM_HIDDEN   8\n")
    f.write("#define NN_NUM_OUTPUTS  5\n\n")
    
    f.write(format_c_matrix_2d(W1, "W1"))
    f.write("\n")
    f.write(format_c_vector_1d(b1, "b1"))
    f.write("\n")
    f.write(format_c_matrix_2d(W2, "W2"))
    f.write("\n")
    f.write(format_c_vector_1d(b2, "b2"))
    f.write("\n#endif /* NN_WEIGHTS_H */\n")

print("\n-> Pesos exportados a 'nn_weights.h' para S32 Design Studio.")

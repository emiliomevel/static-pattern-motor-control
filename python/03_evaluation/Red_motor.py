import argparse

import serial

import threading

import time

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)



# ==============================================================================

# 0. PARSER DE ARGUMENTOS DE LÍNEA DE COMANDOS

# ==============================================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description="Monitoreo, Evaluación de Red Neuronal y Sintonia PID para S32K3."

    )

    parser.add_argument(

        '-p', '--port',

        type=str,

        default='COM10',

        help='Puerto Serie / COM (Ejemplo: COM3, COM10, /dev/ttyUSB0). Defecto: COM10'

    )

    parser.add_argument(

        '-b', '--baud',

        type=int,

        default=115200,

        help='Velocidad en Baudios (Baudrate). Defecto: 115200'

    )

    parser.add_argument(

        '--kp',

        type=float,

        default=None,

        help='Valor inicial opcional para Kp a enviar al microcontrolador al iniciar'

    )

    parser.add_argument(

        '--ki',

        type=float,

        default=None,

        help='Valor inicial opcional para Ki a enviar al microcontrolador al iniciar'

    )

    parser.add_argument(

        '--kd',

        type=float,

        default=None,

        help='Valor inicial opcional para Kd a enviar al microcontrolador al iniciar'

    )

    return parser.parse_args()





# ==============================================================================

# 1. CARGA DIRECTA DE PESOS Y SESGOS EXACTOS DEL MICRO

# ==============================================================================

# W1: Capa Oculta (3 Entradas -> 8 Neuronas)

W1 = np.array([

    [0.00000000,  4.03068296, -0.00000000,  4.67590750, -3.79031800, -0.00000000, -0.54096510,  1.87561512],

    [-0.00000000,  1.98182998,  0.00000000,  0.56315710,  2.43200779,  0.00000000,  1.42707623, -5.75779739],

    [-0.00000000,  3.68275536, -0.00000000,  1.47030866,  3.24881000, -0.00000000, -3.51267466,  3.46091002]

], dtype=np.float64)



# b1: Sesgos Capa Oculta (8 Neuronas)

b1 = np.array([

    -0.06488893,  1.76804471, -0.44361122, -2.89729349,  0.00543361, -0.66993714,  2.85214470,  0.36043297

], dtype=np.float64)



# W2: Capa de Salida (8 Neuronas -> 5 Clases)

W2 = np.array([

    [ 0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000],

    [-2.31064443,  0.65057739,  2.61580883, -2.21336280,  2.09334442],

    [ 0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000],

    [ 3.95035668,  3.33042002, -10.69340320, 2.17670304, -10.00872100],

    [ 7.13225767, -0.11062631, -3.15351499, -7.27677681, -7.48868249],

    [ 0.00000000,  0.00000000, -0.00000000, -0.00000000,  0.00000000],

    [ 7.72773990, -8.78040805, -0.32547316, -16.15821000, -0.62903848],

    [ 2.18826102, -13.99928098, -8.94865388, 4.88262582, -1.93829505]

], dtype=np.float64)



# b2: Sesgos Capa de Salida (5 Clases)

b2 = np.array([

    -1.29631745,  1.12822379,  1.73935804, -1.33033890,  0.73743275

], dtype=np.float64)



def relu(x):

    """Función de activación ReLU"""

    return np.maximum(0, x)



def predict_mlp_exact(p1, p2, p3):

    """ Inferencia MLP exacta ejecutada en Python usando los mismos pesos """

    X_in = np.array([p1, p2, p3], dtype=np.float64)

   

    # Capa Oculta (Entrada @ W1 + b1) -> ReLU

    z1 = np.dot(X_in, W1) + b1

    a1 = relu(z1)

   

    # Capa de Salida (a1 @ W2 + b2)

    z2 = np.dot(a1, W2) + b2

   

    # Inferencia Argmax

    predicted_class = np.argmax(z2)

    return int(predicted_class)





# ==============================================================================

# 2. CONFIGURACIÓN SERIAL Y ESTRUCTURAS DE DATOS

# ==============================================================================

args = parse_args()



SERIAL_PORT = args.port

BAUD_RATE = args.baud



CLASS_NAMES = ["STOP", "Avance Rapido", "Avance Lento", "Reversa Rapida", "Reversa Lenta"]



# Listas globales de almacenamiento

timestamps = []

y_pc_preds = []

y_micro_preds = []

targets_user = []

targets_filt = []

current_rpms = []

pwms = []

kps, kis, kds = [], [], []



# Variables de evaluación dinámica

last_class_micro = -1

t_change_detected = 0.0

latencies = []

start_time = time.time()

running = True



try:

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    print(f"Conectado exitosamente a {SERIAL_PORT} @ {BAUD_RATE} bps")

except Exception as e:

    print(f"Error abriendo puerto serial {SERIAL_PORT}: {e}")

    sys.exit(1)





def send_pid_param(param_name, value):

    """Envía comandos PID usando \r\n para forzar la interrupción del micro"""

    cmd_str = f"{param_name}:{value}\r\n"

    ser.reset_output_buffer()

    ser.write(cmd_str.encode('utf-8'))

    ser.flush()

    time.sleep(0.08)

    sys.stdout.write(f"\n---> [ENVIADO A MICRO]: {param_name}:{value}\n")

    sys.stdout.flush()





# Envío de parámetros iniciales de PID si fueron especificados en los argumentos

if args.kp is not None:

    send_pid_param("KP", args.kp)

if args.ki is not None:

    send_pid_param("KI", args.ki)

if args.kd is not None:

    send_pid_param("KD", args.kd)





# ==============================================================================

# 3. HILO SECUNDARIO DE RECEPCIÓN Y EVALUACIÓN

# ==============================================================================

def read_telemetry():

    global last_class_micro, t_change_detected, running

   

    while running and ser.is_open:

        try:

            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:

                continue



            parts = line.split(',')

            # Trama esperada del Micro (11 valores):

            # P1, P2, P3, Class_Micro, Target_User, Target_Filt, Measured_RPM, PWM, Kp, Ki, Kd

            if len(parts) == 11:

                t_now = time.time() - start_time

                p1, p2, p3 = float(parts[0]), float(parts[1]), float(parts[2])

                clase_micro = int(parts[3])

                target_user = float(parts[4])

                target_filt = float(parts[5])

                current_rpm = float(parts[6])

                pwm = float(parts[7])

                kp_val, ki_val, kd_val = float(parts[8]), float(parts[9]), float(parts[10])



                # 1. Inferencia exacta en PC con los pesos importados

                clase_pc = predict_mlp_exact(p1, p2, p3)



                # 2. Medición de Latencia Detección -> Acción

                if clase_micro != last_class_micro:

                    if last_class_micro != -1:

                        latencia_ms = (t_now - t_change_detected) * 1000.0

                        latencies.append(latencia_ms)

                    t_change_detected = t_now

                    last_class_micro = clase_micro



                # 3. Guardar datos

                timestamps.append(t_now)

                y_pc_preds.append(clase_pc)

                y_micro_preds.append(clase_micro)

                targets_user.append(target_user)

                targets_filt.append(target_filt)

                current_rpms.append(current_rpm)

                pwms.append(pwm)

                kps.append(kp_val); kis.append(ki_val); kds.append(kd_val)



                # 4. Impresión continua limpia usando sys.stdout

                sys.stdout.write(

                    f"\rP1:{p1:.2f} P2:{p2:.2f} P3:{p3:.2f} | "

                    f"Clase PC:[{clase_pc}] Micro:[{clase_micro}] | "

                    f"Ref:{target_user:5.1f} Med:{current_rpm:5.1f} RPM | "

                    f"PWM:{pwm:5.1f} | Kp:{kp_val:.2f} Ki:{ki_val:.2f} Kd:{kd_val:.2f}    "

                )

                sys.stdout.flush()



        except Exception:

            break



# Iniciar hilo de captura

thread = threading.Thread(target=read_telemetry, daemon=True)

thread.start()





# ==============================================================================

# 4. INTERFAZ INTERACTIVA PARA SINTONÍA PID (EXCLUSIVO MICROCONTROLADOR)

# ==============================================================================

print("\n--- MONITOREO Y EVALUACIÓN S32K3 EN TIEMPO REAL ---")

print("Comandos para enviar constantes PID al Microcontrolador:")

print("  KP <valor>  (Ejemplo: KP 3.5)")

print("  KI <valor>  (Ejemplo: KI 0.8)")

print("  KD <valor>  (Ejemplo: KD 0.05)")

print("  EXIT        (Escribe EXIT para terminar y ver las gráficas)\n")



try:

    while True:

        user_input = input().strip().upper()

        if user_input == 'EXIT':

            break



        parts = user_input.split()

        if len(parts) == 2 and parts[0] in ['KP', 'KI', 'KD']:

            send_pid_param(parts[0], parts[1])

        else:

            print("\nFormato inválido. Usa: KP <val>, KI <val>, KD <val>")



except KeyboardInterrupt:

    pass



finally:

    running = False

    ser.close()

    print("\n\nConexión cerrada. Procesando métricas avanzadas...")





# ==============================================================================

# 5. REPORTE COMPLETO: CLASIFICACIÓN, LATENCIA Y ESTABILIDAD

# ==============================================================================

if len(y_pc_preds) > 0:

    y_true = np.array(y_pc_preds)

    y_pred = np.array(y_micro_preds)



    # --------------------------------------------------------------------------

    # A. Métricas de Clasificación

    # --------------------------------------------------------------------------

    acc = accuracy_score(y_true, y_pred) * 100.0

    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)



    print("\n" + "="*70)

    print("1. EVALUACIÓN DE CLASIFICACIÓN (PC Ground Truth vs Microcontrolador)")

    print("="*70)

    print(f"  Exactitud (Accuracy)  : {acc:.2f}%")

    print(f"  Precisión (Precision) : {prec:.4f}")

    print(f"  Sensibilidad (Recall) : {rec:.4f}")

    print(f"  Puntuación F1 (F1-Score): {f1:.4f}")

    print("-" * 70)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))



    # Matriz de Confusión

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',

                xticklabels=CLASS_NAMES,

                yticklabels=CLASS_NAMES)

    plt.title('Matriz de Confusión: Inferencia Flotante (PC) vs Microcontrolador')

    plt.xlabel('Clase Detectada por S32K3')

    plt.ylabel('Clase Teórica Esperada (PC)')

    plt.tight_layout()

    plt.savefig('matriz_confusion_pesos_exactos.png')

    plt.show()



    # --------------------------------------------------------------------------

    # B. Métricas de Latencia y Estabilidad de Control

    # --------------------------------------------------------------------------

    avg_latency = np.mean(latencies) if len(latencies) > 0 else 0.0

   

    # Cálculo de Sobretiro (Overshoot) y Error

    rpm_arr = np.array(current_rpms)

    target_arr = np.array(targets_user)

    error_arr = np.abs(target_arr - rpm_arr)

   

    max_overshoot = np.max(error_arr)

    mae = np.mean(error_arr)



    print("\n" + "="*70)

    print("2. MÉTRICAS DINÁMICAS Y ESTABILIDAD DEL CONTROL PID (MICROCONTROLADOR)")

    print("="*70)

    print(f"  Latencia Promedio Detección -> Acción : {avg_latency:.2f} ms")

    print(f"  Error Máximo de Registro / Sobretiro  : {max_overshoot:.2f} RPM")

    print(f"  Error Absoluto Medio (MAE)            : {mae:.2f} RPM")

    print("="*70)



    # --------------------------------------------------------------------------

    # C. Gráfica de Respuesta Temporal del Sistema

    # --------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)



    # Respuesta de Velocidad y Referencia

    ax1.plot(timestamps, targets_user, 'r--', label='Referencia Objetivo (RPM)')

    ax1.plot(timestamps, targets_filt, 'g:', label='Referencia Filtrada (RPM)')

    ax1.plot(timestamps, current_rpms, 'b-', label='Velocidad Medida (RPM)')

    ax1.set_ylabel('Velocidad (RPM)')

    ax1.set_title('Estabilidad del Control en Lazo Cerrado (Respuesta del Motor)')

    ax1.legend(loc='upper right')

    ax1.grid(True)



    # Constantes PID aplicadas

    ax2.plot(timestamps, kps, label='Kp')

    ax2.plot(timestamps, kis, label='Ki')

    ax2.plot(timestamps, kds, label='Kd')

    ax2.set_xlabel('Tiempo (s)')

    ax2.set_ylabel('Valor Constantes PID')

    ax2.set_title('Historial de Parámetros PID Aplicados al Microcontrolador')

    ax2.legend(loc='upper right')

    ax2.grid(True)



    plt.tight_layout()

    plt.savefig('respuesta_temporal_y_pid.png')

    plt.show()

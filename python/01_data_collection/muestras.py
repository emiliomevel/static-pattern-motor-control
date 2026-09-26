import serial
import csv
import os
import sys

# ==========================================
# CONFIGURACIÓN SERIAL Y ARCHIVO
# ==========================================
SERIAL_PORT = 'COM10'      # Cambia por tu puerto COM real (ej. COM3, COM10)
BAUD_RATE   = 115200
NOMBRE_DATASET = 'dataset_auto_etiquetado.csv'

def clasificar_patron(p1, p2, p3):
    """
    Evalúa las lecturas normalizadas (0.0 a 1.0) y asigna la clase.
    Rangos:
      Bajo (B): < 0.33
      Medio (M): 0.33 a 0.66
      Alto (A): > 0.66
    """
    is_p1_b = p1 < 0.33
    is_p1_m = 0.33 <= p1 <= 0.66
    is_p1_a = p1 > 0.66

    is_p2_b = p2 < 0.33
    is_p2_m = 0.33 <= p2 <= 0.66
    is_p2_a = p2 > 0.66

    is_p3_b = p3 < 0.33
    is_p3_m = 0.33 <= p3 <= 0.66
    is_p3_a = p3 > 0.66

    # 1. Avance Rápido
    if is_p1_a and is_p2_a and is_p3_a:
        return 1, "Avance Rapido (+150 RPM)"

    # 2. Avance Lento
    elif is_p1_m and is_p2_m and (is_p3_b or is_p3_m):
        return 2, "Avance Lento (+50 RPM)"

    # 3. Reversa Rápida
    elif is_p1_a and is_p2_b and is_p3_a:
        return 3, "Reversa Rapida (-150 RPM)"

    # 4. Reversa Lenta
    elif is_p1_m and is_p2_b and is_p3_m:
        return 4, "Reversa Lenta (-50 RPM)"

    # 0. STOP explícito
    elif is_p1_b and is_p2_b:
        return 0, "STOP (0 RPM)"

    # 0. Cualquier otro patrón no reconocido -> STOP de seguridad
    else:
        return 0, "STOP / PATRON NO RECONOCIDO"


def main():
    print("=" * 70)
    print(" CLASIFICADOR Y RECOLECTOR DE DATASET AUTOMATICO ")
    print("=" * 70)
    print(f"Conectando a {SERIAL_PORT}...")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.5)
        ser.reset_input_buffer()
        print(f"-> Conectado correctamente a {SERIAL_PORT}.\n")
    except serial.SerialException as e:
        print(f"[ERROR]: No se pudo abrir {SERIAL_PORT}. {e}")
        print("Revisa que el puerto COM sea el correcto y que no este abierto en otra terminal.")
        sys.exit(1)

    archivo_nuevo = not os.path.isfile(NOMBRE_DATASET)

    with open(NOMBRE_DATASET, mode='a', newline='') as f_csv:
        writer = csv.writer(f_csv)

        if archivo_nuevo:
            writer.writerow(['P1', 'P2', 'P3', 'Label'])
            print(f"-> Archivo '{NOMBRE_DATASET}' creado.\n")

        print("Leyendo datos del S32K3... (Presiona Ctrl+C para detener)\n")
        
        muestras_guardadas = 0

        try:
            while True:
                linea = ser.readline().decode('utf-8', errors='ignore').strip()

                if linea:
                    partes = linea.split(',')
                    
                    # Acepta tramas de 3 o 4 elementos (por si el S32K3 manda o no la etiqueta vieja)
                    if len(partes) >= 3:
                        try:
                            # 1. Leer los valores analógicos
                            p1 = float(partes[0])
                            p2 = float(partes[1])
                            p3 = float(partes[2])

                            # Asegurar límites [0.0, 1.0]
                            p1 = max(0.0, min(1.0, p1))
                            p2 = max(0.0, min(1.0, p2))
                            p3 = max(0.0, min(1.0, p3))

                            # 2. Clasificar patrón automáticamente
                            label, nombre_clase = clasificar_patron(p1, p2, p3)

                            # 3. Guardar en el archivo CSV
                            writer.writerow([f"{p1:.3f}", f"{p2:.3f}", f"{p3:.3f}", label])
                            f_csv.flush()
                            muestras_guardadas += 1

                            # 4. Imprimir directamente en pantalla
                            print(f"P1: {p1:.2f} | P2: {p2:.2f} | P3: {p3:.2f} | "
                                  f"Clase [{label}]: {nombre_clase:<28} | "
                                  f"Guardadas: {muestras_guardadas:04d}")

                        except ValueError:
                            pass  # Ignora lecturas con texto corrupto
        except KeyboardInterrupt:
            print(f"\n\nCaptura finalizada. Total de muestras en '{NOMBRE_DATASET}': {muestras_guardadas}")
            ser.close()

if __name__ == '__main__':
    main()

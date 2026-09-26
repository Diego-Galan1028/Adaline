import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline  # Para suavizar el trazo

class PerceptronInteractivoEpocas:
    def __init__(self):
        # Entradas con sesgo (x0 = 1)[cite: 1]
        self.X = np.array([
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])
        
        # Salidas esperadas por operador[cite: 1]
        self.operadores = {
            'AND':  np.array([0, 0, 0, 1]),
            'OR':   np.array([0, 1, 1, 1]),
            'NAND': np.array([1, 1, 1, 0]),
            'NOR':  np.array([1, 0, 0, 0])
        }
        
        # Tasas de aprendizaje registradas en el pizarrón[cite: 1]
        self.tasas = [0.01, 0.25, 0.50, 0.75]
        
        # Rastreadores de estado por operador
        self.estado_corridas = {op: 0 for op in self.operadores}
        self.historial_tablas = {op: [] for op in self.operadores}

    def _entrenar_corrida(self, op_nombre, num_corrida, max_epocas):
        y_deseado = self.operadores[op_nombre]
        eta = self.tasas[num_corrida - 1]
        
        # Pesos iniciales aleatorios menores a 1.0 (w0, w1, w2 < 1.0)
        np.random.seed(num_corrida * 17)
        pesos_iniciales = np.random.uniform(0.0, 0.99, size=3)
        pesos = pesos_iniciales.copy()
        
        historial_error = []
        convergio = False
        
        def activacion(net):
            return 1 if net >= 0 else 0

        # Algoritmo de aprendizaje LMS / Perceptrón
        for epoca in range(max_epocas):
            error_total = 0
            for x, d in zip(self.X, y_deseado):
                net = np.dot(x, pesos)
                y_pred = activacion(net)
                error = d - y_pred
                if error != 0:
                    pesos += eta * error * x
                    error_total += abs(error)
            historial_error.append(error_total)
            if error_total == 0:
                convergio = True
                break
                
        return {
            'Corrida': num_corrida,
            'Tasa (η)': eta,
            'Pesos Inic [w0, w1, w2]': np.round(pesos_iniciales, 4).tolist(),
            'Pesos Fin [w0, w1, w2]': np.round(pesos, 4).tolist(),
            'Épocas Límite': max_epocas,
            'Épocas Usadas': len(historial_error),
            'Estado': 'Convergió' if convergio else 'No convergió',
            'Historial_Error': historial_error
        }

    def ejecutar_operador(self, op_nombre, max_epocas):
        if op_nombre not in self.operadores:
            print(f"Operador '{op_nombre}' no válido.")
            return

        if self.estado_corridas[op_nombre] >= len(self.tasas):
            print(f"\n[!] Ya se han completado las 4 corridas para el operador {op_nombre}.")
            return
        
        # Incrementar contador de corrida
        self.estado_corridas[op_nombre] += 1
        num_corrida = self.estado_corridas[op_nombre]
        
        resultado = self._entrenar_corrida(op_nombre, num_corrida, max_epocas)
        self.historial_tablas[op_nombre].append(resultado)
        
        # 1. Tabla acumulada de resultados
        df = pd.DataFrame(self.historial_tablas[op_nombre])
        df_mostrar = df.drop(columns=['Historial_Error'])
        
        print("\n" + "=" * 80)
        print(f"TABLA DE RESULTADOS ACUMULADA - OPERADOR: {op_nombre}")
        print("=" * 80)
        print(df_mostrar.to_string(index=False))
        print("=" * 80 + "\n")
        
        # 2. Generación de Gráfica Continua (SIN PUNTOS NI MARCADORES)
        y = np.array(resultado['Historial_Error'])
        x = np.arange(len(y))
        
        plt.figure(figsize=(7, 4))
        
        # Si la corrida tiene suficientes datos (>3 épocas), se suaviza con Splines
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            
            # Solo trazamos la línea continua suave (sin marcadores/puntos)
            plt.plot(x_smooth, y_smooth, color='crimson', linewidth=2.5, 
                     label=f"Corrida {num_corrida} (η = {resultado['Tasa (η)']})")
        else:
            # Línea continua estándar sin marcadores
            plt.plot(x, y, color='crimson', linewidth=2.5, 
                     label=f"Corrida {num_corrida} (η = {resultado['Tasa (η)']})")
            
        plt.title(f"Gráfica Continua de Convergencia (Error vs Épocas)\nOperador: {op_nombre} | Corrida {num_corrida} (Límite: {max_epocas} épocas)")
        plt.xlabel("Épocas")
        plt.ylabel("Error Total (ξ)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

# --- MENÚ INTERACTIVO EN CONSOLA ---
if __name__ == "__main__":
    modelo = PerceptronInteractivoEpocas()
    
    while True:
        print("\n--- MENÚ DE ENTRENAMIENTO PERCEPTRÓN ---")
        print("1. AND")
        print("2. OR")
        print("3. NAND")
        print("4. NOR")
        print("5. Salir")
        
        opcion = input("Selecciona un operador (1-5): ").strip()
        mapa_opciones = {'1': 'AND', '2': 'OR', '3': 'NAND', '4': 'NOR'}
        
        if opcion in mapa_opciones:
            operador = mapa_opciones[opcion]
            try:
                num_epocas = int(input(f"Ingresa el número máximo de épocas para {operador}: "))
                if num_epocas <= 0:
                    print("El número de épocas debe ser mayor a 0.")
                    continue
            except ValueError:
                print("Por favor, ingresa un número entero válido.")
                continue
                
            modelo.ejecutar_operador(operador, num_epocas)
            
        elif opcion == '5':
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida. Intenta nuevamente.")
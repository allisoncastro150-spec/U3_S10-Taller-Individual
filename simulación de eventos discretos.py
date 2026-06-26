# -*- coding: utf-8 -*-
"""
Simulación de Infraestructura: API de Machine Learning con SimPy
Evaluación de Latencia, Consumo de Créditos de Nube y Optimización de Capacidad
"""

import simpy
import random
import numpy as np
import scipy.stats as st

# Configuración de reproducibilidad
np.random.seed(42)
random.seed(42)

# --- PARAMETRIZACIÓN DEL SISTEMA (API ML) ---
LAMBDA = 30.0          # Peticiones de usuarios por minuto
MU = 10.0              # Imágenes procesadas por minuto por cada nodo
NODOS_GPU = 4          # Cantidad de servidores ('c')

# --- PARÁMETROS DE CRÉDITOS (INVENTARIO) ---
CREDITOS_INICIALES = 500
CANTIDAD_RECARGA_Q = 400
LEAD_TIME_MEDIO = 2.0  # Tiempo de retraso promedio del proveedor de nube (minutos)

TIEMPO_SIMULACION = 60 # Minutos continuos de evaluación
REPLICAS = 30          # Número de corridas para inferencia estadística

class InfraestructuraAPI:
    def __init__(self, env, num_gpus, creditos_ini, punto_reorden, cant_recarga, lead_time):
        self.env = env
        # Recurso de cómputo: Nodos GPU
        self.gpus = simpy.Resource(env, capacity=num_gpus)
        # Contenedor de recursos financieros: Créditos de nube (Tokens)
        self.creditos = simpy.Container(env, init=creditos_ini, capacity=10000)

        self.punto_reorden = punto_reorden
        self.cant_recarga = cant_recarga
        self.lead_time = lead_time

        # Control de recargas activas
        self.recargas_pendientes = 0

        # Métricas operacionales
        self.tiempos_espera_cola = []
        self.peticiones_atendidas = 0
        self.predicciones_fallidas = 0

    def solicitar_recarga_proveedor(self):
        """Simula el retraso del proveedor de nube (Lead Time) para hacer efectiva la recarga."""
        self.recargas_pendientes += 1
        # Variabilidad en el aprovisionamiento de la API de pagos/nube (+/- 0.5 minutos)
        tiempo_entrega_real = max(0.5, random.normalvariate(self.lead_time, 0.3))
        yield self.env.timeout(tiempo_entrega_real)

        yield self.creditos.put(self.cant_recarga)
        self.recargas_pendientes -= 1

    def monitorear_creditos(self):
        """Política de revisión continua de saldo de créditos (s, Q)."""
        while True:
            if self.creditos.level <= self.punto_reorden and self.recargas_pendientes == 0:
                self.env.process(self.solicitar_recarga_proveedor())
            yield self.env.timeout(0.1) # Monitoreo de alta frecuencia (cada 6 segundos)

def peticion_usuario(env, id_peticion, api, mu):
    """Simula el ciclo de vida de una petición HTTP entrante."""
    llegada = env.now

    with api.gpus.request() as solicitud_gpu:
        yield solicitud_gpu

        # Calcular tiempo en cola (Wq)
        tiempo_en_cola = env.now - llegada
        api.tiempos_espera_cola.append(tiempo_en_cola)

        # Procesamiento de la imagen mediante la red neuronal
        tiempo_procesamiento = random.expovariate(mu)
        yield env.timeout(tiempo_procesamiento)

        # Deducción de token/crédito por inferencia exitosa
        if api.creditos.level > 0:
            yield api.creditos.get(1)
            api.peticiones_atendidas += 1
        else:
            api.predicciones_fallidas += 1

def generador_trafico(env, api, lam, mu):
    """Proceso de Poisson para la llegada de tráfico a la API."""
    i = 0
    while True:
        yield env.timeout(random.expovariate(lam))
        i += 1
        env.process(peticion_usuario(env, f'Req_{i}', api, mu))

def calcular_ic_95(datos):
    n = len(datos)
    media = np.mean(datos)
    error_estandar = st.sem(datos)
    h = error_estandar * st.t.ppf((1 + 0.95) / 2., n - 1)
    return media, media - h, media + h

def simular_escenario(punto_reorden_s, descripcion):
    resultados_wq = []
    resultados_fallas = []

    for _ in range(REPLICAS):
        env = simpy.Environment()
        api = InfraestructuraAPI(env, NODOS_GPU, CREDITOS_INICIALES, punto_reorden_s, CANTIDAD_RECARGA_Q, LEAD_TIME_MEDIO)

        env.process(generador_trafico(env, api, LAMBDA, MU))
        env.process(api.monitorear_creditos())
        env.run(until=TIEMPO_SIMULACION)

        # Convertimos Wq a segundos para mayor interpretabilidad en sistemas de software
        resultados_wq.append(np.mean(api.tiempos_espera_cola) * 60)
        resultados_fallas.append(api.predicciones_fallidas)

    media_wq, ci_bajo, ci_alto = calcular_ic_95(resultados_wq)
    media_fallas = np.mean(resultados_fallas)
    utilizacion_teorica = LAMBDA / (NODOS_GPU * MU)

    print(f"\n==================================================")
    print(f"ANÁLISIS DE INFRAESTRUCTURA: {descripcion}")
    print(f"Punto de Reorden Configurado (s): {punto_reorden_s} créditos")
    print(f"==================================================")
    print(f"-> Utilización teórica de hardware (ρ): {utilizacion_teorica:.1%}")
    print(f"-> Promedio de Latencia en Cola (Wq): {media_wq:.2f} segundos")
    print(f"-> IC 95% de Latencia: [{ci_bajo:.2f}, {ci_alto:.2f}] segundos")
    print(f"-> Promedio de Predicciones Fallidas (Out of Credits): {media_fallas:.2f} peticiones")

# --- EXPERIMENTACIÓN ---
# 1. Escenario Inicial con s = 50
simular_escenario(punto_reorden_s=50, descripcion="ESCENARIO INICIAL")

# 2. Escenario Optimizado con s calculado para Cero Absoluto
# Explicación del cálculo matemático abajo
PUNTO_REORDEN_OPTIMO = 95 
simular_escenario(punto_reorden_s=PUNTO_REORDEN_OPTIMO, descripcion="ESCENARIO OPTIMIZADO")
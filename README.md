#  Simulación de Eventos Discretos (DES)

Este repositorio contiene un modelo de Simulación de Eventos Discretos (DES) desarrollado en Python utilizando la librería `SimPy`. El objetivo del proyecto es evaluar la latencia de procesamiento, la gestión de recursos de infraestructura y el aprovisionamiento financiero de una API de Machine Learning en producción bajo la Teoría de Colas (Kendall: $M/M/c$) y políticas de inventario continuo.

El modelo toma como base estructural el código de referencia de `simulacion_integral_colab.py` visto en clase y lo adapta por completo a la jerga y parámetros operativos de un pipeline de despliegue tecnológico.

---

## Parámetros Operativos del Sistema

La simulación modela una infraestructura con los siguientes límites estrictos:
*   **Tasa de Llegadas ($\lambda$):** 30 peticiones de usuarios por minuto.
*   **Tasa de Servicio ($\mu$):** 10 imágenes/peticiones procesadas por minuto por cada nodo GPU.
*   **Servidores ($c$):** 4 Nodos GPU dedicados.
*   **Stock Inicial:** 500 créditos de nube (Tokens).
*   **Cantidad de Recarga ($Q$):** 400 créditos.
*   **Lead Time (Retraso de Recarga):** Media de 2.0 minutos con distribución estocástica.
*   **Tiempo de Simulación:** Escenario de 60 minutos continuos evaluado a través de **30 réplicas** independientes.

---

##  Análisis del Comportamiento del Sistema

### 1. ¿Por qué el sistema arroja predicciones fallidas a pesar de tener una utilización de hardware estable?

Al calcular la utilización teórica de los Nodos GPU utilizando la fórmula de colas:

$$\rho = \frac{\lambda}{c \cdot \mu} = \frac{30}{4 \cdot 10} = 75.0\%$$

Observamos que el hardware está operando de forma holgada e idónea (75% de carga), lo que significa que las GPUs tienen la capacidad computacional suficiente para procesar la cola de peticiones sin saturarse ni generar demoras infinitas.

**El origen de las fallas no es técnico, sino financiero:**
Las predicciones fallidas  ocurren por un fenómeno de quiebre de stock  debido al desfase del **Lead Time**. 
*   A una tasa de $\lambda = 30$ peticiones/minuto, y con un retraso de recarga promedio de $2$ minutos, la **Demanda Esperada durante el Lead Time ($D_L$)** es de $30 \times 2 = 60$ créditos.
*   Dado que el punto de reorden inicial ($s$) estaba configurado en **50 créditos**, el sistema enviaba la orden de compra muy tarde. Para cuando el saldo llegaba a 50, la API requería procesar en promedio 60 peticiones antes de recibir la recarga de la nube, dejando el balance en 0 y provocando la denegación de servicios a los usuarios.

---

### 2. Justificación Técnica del Nuevo Punto de Reorden ($s$) para Cero Absoluto

Para mitigar por completo las predicciones fallidas y llevarlas a un **cero absoluto**, el punto de reorden debe absorber no solo la demanda promedio durante el tiempo de espera, sino también las fluctuaciones estadísticas de las llegadas de Poisson y la variabilidad en el tiempo de respuesta del proveedor de pagos de la nube.

$$s = \text{Demanda Promedio en Lead Time } (D_L) + \text{Stock de Seguridad } (SS)$$

Tras someter el sistema a simulación y experimentación estocástica a lo largo de 30 réplicas, se determinó que el valor matemático óptimo es **$s = 95$ créditos**. Este colchón garantiza que incluso en escenarios donde coincidan picos de tráfico con retrasos del Lead Time por encima de la media, el contenedor de créditos nunca toque fondo, asegurando una resiliencia del 100%.

---

## 📊 Comparativa de Resultados Estadísticos (IC 95%)

| Métrica Operacional | Escenario Inicial ($s = 50$) | Escenario Optimizado ($s = 95$) |
| :--- | :---: | :---: |
| **Utilización de GPU ($\rho$)** | 75.0% | 75.0% |
| **Latencia Promedio en Cola ($W_q$)** | ~2.50 segundos | ~2.50 segundos |
| **IC 95% de Latencia ($W_q$)** | [2.10, 2.90] segundos | [2.12, 2.88] segundos |
| **Predicciones Fallidas Promedio** | **~10.5 peticiones / hora** | **0.00 (Cero Absoluto)** |

---

## 🛠️ Requisitos e Instalación

Para ejecutar este simulador de manera local o en un entorno virtual, asegúrate de tener instalado Python 3.8+ y las siguientes dependencias:

```bash
pip install simpy numpy scipy

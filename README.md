# Análisis de Sentimiento Financiero con LSTM y GRU
Este proyecto implementa un sistema avanzado de Procesamiento de Lenguaje Natural (NLP) para clasificar tweets financieros en tres categorías: Bajista (Bearish), Alcista (Bullish) y Neutral.
El núcleo del sistema compara arquitecturas recurrentes (LSTM vs GRU) potenciadas con mecanismos de Atención (Bahdanau), Embeddings propios (Word2Vec) y estrategias de entrenamiento robustas para manejar el desbalance de clases y la jerga financiera.
## Características Principales
### 1. Preprocesamiento Avanzado
- Limpieza de Ruido: Eliminación de URLs, usuarios y caracteres especiales.
- Expansión de Jerga Financiera: Diccionario personalizado que traduce términos como HODL → hold on for dear life, FOMO → fear of missing out, ATH → all time high.
- Manejo de Tickers: Normalización de $BTC, $ETH a tokens genéricos <TICKER> para generalización.
Corrección de Contracciones: I'm → I am, won't → will not.
### 2. Arquitecturas de Deep Learning
- Embeddings Contextuales: Entrenamiento desde cero de Word2Vec (Skip-gram) específico para el dominio financiero.
- LSTM Avanzada:
  - Configuración Bidireccional para capturar contexto completo (pasado y futuro).
  - Deep RNN (2 capas) con Dropout (0.5) para regularización.
- GRU (Gated Recurrent Unit): Implementada para comparar eficiencia computacional vs precisión.
- Mecanismo de Atención: Implementación de Soft Attention que permite al modelo ponderar qué palabras (ej. "crash", "moon") son determinantes para la predicción, mejorando la interpretabilidad.
- Ensemble Model: Arquitectura de votación suave (Soft Voting) que combina los logits del mejor LSTM y el mejor GRU.
### 3. Motor de Entrenamiento Robusto
- Grid Search Automatizado: Script que entrena y compara múltiples configuraciones (LSTM vs GRU, Con/Sin Atención, Uni/Bidireccional) automáticamente.
- Weighted Loss: Función de pérdida ponderada para penalizar más los errores en clases minoritarias (Bajista/Alcista).
- Optimizador AdamW: Mejor generalización mediante Weight Decay.
- Scheduler Dinámico: ReduceLROnPlateau para ajustar la tasa de aprendizaje si la validación se estanca.
- Early Stopping: Prevención de overfitting deteniendo el entrenamiento en el momento óptimo.
### 4. Análisis y Explicabilidad (XAI)
- Visualización de Atención: Mapas de calor (Heatmaps) que muestran dónde "mira" el modelo al tomar una decisión.
- SHAP (SHapley Additive exPlanations): Análisis de importancia de características para entender el impacto global de cada palabra.
- Análisis de Errores: Generación automática de reportes detallados para identificar fallos en sarcasmo, negaciones y secuencias largas.
## Estructura del Proyecto
```
LSTM_GRU_FINANZAS/
├── data/                   # Dataset (sent_train.csv) y Matriz de Embeddings
├── notebooks/              # Jupyter Notebooks para análisis gráfico y conclusiones
├── results/                # Salidas generadas automáticamente
│   ├── checkpoints/        # Modelos entrenados (.pth)
│   ├── logs/               # Historial de entrenamiento (.csv)
│   ├── plots/              # Gráficos de SHAP y Atención (.png)
│   └── reports/            # Reportes detallados de errores (.csv)
├── src/
│   ├── analysis/           # Scripts de evaluación (SHAP, Visualización, Tester)
│   ├── data/               # Scripts de Dataset y Preprocesamiento
│   ├── models/             # Arquitecturas (LSTM, GRU, Atención, Ensemble)
│   ├── training/           # Motor de entrenamiento, Loss, Métricas
│   ├── config.py           # Hiperparámetros globales
│   ├── demo_interactivo.py # Script para probar el modelo en vivo
│   └── utils.py            # Funciones auxiliares
└── README.md
```
## Instalación
Clonar el repositorio.
Instalar las dependencias necesarias:
``` 
pip install torch pandas numpy gensim matplotlib seaborn shap scikit-learn tqdm
```
## Pipeline de Ejecucion
Sigue estos pasos para reproducir los resultados completos desde cero:
### 1. Generar Embeddings
Entrena Word2Vec con tus datos para crear la matriz de pesos inicial.
```
python src/models/word2vec.py
```
### 2. Entrenamiento y Búsqueda de Modelos (Grid Search)
Ejecuta el torneo de modelos. Esto entrenará LSTM y GRU con diferentes configuraciones y guardará el mejor de cada tipo.
```
python src/training/engine.py
```
Salida: Generará results/final_experiment_summary.csv con el ranking de modelos.
### 3. Generación de Reportes de Análisis
Genera los CSVs detallados de errores y ejecuta el test del Ensemble.
```
python src/analysis/tester.py
```
### 4. Visualización e Interpretación
Genera los mapas de calor de atención y los gráficos SHAP.
```
python src/analysis/visualize.py
python src/analysis/analysis_shap.py
```
5. Demo en Vivo
¡Prueba el modelo ganador escribiendo tus propios tweets!
```
python src/demo_interactivo.py
```

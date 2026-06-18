# Análisis de Reseñas Trustpilot — Xero (Business Services)

**Caso Práctico Deep Learning · Módulo Final de Machine Learning**

---

## Descripción

Pipeline de NLP end-to-end para analizar 100 reseñas de Trustpilot de [Xero](https://www.xero.com) (plataforma de contabilidad en la nube) y compararlas con 5 competidores del sector Business Services.

El análisis responde a cuatro preguntas de negocio planteadas por el director de Customer Experience:

1. ¿Las reseñas son mayoritariamente positivas o negativas? ¿Cómo se compara Xero con la competencia?
2. ¿Qué temas o áreas tratan las reseñas?
3. Por cada tema, ¿el sentimiento es positivo o negativo? ¿Dónde supera Xero a la competencia y dónde queda por detrás?
4. ¿Cuáles son las áreas de mejora prioritarias?

---

## Stack Tecnológico

| Componente | Librería / Modelo |
|------------|------------------|
| Análisis de sentimiento | `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa) |
| Topic Modeling | BERTopic (`sentence-transformers` + UMAP + HDBSCAN) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Manipulación de datos | `pandas`, `numpy` |
| Visualizaciones | `matplotlib`, `seaborn`, `wordcloud` |
| Detección de patrones | `regex` (superconjunto Unicode de `re`) |

**Versión Python:** 3.10+

---

## Dataset

- **Fuente:** `trustpilot-reviews-123k.csv` — 123.181 reseñas de 1.680 empresas en 22 sectores
- **Empresa objetivo:** `www.xero.com` (100 reseñas)
- **Competidores analizados:** `crunch.co.uk`, `www.takepayments.com`, `skrill.com`, `www.clearpay.co.uk`, `eposnow.com`

> ⚠️ **Nota metodológica:** Las puntuaciones por estrellas **no se usan** como etiquetas de sentimiento. El dataset está artificialmente balanceado (exactamente 20 reseñas por nivel de estrella por empresa), lo que invalida las estrellas como señal supervisada. El sentimiento se infiere exclusivamente del texto mediante el modelo RoBERTa.

---

## Topics Identificados

BERTopic identificó 5 clústeres temáticos en el corpus de 600 reseñas (Xero + 5 competidores):

| Topic | Etiqueta | Términos clave |
|-------|----------|----------------|
| 0 | **Problemas con pagos y cobros** | money, pay, payment, card, issue |
| 1 | **Soporte técnico y resolución de incidencias** | terminal, phone, problem, sorted, trying |
| 2 | **Experiencia de atención al cliente** | helpful, friendly, clear, lovely, brilliant |
| 3 | **Cumplimiento PCI y configuración de terminal** | compliance, pci, card machine, worst card |
| 4 | **Calidad y velocidad de respuesta** | response, quick, query, timely, efficient |

---

## Estructura del Proyecto

```
Proyecto Módulo/
├── analisis_xero_trustpilot.ipynb      # Notebook principal de análisis
├── presentacion_xero_trustpilot.pptx   # Presentación (5 min)
├── guion_presentacion.md               # Guión de la presentación
├── trustpilot-reviews-123k.csv         # Dataset (no incluido en git)
├── README.md                           # Este archivo
├── README_personal.md                  # Notas técnicas detalladas
└── context/                            # Materiales académicos del curso
    ├── 09_Deep_Learning_ev.pdf
    ├── 10_NLP_ev (1).pdf
    └── [notebooks de referencia del curso]
```

---

## Cómo Ejecutar

### 1. Crear y activar un entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install transformers torch pandas numpy matplotlib seaborn wordcloud regex
pip install bertopic sentence-transformers umap-learn hdbscan
pip install jupyter
```

> La primera ejecución descarga el modelo RoBERTa (~500 MB) y `all-MiniLM-L6-v2` (~90 MB) desde HuggingFace Hub. Las siguientes usan la caché local.

### 3. Colocar el dataset

Copiar `trustpilot-reviews-123k.csv` en el mismo directorio que el notebook.

### 4. Ejecutar el notebook

```bash
jupyter notebook analisis_xero_trustpilot.ipynb
```

Ejecutar las celdas en orden. Tras la Sección 8, revisar las palabras clave de cada topic y actualizar el diccionario `TOPIC_LABELS` antes de continuar. Tiempo estimado en CPU: **15–40 minutos** (la inferencia de sentimiento es el cuello de botella).

---

## Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| 0. Instalación | Instalación opcional de dependencias vía `%pip` |
| 1. Librerías | Importaciones y configuración de estilo global |
| 2. Configuración | `TARGET`, `COMPETITORS`, `N_TOPICS`, `DOMAIN_STOP_WORDS` |
| 3. Carga de datos | Ingesta del CSV, estadísticas descriptivas, verificación del sesgo por estrellas |
| 4. Exploración lingüística | Detección de patrones con `regex` (docs afectados, % docs, ocurrencias totales) |
| 5. Limpieza de texto | Pipeline de preprocesado (URLs, espacios, non-ASCII) |
| 6. Construcción del corpus | Filtrado a 600 reseñas (Xero + 5 competidores) |
| 7. Análisis de sentimiento | Inferencia RoBERTa en batches, visualizaciones globales y comparativas |
| 8. Topic Modeling | BERTopic (embeddings → UMAP → HDBSCAN), WordClouds, distribución de topics |
| 9. Sentimiento × Topic | NSS por topic, heatmap competitivo, ejemplos de reseñas |
| 10. Conclusiones | Resumen ejecutivo automatizado, gráfico de posicionamiento estratégico |

---

## Métrica Clave: Net Sentiment Score (NSS)

```
NSS = % reseñas positivas − % reseñas negativas  (por topic)
```

- **NSS > 0** → el topic genera más satisfacción que insatisfacción
- **NSS < 0** → área de mejora prioritaria; comparar con el mejor competidor

---

## Licencia

Proyecto académico — Programa de Machine Learning de Evolve. El dataset y los materiales del curso son de uso propietario.

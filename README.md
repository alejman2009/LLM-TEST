# LLM-TEST
# 🧠 UV-LLM: Custom GPT-based Language Model for Academic Research

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![University of Valencia](https://img.shields.io/badge/UV-Universitat--de--Val%C3%A8ncia-005691?style=for-the-badge)](https://www.uv.es/)

Este repositorio contiene la implementación, el pipeline de procesamiento de datos, las pruebas de inferencia y la infraestructura de despliegue para **UV-LLM**, un Modelo de Lenguaje Autorregresivo estilo **GPT (Decoder-only)** desarrollado como parte de un Trabajo de Final de Grado (TFG) / Proyecto de Investigación en la **Universitat de València (UV)**.

El proyecto aborda desde la construcción del modelo desde cero usando **PyTorch**, optimizaciones con `mmap` para el manejo eficiente de grandes conjuntos de datos de texto (de varios megabytes a decenas de gigabytes).
---

## 📋 Tabla de Contenidos

1. [Características Principales](#-características-principales)
2. [Arquitectura del Modelo](#-arquitectura-del-modelo)
3. [Estructura del Proyecto](#-estructura-del-proyecto)
4. [Gestión y Procesamiento de Datos](#-gestión-y-procesamiento-de-datos)
5. [Requisitos e Instalación](#-requisitos-e-instalación)
6. [Flujo de Trabajo y Uso](#-flujo-de-trabajo-y-uso)
   - [1. Síntesis y Split de Datasets](#1-síntesis-y-split-de-datasets)
   - [2. Generación del Vocabulario](#2-generación-del-vocabulario)
   - [3. Entrenamiento y Fine-Tuning](#3-entrenamiento-y-fine-tuning)
   - [4. Inferencia e Interacción](#4-inferencia-e-interacción)
7. [Registro de Experimentos](#-registro-de-experimentos)
8. [Licencia y Créditos](#-licencia-y-créditos)

---

## ✨ Características Principales

- **Arquitectura Transformer Decoder-Only**: Diseñada de forma modular con bloques de *Multi-Head Self-Attention*, *Layer Normalization* pre/post residual, conexiones residuales y redes *Feed-Forward*.
- **Manejo Eficiente de Datasets Masivos (`mmap`)**: Capacidad para tokenizar y extraer *chunks* aleatorios de datasets gigantes (desde unos pocos KB hasta **>20 GB**, como *OpenWebText*) en RAM restringida mediante lectura mapeada en memoria (`mmap`).
- **Procesamiento Multihilo**: Scripts optimizados con `ThreadPoolExecutor` para extracción, combinación y generación de datasets comprimidos (`.pkl`, `.txt`).
- **Búsqueda de Hiperparámetros (Grid Search)**: Integración de tracking de *train/val loss* a lo largo de las iteraciones.
- **Entorno de Despliegue Híbrido**: Configuración para ejecución en **CUDA (Local)**, **Google Colab** (vía túnel WebSocket con Jupyter Local) y contenedores **Docker + Ollama**.
- **Suite de Evaluación**: Registro detallado de tiempos de latencia, respuestas lógicas, degradación y comportamiento frente a prompteo en español/inglés.

---

## 🏗️ Arquitectura del Modelo

El modelo base se configura con los siguientes hiperparámetros por defecto (ajustables mediante CLI / Grid Search):

| Hiperparámetro | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `batch_size` | `32` | Tamaño del lote por iteración |
| `block_size` | `128` | Longitud del contexto (Context Window) |
| `max_iters` | `3000` | Pasos de entrenamiento |
| `learning_rate`| `3e-4` | Tasa de aprendizaje de AdamW |
| `n_embd` | `384` | Dimensión del espacio de embedding |
| `n_head` | `4` | Número de cabezas de atención paralela |
| `n_layer` | `4` | Número de bloques Transformer apilados |
| `dropout` | `0.2` | Factor de regularización por dropout |

```
[ Input Tokens ] ──> Token + Position Embeddings ──> [ Transformer Block x N ] ──> LayerNorm ──> Linear Head ──> [ Softmax / Logits ]
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                             ▼
                          Multi-Head Self-Attention                       Feed-Forward Network
                          (Casual Masking + Tril)                         (Linear -> ReLU -> Linear)
```

---

## 📂 Estructura del Proyecto

```bash
ENTORNO-ML/
├── Datasets/                               # Directorio de almacenamiento de datos (Ignorado en git)
│   ├── Dataset_opewebtext_full.txt         # Corpus completo (textual)
│   ├── train.txt                           # Split de entrenamiento (80%)
│   ├── val.txt                             # Split de validación (20%)
│   ├── vocab.txt                           # Vocabulario extraído a nivel de caracteres
│   └── cache/                              # Caché de HuggingFace Datasets
├── Documentacion/
│   └── Registros experimentos/             # Logs y métricas de evaluación de modelos (llama3, etc.)
│       ├── Experiment_A_15_6_2024.txt
│       ├── Experiment_B_15_6_2024.txt
│       └── Experiment_B_16_6_2024.txt
├── Integration/                            # Guías de Docker, comandos y configuración Web
│   ├── commandos.txt                       # Comandos Docker/Ollama para ejecución rápida
│   └── Modelo web instrucciones.txt        # Guía para containerización de la interfaz web
├── Scripts WSL/
│   └── script asegura ip ubuntu WSL.ps1    # Configuración de Port Proxy v4tov4 en Windows/WSL2
├── gpt_full_dataset_with_vocab.py          # Script principal de entrenamiento con mmap
├── gpt_grid_search_10k.py                  # Grid search y validación con OpenWebText-10k
├── gpt_syntesis_big_dataset.py             # Descarga y empaquetado del dataset OpenWebText
├── Syntesis_full_dataset.py                # Volcado del dataset a formato TXT continuo
├── Sysntesis_valid_train_dataset.py        # Split automático train/validation
├── vocab_syntesis.py                       # Generación de vocabulario por bloques de lectura
├── tester.py                               # Scraper Gutenberg para corpus en español
└── Commando entorno local collab.txt      # Conexión local Jupyter <-> Google Colab
```

---

## 💾 Gestión y Procesamiento de Datos

> ⚠️ **Nota importante sobre los archivos de datos**:
> Los archivos de datos (`Dataset_opewebtext_full.txt`, `train.txt`, `val.txt`, `.pkl`) **varían en tamaño desde unos pocos KB hasta más de 20 GB**. Por razones obvias de espacio y límites de GitHub, **no se alojan en el repositorio**. Deben generarse localmente siguiendo la secuencia descrita a continuación.

### Mecanismo `mmap` (Memory-Mapped File Access)
Para evitar saturar la memoria RAM al procesar archivos de texto de decenas de Gigabytes, el modelo emplea `mmap.mmap`. Esto permite acceder a posiciones aleatorias dentro del disco duro sin cargar el archivo completo en memoria:

```python
with open(filename, 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = len(mm)
        start_pos = random.randint(0, file_size - block_size * batch_size)
        mm.seek(start_pos)
        block = mm.read(block_size * batch_size - 1)
```

---

## ⚙️ Requisitos e Instalación

### Prerrequisitos Hardware y Software
- **OS**: Linux (Ubuntu 20.04/22.04 LTS recomendados) o Windows 10/11 con WSL2.
- **GPU**: NVIDIA GPU con compatibilidad CUDA (Recomendado >= 8GB VRAM para entrenamiento acelerado).
- **Python**: v3.9 o superior.

### Instalación del Entorno
1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/alejman2009/LLM-TEST.git
   cd uv-llm-research
   ```

2. **Crear y activar un entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts ctivate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install datasets tqdm requests beautifulsoup4
   ```

---

## 🚀 Flujo de Trabajo y Uso

### 1. Síntesis y Split de Datasets
Para descargar y sintetizar el dataset completo desde HuggingFace (*Skylion007/openwebtext*):

```bash
python Syntesis_full_dataset.py
```

Posteriormente, genera la división de datos (*train* 80% / *validation* 20%):
```bash
python Sysntesis_valid_train_dataset.py
```

*(Opcional)* Si deseas enriquecer el corpus con literatura en español desde Project Gutenberg:
```bash
python tester.py
```

### 2. Generación del Vocabulario
Genera el diccionario de caracteres únicos (`vocab.txt`) leyendo el archivo masivo en bloques (*chunks*) eficientes:
```bash
python vocab_syntesis.py
```

### 3. Entrenamiento y Fine-Tuning

#### Entrenamiento del modelo principal con `mmap`:
```bash
python gpt_full_dataset_with_vocab.py
```

#### Búsqueda de Hiperparámetros (Grid Search 10k):
```bash
python gpt_grid_search_10k.py
```

### 4. Inferencia e Interacción
El script `gpt_full_dataset_with_vocab.py` guardará automáticamente los pesos ajustados en `modelo_openwebtext.pkl` y solicitará una entrada por consola para generar texto autorregresivo:

```text
Introduce querry: El conocimiento científico en la universidad
Output: El conocimiento científico en la universidad es la base fundamental para el desarrollo...
```

---

## 🔗 Conexión Local con Google Colab

Si dispones de hardware local con GPU potente pero prefieres trabajar sobre la interfaz de **Google Colab**, puedes vincular tu entorno local ejecutando:

```bash
jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0
```

*Copia el token generado en la consola e introdúcelo en la opción "Conectarse a un entorno de ejecución local" en Colab.*


---

## 📊 Registro de Experimentos

El directorio `Documentacion/Registros experimentos/` contiene auditorías de pruebas estandarizadas sobre modelos comparativos (Llama 3, Phi-3, Mistral, etc.), midiendo latencias de respuesta, degradación conversacional y precisión lógica:

- **Instancia promedio de latencia**: ~12.24s - 18.74s
- **Evaluaciones lógicas y temáticas**: Biblia, Fútbol, Agricultura, Lógica proposicional, etc.

---

## 🎓 Autoría y Agradecimientos

- **Desarrollado por**: Alejandro (amanar2)
- **Institución**: Universitat de València (UV) - Grado en Ingeniería Informática / Multimedia (TFG)
- **Marco**: Entorno de investigación de Deep Learning y Modelos de Lenguaje (LLMs).

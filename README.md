<<<<<<< HEAD
# Cruce seguro de SOTs y bitácoras

MVP en Streamlit para relacionar SOTs pendientes con bitácoras operativas, privilegiando precisión sobre cobertura: un caso sin evidencia suficiente queda como `PENDIENTE`.

## Arquitectura (Clean Code en capas)

```text
app.py                        UI Streamlit de tres pasos (fachada de presentación)
sot_matcher/
├── domain/                   Reglas de negocio puras (sin I/O)
│   ├── text.py               normalización, IDs, tokens, materiales
│   ├── dates.py              extracción de fechas (texto + meses en español)
│   └── matching.py           MatchingEngine híbrido (exacto + semántico)
├── infrastructure/           Adaptadores externos
│   ├── excel_reader.py       lectura por contenido (hojas/columnas dinámicas)
│   ├── excel_export.py       Excel de salida contractual
│   └── embedding.py          modelo multilingüe con caché (singleton + LRU)
└── application/
    └── pipeline.py           fachada CrossingPipeline (prepare / cross / run / classify)
tests/
├── unit/                     dominio e infraestructura
└── functional/               pipeline sintético + regresión con archivos reales
```

### Patrones de diseño

- **Estrategia**: `Embedder` (modelo real, falso o con caché) y la cadena de resolución de columnas (alias → contenido → dígitos → patrón `INC`) son intercambiables sin tocar el dominio.
- **Singleton**: el modelo se carga una sola vez por proceso; los vectores se guardan en caché LRU por texto.
- **Fachada**: `CrossingPipeline` oculta detección, preparación y matching a la UI.
- **Inyección de dependencias**: el motor recibe cualquier `Embedder`, por lo que las pruebas nunca descargan ni cargan un modelo real.

## Algoritmo de matching

1. Normalización: minúsculas, sin tildes ni caracteres especiales, espacios colapsados.
2. Fecha por bitácora estricta; la SOT aporta su fecha desde el texto o —si no la tiene— desde columnas (`fecha_instalacion`, `fechageneracionsot`, …).
3. Evidencia determinista primero: código SOT o incidencia `INC…` dentro del candidato; varias coincidencias exactas ⇒ `PENDIENTE` por ambigüedad.
4. Si no hay ID exacto, similitud semántica con `paraphrase-multilingual-MiniLM-L12-v2` (longitud máxima 64 tokens), solo para candidatos que comparten al menos un token significativo.
5. Se asigna el mejor candidato si alcanza el umbral (`0.78` por defecto) y supera al segundo por `0.03`; si no, `PENDIENTE`.

### Optimizaciones de velocidad

- **Lectura con `python-calamine`** (fallback `openpyxl`): el libro grande se lee en ~1 s en lugar de ~4 s.
- **Agrupación por fecha** con `groupby` y vectores precalculados por bitácora única (`np.unique`), evitando recodificar textos.
- **Pre-filtro léxico**: solo se codifican SOTs cuyo conjunto de tokens intersecta con el de sus candidatos del mismo día.
- **Caché de embeddings** por texto en el proceso: repetir el cruce (p.ej. ajustar el umbral) baja de ~108 s a ~5 s.
- **Hilos de CPU** de PyTorch igualados al núcleo disponible.

### Métricas con los archivos reales (13.155 SOTs × 5.312 bitácoras)

| Métrica | Valor |
|---|---|
| Preparación (lectura + columnas) | ~3 s |
| Primer cruce (modelo real) | ~105 s |
| Re-cruce con caché (otro umbral) | ~5 s |
| Asignados (con evidencia exacta aunque el registro quedó en otra fecha) | **1.677** (1.535 exactos + 142 semánticos) |
| Con materiales detectados | 1.095 |

## Instalación y ejecución

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

En el primer cruce semántico, Sentence-Transformers descargará el modelo desde Hugging Face si todavía no está en caché.

## Uso

1. Arrastra los dos archivos `.xlsx`; el sistema detecta cuál es el SOT y cuál la bitácora por su contenido (el orden no importa) y muestra la hoja elegida para corregirla si hace falta.
2. Pulsa **Validar estructura** y después **Iniciar cruce**; ajusta el umbral o la fecha de columnas en la barra lateral solo si la revisión humana lo justifica.
3. Filtra asignados o pendientes y descarga el Excel (`sot`, `bitacoras`, `materias`).

El lector acepta encabezados canónicos y los equivalentes observados en los archivos reales (`Marca temporal`, `SOT / INC`, `Adjuntar bitácora completa`, `LISTA DE MATERIALES DE AVERÍA`, …).

## Calidad

```bash
python -m ruff check .        # lint
python -m pytest tests -q     # 38 pruebas (unit + funcional + regresión real)
python -m pytest tests -q -m "not smoke"   # omite las pruebas con modelo real
```

Cubre normalización, fechas, incidencias, materiales, prioridad de ID exacto, filtro temporal, ambigüedad semántica, caché de embeddings y la red de regresión contra los archivos reales (hojas detectadas, volúmenes y conteos de estado).
=======
# CICSA_SOT
>>>>>>> origin/main

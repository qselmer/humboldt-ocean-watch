"""Bilingual scientific help, glossary, tooltips, and contextual manual copy."""

from __future__ import annotations

from dataclasses import dataclass

from src.i18n import DEFAULT_LANGUAGE, normalize_language


@dataclass(frozen=True)
class HelpSection:
    key: str
    title: str
    body: str


SECTION_TITLES = {
    "quick_start": {"es": "Inicio rápido", "en": "Quick start"},
    "what_seeing": {"es": "¿Qué estoy viendo?", "en": "What am I seeing?"},
    "key_concepts": {"es": "Conceptos clave", "en": "Key concepts"},
    "controls": {"es": "Cómo usar los controles", "en": "How to use the controls"},
    "interpretation": {"es": "Cómo interpretar los resultados", "en": "How to interpret results"},
    "limitations": {"es": "Limitaciones", "en": "Limitations"},
}


CONTEXT_HELP: dict[str, dict[str, dict[str, str]]] = {
    "overview": {
        "es": {
            "quick_start": "Seleccione una fecha y empiece por la interpretación regional antes de abrir detalles espaciales.",
            "what_seeing": "Un resumen compacto de la anomalía media, cobertura válida, área cálida y representatividad espacial.",
            "key_concepts": "La media regional resume el área; la representatividad indica cuánto oculta o conserva la estructura espacial.",
            "controls": "La fecha cambia el diagnóstico. En modo Avanzado puede ajustar el periodo, umbral y ventana de persistencia.",
            "interpretation": "Lea primero el signo y magnitud de la anomalía, luego verifique cobertura y representatividad.",
            "limitations": "Un promedio regional no describe por sí solo la ubicación, forma ni evolución de las anomalías.",
        },
        "en": {
            "quick_start": "Select a date and begin with the regional interpretation before opening spatial detail.",
            "what_seeing": "A compact summary of mean anomaly, valid coverage, warm area, and spatial representativeness.",
            "key_concepts": "The regional mean summarizes the area; representativeness indicates how much spatial structure it preserves or hides.",
            "controls": "The date changes the diagnosis. Calculation settings expose the period, threshold, and persistence window.",
            "interpretation": "Read anomaly sign and magnitude first, then check coverage and representativeness.",
            "limitations": "A regional average alone does not describe anomaly location, shape, or evolution.",
        },
    },
    "maps": {
        "es": {
            "quick_start": "Compare la TSM actual con la anomalía y el cambio diario usando la misma fecha.",
            "what_seeing": "Campos espaciales de TSM, anomalía, anomalía estandarizada, cambio diario y persistencia.",
            "key_concepts": "Los límites de color son fijos entre fechas para permitir comparaciones visuales consistentes.",
            "controls": "Cambie la fecha en la barra lateral; los límites geográficos permanecen fijos en Niño 1+2.",
            "interpretation": "Use la barra de color y la distribución espacial, no solo el color más intenso del mapa.",
            "limitations": "Celdas blancas pueden ser tierra o datos faltantes; los mapas no son pronósticos.",
        },
        "en": {
            "quick_start": "Compare current SST with anomaly and daily change for the same date.",
            "what_seeing": "Spatial fields of SST, anomaly, standardized anomaly, daily change, and persistence.",
            "key_concepts": "Color limits are fixed between dates to support consistent visual comparison.",
            "controls": "Change the date in the sidebar; the Niño 1+2 geographic extent remains fixed.",
            "interpretation": "Use the colorbar and spatial distribution, not only the most intense map color.",
            "limitations": "White cells may be land or missing data; these maps are not forecasts.",
        },
    },
    "time_series": {
        "es": {
            "quick_start": "Elija un periodo y compare el valor diario con su promedio móvil de siete días.",
            "what_seeing": "La evolución temporal de TSM, anomalías, extensión cálida, máximos y centroides.",
            "key_concepts": "Las líneas no deben interpretarse a través de fechas faltantes; cada métrica tiene unidades propias.",
            "controls": "El periodo limita la ventana visible sin recalcular los productos científicos.",
            "interpretation": "Busque duración y consistencia, no un único pico aislado.",
            "limitations": "La serie regional reduce la estructura espacial a uno o pocos valores diarios.",
        },
        "en": {
            "quick_start": "Choose a period and compare each daily value with its seven-day mean.",
            "what_seeing": "Temporal evolution of SST, anomalies, warm extent, maxima, and centroids.",
            "key_concepts": "Lines should not be interpreted across missing dates; each metric has its own units.",
            "controls": "The period limits the visible window without recalculating scientific products.",
            "interpretation": "Look for duration and consistency rather than one isolated peak.",
            "limitations": "The regional series reduces spatial structure to one or a few daily values.",
        },
    },
    "spatial_behaviour": {
        "es": {
            "quick_start": "Revise persistencia, trayectoria del centro cálido y perfiles latitudinales y longitudinales.",
            "what_seeing": "Cómo se organiza y desplaza la anomalía dentro de la región.",
            "key_concepts": "El centroide es un centro ponderado de la señal cálida, no la ubicación de una parcela de agua.",
            "controls": "El umbral y la ventana de persistencia determinan qué señal espacial se resume.",
            "interpretation": "Combine trayectoria, persistencia y perfiles; ninguna vista es suficiente por sí sola.",
            "limitations": "El centroide no existe cuando ninguna celda supera el umbral y puede saltar entre núcleos separados.",
        },
        "en": {
            "quick_start": "Review persistence, warm-centroid trajectory, and latitude/longitude profiles.",
            "what_seeing": "How the anomaly is organized and moves within the region.",
            "key_concepts": "The centroid is a weighted center of the warm signal, not the location of a water parcel.",
            "controls": "The threshold and persistence window determine which spatial signal is summarized.",
            "interpretation": "Combine trajectory, persistence, and profiles; no single view is sufficient.",
            "limitations": "The centroid is undefined when no cell exceeds the threshold and may jump between separate cores.",
        },
    },
    "quality_representativeness": {
        "es": {
            "quick_start": "Lea la clase, cobertura válida y explicación automática antes de revisar métricas detalladas.",
            "what_seeing": "Una evaluación nominal de cuánto representa la media regional a las condiciones espaciales.",
            "key_concepts": "Las clases no son una escala de severidad; el puntaje de evidencia no es una probabilidad.",
            "controls": "La fecha selecciona la evaluación diaria. El modo Avanzado muestra reglas y series de diagnóstico.",
            "interpretation": "Una clase mixta o distribuida advierte que la media puede ocultar contrastes importantes.",
            "limitations": "La clasificación depende de umbrales configurados y de la cobertura espacial disponible.",
        },
        "en": {
            "quick_start": "Read the class, valid coverage, and automatic explanation before detailed metrics.",
            "what_seeing": "A nominal assessment of how well the regional mean represents spatial conditions.",
            "key_concepts": "Classes are not a severity scale; the evidence score is not a probability.",
            "controls": "The date selects the daily assessment. Expanders provide rules and diagnostic series.",
            "interpretation": "A mixed or distributed class warns that the mean may hide important contrasts.",
            "limitations": "Classification depends on configured thresholds and available spatial coverage.",
        },
    },
    "thermal_events": {
        "es": {
            "quick_start": "Empiece por el resumen del evento; abra parches, tracks y familias solo si necesita el detalle espacial.",
            "what_seeing": "Cuatro niveles distintos: evento regional, parches diarios, tracks no ramificados y familias con linaje.",
            "key_concepts": "Evento univariado != parche diario != track != familia de eventos.",
            "controls": "La fecha y selectores cambian la vista de productos ya calculados; no vuelven a ejecutar el tracking.",
            "interpretation": "Use el evento para contexto regional y las familias para entender continuidad, divisiones y fusiones espaciales.",
            "limitations": "Los tracks describen rasgos térmicos, no parcelas individuales de agua.",
        },
        "en": {
            "quick_start": "Begin with the event overview; open patches, tracks, and families only when spatial detail is needed.",
            "what_seeing": "Four distinct levels: regional event, daily patches, non-branching tracks, and lineage families.",
            "key_concepts": "Univariate event != daily patch != track != event family.",
            "controls": "Dates and selectors change views of cached products; they never rerun tracking.",
            "interpretation": "Use the event for regional context and families to understand spatial continuation, splits, and merges.",
            "limitations": "Tracks describe thermal features, not individual water parcels.",
        },
    },
    "data_methods": {
        "es": {
            "quick_start": "Confirme la fuente, periodo de referencia, climatología activa y advertencias metodológicas.",
            "what_seeing": "Metadatos y definiciones que permiten reproducir e interpretar el producto.",
            "key_concepts": "La climatología diaria suavizada 1991–2020 es la referencia principal.",
            "controls": "Cambie a modo Avanzado para ver parámetros internos de detección y enlace.",
            "interpretation": "Use esta sección para verificar qué método produjo cada diagnóstico.",
            "limitations": "El producto es experimental y no constituye una clasificación oficial.",
        },
        "en": {
            "quick_start": "Confirm the source, reference period, active climatology, and methodological warnings.",
            "what_seeing": "Metadata and definitions needed to reproduce and interpret the product.",
            "key_concepts": "The 1991–2020 smoothed daily climatology is the primary reference.",
            "controls": "Open the relevant methods expander to view detection and linking parameters.",
            "interpretation": "Use this section to verify which method produced each diagnostic.",
            "limitations": "The product is experimental and is not an official classification.",
        },
    },
    "export": {
        "es": {
            "quick_start": "Descargue solo el producto necesario y conserve la fecha y método en el nombre o metadatos.",
            "what_seeing": "Exportaciones reproducibles del diagnóstico seleccionado y sus métricas.",
            "key_concepts": "JSON conserva números, fechas ISO, booleanos y valores faltantes como null.",
            "controls": "Cada botón descarga un formato distinto; NetCDF conserva campos espaciales.",
            "interpretation": "Revise unidades y método antes de combinar archivos en análisis externos.",
            "limitations": "Una exportación refleja los productos almacenados al momento de la descarga.",
        },
        "en": {
            "quick_start": "Download only the needed product and retain the date and method in its name or metadata.",
            "what_seeing": "Reproducible exports of the selected diagnosis and metrics.",
            "key_concepts": "JSON preserves numbers, ISO dates, booleans, and missing values as null.",
            "controls": "Each button downloads a different format; NetCDF preserves spatial fields.",
            "interpretation": "Check units and method before combining files in external analyses.",
            "limitations": "An export reflects the cached products available at download time.",
        },
    },
}


TAB_ALIASES = {
    "Overview": "overview", "Maps": "maps", "Time series": "time_series",
    "Spatial behaviour": "spatial_behaviour",
    "Quality and representativeness": "quality_representativeness",
    "Thermal events": "thermal_events", "Data and methods": "data_methods", "Export": "export",
}


REPRESENTATIVENESS_DEFINITIONS = {
    "insufficient_coverage": {
        "es": "Cobertura insuficiente para producir una clasificación espacial confiable.",
        "en": "Insufficient valid coverage for a reliable spatial classification.",
    },
    "strong_coherent": {
        "es": "Señal regional fuerte, espacialmente uniforme y con predominio de un mismo signo.",
        "en": "Strong regional signal with spatially uniform conditions and one dominant anomaly sign.",
    },
    "strong_heterogeneous": {
        "es": "Señal regional fuerte, pero con diferencias espaciales importantes en su intensidad.",
        "en": "Strong regional signal with substantial spatial differences in intensity.",
    },
    "compensated_mixed": {
        "es": "Coexisten anomalías positivas y negativas que pueden compensarse en la media regional.",
        "en": "Positive and negative anomalies coexist and may compensate in the regional mean.",
    },
    "patch_distributed": {
        "es": "La señal está distribuida entre varios parches espaciales, sin un único núcleo dominante.",
        "en": "The signal is distributed among several spatial patches without one dominant core.",
    },
    "weak_homogeneous": {
        "es": "La señal es débil, pero presenta una configuración espacial relativamente uniforme.",
        "en": "The signal is weak but spatially relatively uniform.",
    },
}


REPRESENTATIVENESS_NOTES = {
    "es": (
        "Las clases son nominales, no una escala ordinal de severidad. El puntaje de evidencia resume "
        "la evidencia de las reglas y no es una probabilidad. La cobertura válida es la fracción ponderada "
        "del área espacial utilizable."
    ),
    "en": (
        "Classes are nominal, not an ordinal severity scale. The evidence score summarizes rule evidence "
        "and is not a probability. Valid coverage is the weighted fraction of usable spatial area."
    ),
}


EVENT_CONCEPTS = {
    "es": {
        "univariate_event": "Evento univariado: superación de un umbral detectada en una serie temporal regional.",
        "daily_patch": "Parche diario: región conectada que supera el umbral en una fecha.",
        "track": "Track: secuencia temporal máxima, no ramificada, de parches enlazados.",
        "event_family": "Familia de eventos: todos los tracks conectados mediante continuación, división, fusión o relaciones complejas de linaje.",
        "identity": "Evento univariado != parche diario != track != familia de eventos.",
        "details": (
            "Los ID de parches son locales a cada fecha. Los ID de track persisten solo dentro de segmentos "
            "no ramificados. Los ID de familia agrupan tracks relacionados. Una división ocurre cuando un "
            "parche produce varios sucesores; una fusión, cuando varios parches producen un sucesor. Los tracks "
            "térmicos no representan parcelas individuales de agua."
        ),
    },
    "en": {
        "univariate_event": "Univariate event: a threshold exceedance detected in a regional time series.",
        "daily_patch": "Daily patch: a connected threshold-exceedance region on one date.",
        "track": "Track: a maximal non-branching temporal sequence of linked patches.",
        "event_family": "Event family: all tracks connected through continuation, split, merge, or complex lineage relationships.",
        "identity": "Univariate event != daily patch != track != event family.",
        "details": (
            "Daily patch IDs are local to each date. Track IDs persist only within non-branching segments. "
            "Event-family IDs group related tracks. A split means one patch produces multiple successors; a "
            "merge means multiple patches produce one successor. Thermal tracks are not individual water parcels."
        ),
    },
}


METRIC_TOOLTIPS = {
    "evidence_score": {"es": "Evidencia de la regla activada; no es una probabilidad.", "en": "Evidence supporting the triggered rule; it is not a probability."},
    "valid_coverage": {"es": "Fracción ponderada del área espacial con datos utilizables.", "en": "Area-weighted fraction of the spatial region with usable data."},
    "mean_anomaly": {"es": "Promedio espacial ponderado de TSM menos su climatología correspondiente.", "en": "Area-weighted spatial mean of SST minus its matching climatology."},
    "spatial_standard_deviation": {"es": "Dispersión espacial ponderada de la anomalía alrededor de su media.", "en": "Area-weighted spatial spread of anomaly around its mean."},
    "sign_coherence": {"es": "Predominio espacial de un mismo signo de anomalía; varía entre 0 y 1.", "en": "Spatial dominance of one anomaly sign, ranging from 0 to 1."},
    "signal_heterogeneity_ratio": {"es": "Magnitud de la media dividida por la heterogeneidad espacial, con estabilización numérica.", "en": "Mean magnitude divided by spatial heterogeneity, with numerical stabilization."},
    "positive_fraction": {"es": "Fracción ponderada del área con anomalía positiva fuera de la banda neutral.", "en": "Weighted area fraction with positive anomaly outside the neutral band."},
    "negative_fraction": {"es": "Fracción ponderada del área con anomalía negativa fuera de la banda neutral.", "en": "Weighted area fraction with negative anomaly outside the neutral band."},
    "neutral_fraction": {"es": "Fracción ponderada del área dentro de la banda neutral alrededor de cero.", "en": "Weighted area fraction inside the neutral band around zero."},
    "dominant_patch_fraction": {"es": "Fracción del área umbral que pertenece al parche más grande.", "en": "Fraction of threshold area belonging to the largest patch."},
    "track": {"es": "Secuencia temporal máxima y no ramificada de parches enlazados.", "en": "Maximal non-branching temporal sequence of linked patches."},
    "event_family": {"es": "Conjunto de tracks conectados por continuaciones, divisiones o fusiones.", "en": "Set of tracks connected by continuations, splits, or merges."},
    "iou": {"es": "Intersección dividida por unión de dos parches consecutivos, ponderada por área.", "en": "Area-weighted intersection divided by union for two consecutive patches."},
    "link_score": {"es": "Puntaje determinístico de similitud espacial usado para aceptar un enlace; no es probabilidad.", "en": "Deterministic spatial-similarity score used to accept a link; not a probability."},
    "cumulative_severity": {"es": "Suma diaria de excedencia media multiplicada por el área del parche.", "en": "Daily sum of mean exceedance multiplied by patch area."},
    "trajectory_length": {"es": "Suma de las distancias geodésicas entre centroides consecutivos.", "en": "Sum of geodesic distances between consecutive centroids."},
    "tortuosity": {"es": "Longitud de trayectoria dividida por desplazamiento neto; no se calcula si este es cero.", "en": "Trajectory length divided by net displacement; undefined when displacement is zero."},
    "expansion_rate": {"es": "Cambio positivo de área por día calendario transcurrido.", "en": "Positive area change per elapsed calendar day."},
    "contraction_rate": {"es": "Magnitud del cambio negativo de área por día calendario transcurrido.", "en": "Magnitude of negative area change per elapsed calendar day."},
}


FILTER_HELP = {
    "event_analysis_date": {"es": "Selecciona una fecha existente en los productos; actualiza mapas y estados diarios.", "en": "Selects a date present in cached products and updates daily maps and states."},
    "source_variable": {"es": "Limita los eventos a la variable científica elegida.", "en": "Restricts events to the selected scientific source variable."},
    "threshold_type": {"es": "Limita resultados al método de umbral seleccionado.", "en": "Restricts results to the selected threshold method."},
    "direction": {"es": "Distingue superaciones por encima o por debajo del umbral.", "en": "Distinguishes exceedances above or below the threshold."},
    "event_status": {"es": "Filtra por estado de calidad o cálculo del producto.", "en": "Filters by product calculation or quality status."},
    "minimum_track_duration": {"es": "Oculta tracks con duración menor, sin recalcularlos.", "en": "Hides tracks shorter than this duration without recalculating them."},
    "minimum_track_area": {"es": "Oculta tracks cuyo máximo de área es menor al valor indicado.", "en": "Hides tracks whose maximum area is below the selected value."},
    "family_selector": {"es": "Restringe la lista de tracks a una familia de linaje.", "en": "Restricts the track list to one lineage family."},
    "track_selector": {"es": "Selecciona una secuencia no ramificada y determina su familia asociada.", "en": "Selects one non-branching sequence and identifies its associated family."},
    "reset_event_filters": {"es": "Devuelve los filtros de eventos a sus valores iniciales.", "en": "Returns event filters to their initial values."},
}


GLOSSARY = {
    "anomaly": {"es": "Anomalía: TSM observada menos la climatología correspondiente.", "en": "Anomaly: observed SST minus the matching climatology."},
    "daily_smoothed_climatology": {"es": "Climatología diaria suavizada: referencia por día calendario con muestreo circular y suavizado de 31 días.", "en": "Smoothed daily climatology: calendar-day reference with circular sampling and 31-day smoothing."},
    "valid_coverage": METRIC_TOOLTIPS["valid_coverage"],
    "spatial_heterogeneity": {"es": "Heterogeneidad espacial: variación de la anomalía entre celdas válidas.", "en": "Spatial heterogeneity: anomaly variation among valid cells."},
    "representativeness": {"es": "Representatividad: evaluación de cuánto resume la media regional las condiciones espaciales.", "en": "Representativeness: assessment of how well the regional mean summarizes spatial conditions."},
    "patch": {"es": "Parche: región conectada que cumple una definición de umbral en una fecha.", "en": "Patch: connected region satisfying a threshold definition on one date."},
    "track": METRIC_TOOLTIPS["track"],
    "event_family": METRIC_TOOLTIPS["event_family"],
    "split": {"es": "División: un parche predecesor enlaza con dos o más sucesores.", "en": "Split: one predecessor patch links to two or more successors."},
    "merge": {"es": "Fusión: dos o más predecesores enlazan con un sucesor.", "en": "Merge: two or more predecessors link to one successor."},
    "lineage": {"es": "Linaje: grafo temporal de continuaciones, divisiones y fusiones.", "en": "Lineage: temporal graph of continuations, splits, and merges."},
    "iou": METRIC_TOOLTIPS["iou"],
    "link_score": METRIC_TOOLTIPS["link_score"],
    "centroid": {"es": "Centroide: centro espacial ponderado por área o intensidad.", "en": "Centroid: spatial center weighted by area or intensity."},
    "trajectory": {"es": "Trayectoria: secuencia cronológica de centroides de un track o familia.", "en": "Trajectory: chronological sequence of track or family centroids."},
    "cumulative_severity": METRIC_TOOLTIPS["cumulative_severity"],
}


DISCLAIMER = {
    "es": "Este producto es experimental y no constituye una clasificación oficial de la magnitud de El Niño Costero.",
    "en": "This is an experimental product and does not constitute an official classification of Coastal El Niño magnitude.",
}


def context_tab_key(tab: str | None) -> str:
    if tab is None:
        return "overview"
    return TAB_ALIASES.get(tab, tab if tab in CONTEXT_HELP else "overview")


def get_context_help(tab: str | None, language: str = DEFAULT_LANGUAGE) -> list[HelpSection]:
    selected = normalize_language(language)
    tab_key = context_tab_key(tab)
    content = CONTEXT_HELP[tab_key][selected]
    return [HelpSection(key, SECTION_TITLES[key][selected], content[key]) for key in SECTION_TITLES]


def representativeness_definition(classification: str, language: str = DEFAULT_LANGUAGE) -> str:
    selected = normalize_language(language)
    values = REPRESENTATIVENESS_DEFINITIONS.get(classification)
    return values.get(selected) if values else classification.replace("_", " ").capitalize()


def representativeness_notes(language: str = DEFAULT_LANGUAGE) -> str:
    return REPRESENTATIVENESS_NOTES[normalize_language(language)]


def event_concept_hierarchy(language: str = DEFAULT_LANGUAGE) -> list[str]:
    content = EVENT_CONCEPTS[normalize_language(language)]
    return [content[key] for key in ("univariate_event", "daily_patch", "track", "event_family", "identity", "details")]


def metric_tooltip(metric: str, language: str = DEFAULT_LANGUAGE) -> str:
    selected = normalize_language(language)
    values = METRIC_TOOLTIPS.get(metric)
    return values.get(selected) if values else metric.replace("_", " ").capitalize()


def filter_description(filter_name: str, language: str = DEFAULT_LANGUAGE) -> str:
    selected = normalize_language(language)
    values = FILTER_HELP.get(filter_name)
    return values.get(selected) if values else filter_name.replace("_", " ").capitalize()


def glossary(language: str = DEFAULT_LANGUAGE) -> dict[str, str]:
    selected = normalize_language(language)
    return {key: values[selected] for key, values in GLOSSARY.items()}


def scientific_disclaimer(language: str = DEFAULT_LANGUAGE) -> str:
    return DISCLAIMER[normalize_language(language)]

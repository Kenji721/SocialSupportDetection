# SSD-2026: enfoques de ML tradicional y deep learning para detección de apoyo social y objetivo del apoyo

## Panorama del problema y qué lo hace distinto a “sentiment”
La detección de apoyo social (SSD) se ha planteado explícitamente como un problema de NLP que va más allá de “sentimiento positivo/negativo”, porque busca reconocer interacciones prosociales accionables (p. ej., alentar, cuidar, admirar, ayudar, expresar solidaridad) y, crucialmente, **a quién** va dirigido ese apoyo (persona vs colectivo, y qué colectivo). citeturn12view0turn29view0turn26view2

En la literatura de ciencias sociales, “apoyo social” se vincula con la percepción de estar cuidado, valorado y perteneciente a redes de reciprocidad, y también se descompone en tipos como **emocional**, **informacional**, **estima**, **red/afiliación** y **tangible** (una tipología útil para feature engineering y análisis de errores aunque tu task sea binario). citeturn32view0turn29view0turn17view0

Un punto clave para tu *scope constraint* (“si promueve explícitamente violencia/daño → Not Support”) es que **no todo lo “pro-grupo” es prosocial**: puedes tener mensajes que “defienden” a un grupo pero mediante incitación a daño, lo que obliga a operacionalizar apoyo social como prosocialidad **sin violencia explícita** (eso ya te pone cerca de los dilemas y sesgos que se ven en moderación de contenido y en “target identity”). citeturn11view0turn30view0

## Datasets y formulaciones de tarea que ya existen y se parecen mucho a tu SSD-2026
La buena noticia: tu definición de subtareas **no es inventada en el vacío**; hay trabajos 2024–2025 que estructuran SSD de forma extremadamente similar (dos binarios + un multiclass de comunidades).

**Corpus en inglés (comentarios de video, 3 subtareas):**  
Un trabajo posterior resume un dataset en inglés recolectado de comentarios en **entity["company","YouTube","video platform, us"]**: parte de 66,272 comentarios, se depura a 42,695 tras quitar duplicados y comentarios no-inglés, y luego se seleccionan 10,000 (5k por keywords + 5k aleatorios) para anotación. Las 3 tareas son: (1) Support vs Not Support, (2) Individual vs Group, y (3) si es Group, clasificar en Nation/LGBTQ/Black people/Women/Religion/Other. citeturn32view0turn12view0  
Ese mismo trabajo reporta además el típico problema real: **las clases “pequeñas” son minúsculas** (p. ej., Women y Religion aparecen con conteos de decenas en Task 3), lo que vuelve la evaluación macro-F1 mucho más informativa que accuracy. citeturn24view2turn28view23

**Corpus en español (comentarios de video, 3 subtareas):**  
Para español hay un dataset explícito en comentarios de YouTube: 3,189 comentarios en español, donde solo 679 son Support (vs 2,510 Not Support), con tópicos como nacionalidad, comunidad negra, mujeres, religión y LGBTQ+. El paper lo presenta como el **primer dataset específicamente anotado para apoyo social en español** y prueba ML tradicional, deep learning, transformers y un LLM para lidiar con el desbalance. citeturn26view0turn32view2  
También deja claro que el dataset (como el de inglés) **no está “publicado libremente”**: se obtiene “upon request” al autor correspondiente, lo cual impacta reproducibilidad y comparabilidad. citeturn32view2turn32view0

**Datasets cercanos (te sirven para pretraining/auxiliary tasks o sanity checks):**  
- ENSYNET (salud): 6,500 oraciones anotadas en “encouragement” y “sympathy”, con baselines (LR+TF-IDF) y fine-tuning BERT; reporta que distinguir matices de apoyo emocional es difícil y que muchas instancias mezclan tipos de apoyo. citeturn16view0  
- CHQ-SocioEmo (salud, preguntas/respuestas): 1,500 pares Q/A en inglés anotados con emociones y necesidades de apoyo (incluye emocional/estima/red/tangible), y benchmarking con múltiples modelos; es útil como evidencia de **ambigüedad** y **desbalance** en anotación del apoyo. citeturn17view0  
- RedditESS (salud mental, “efectividad” del apoyo): dataset en **entity["company","Reddit","social media platform"]** que añade bucles de feedback (reacción del OP y señales comunitarias) para etiquetar apoyo “efectivo” vs no; demuestra entrenamiento de clasificadores (BERT/RoBERTa) y uso de LIWC para análisis lingüístico. Esto es relevante porque tu SSD-2026 evalúa “apoyo” pero **en deployment** normalmente importa si el apoyo es útil/seguro. citeturn31view0  
- Hope speech (positividad pro-EDI): dataset multilingüe de comentarios en YouTube que intenta detectar y promover positividad; reporta experimentos con TF-IDF + ML clásico y un CNN propuesto, subrayando que hay una tradición paralela de “prosocial speech” que se parece a SSD aunque no tenga “target-of-support” como subtarea. citeturn27view0

## Enfoques de ML tradicional que se han usado y por qué siguen siendo fuertes
En SSD (especialmente con datasets medianos/pequeños y muy desbalanceados), los enfoques tradicionales siguen apareciendo como **baselines competitivos**, sobre todo cuando combinas:

1) **Representaciones sparse** tipo TF‑IDF (n‑grams), y  
2) **features lexicográficas/psicolingüísticas** (p. ej., LIWC), más señales de emoción/sentimiento. citeturn32view0turn24view0

En el trabajo de 2025 sobre SSD en inglés, el “mejor baseline” reportado para ML tradicional incluye:
- SVM lineal + (TF‑IDF + LIWC) con macro‑F1 ≈ 0.783 para Task 1 y ≈ 0.797 para Task 2, y  
- un ensamble “soft voting” para Task 3 con macro‑F1 ≈ 0.726 usando TF‑IDF. citeturn24view0

En el trabajo en español, explícitamente se usan **regresión logística**, SVM (lineal y RBF), **XGBoost** y **Random Forest** como modelos tradicionales, y el análisis de resultados muestra un patrón típico: modelos como RF/SVM lineal tienden a aguantar mejor que LR/SVM‑RBF cuando hay minorías duras y muchas clases, aunque el performance “macro” siga siendo el cuello de botella real. citeturn26view1turn25view1turn25view2

Por qué ML clásico funciona sorprendentemente bien aquí:
- Mucho del apoyo social en comentarios tiene **marcadores explícitos** (agradecimientos, “ánimo”, “orgullo”, “fuerza”, bendiciones/rezos, etc.) que los n‑grams capturan rápido. ENSYNET incluso muestra que los tipos de apoyo (sympathy/encouragement) co‑ocurren y se mezclan, lo que favorece modelos lineales robustos con buena regularización. citeturn16view0  
- En datasets pequeños, un transformer puede sobreajustar si no hay cuidado; en cambio, un SVM lineal + TF‑IDF suele ser un baseline duro de tumbar. citeturn24view0turn28view6

Detalles prácticos (que en papers sí aparecen como parte del pipeline):
- Preprocesamiento estándar: deduplicación, filtrado por idioma, normalización (tokenización, lowercasing, stopwords, stemming/lemmatización), y una decisión importante: **convertir emojis/emoticonos a texto** en vez de borrarlos. citeturn24view2turn12view0  
- Métrica: con desbalance severo, se insiste en macro‑F1 para no esconder el colapso en clases raras; hay referencias explícitas a que macro‑F1 es útil precisamente porque pondera clases por igual. citeturn28view23turn28search18

## Deep learning “pre-transformer”: CNN/BiLSTM con embeddings (y cuándo sí vale la pena)
Antes (y todavía hoy en baselines), se prueban arquitecturas como CNNs y BiLSTMs con embeddings (FastText, GloVe). En el estudio en español se reportan comparaciones explícitas de CNN/BiLSTM con FastText vs GloVe y se observa que:
- BiLSTM con GloVe logra macro‑F1 ≈ 0.731 en Subtask 1 (Support vs Not Support) en su evaluación, y  
- en Subtask 2 y 3, los resultados varían y el desempeño macro tiende a reflejar lo difícil que es sostener minorías con pocos ejemplos. citeturn25view1turn26view2turn26view2

Dónde **sí** aporta este bloque (CNN/BiLSTM) en SSD-2026:
- Si tu objetivo es un modelo ligero para deployment (latencia/costo), CNN/BiLSTM puede ser un “sweet spot” entre ML clásico y transformer, dependiendo de tu infraestructura. El propio estudio en español discute el costo computacional más alto de transformers como un trade-off real para despliegue. citeturn13view0  
- Si haces aprendizaje **multitarea** o jerárquico (Support → Individual/Group → GroupClass), una BiLSTM con cabezas múltiples puede ser un baseline fuerte y barato, aunque los papers citados reportan que, en promedio, los transformers gana. citeturn26view2turn24view3

Asimismo, trabajos vecinos de “prosocialidad” como hope speech (comentarios en YouTube) reportan un CNN propuesto (con embeddings tipo T5‑Sentence) que supera a modelos clásicos en macro‑F1 para inglés y otros idiomas, reforzando que **deep learning no-transformer** puede ser competitivo cuando el diseño de representación está bien pensado. citeturn27view0

## Transformers y LLMs: el estado del arte práctico para SSD
En SSD (especialmente cuando hay matices o apoyo implícito), la línea dominante reciente es **fine‑tuning de transformers** + **técnicas de balance** + (a veces) **zero‑shot** con LLMs.

**Fine-tuning de transformers (supervisado):**  
En el trabajo 2025 del dataset en inglés, se reporta que modelos transformer (p. ej., RoBERTa-base y mBERT) son top performers en macro‑F1, con números como:
- Task 1: RoBERTa-base macro‑F1 ~0.80,  
- Task 2: RoBERTa-base macro‑F1 ~0.86,  
- Task 3: resultados altos con DistilBERT/RoBERTa, macro‑F1 reportados alrededor de ~0.79–0.80 para los mejores (dependiendo del split/configuración reportada). citeturn24view3turn24view2  

En español, se reporta un mejor desempeño para modelos tipo RoBERTa entrenados en español (p. ej., *robertuito*) y también variantes con XLM‑RoBERTa; el paper identifica explícitamente a “pysentimiento/robertuito-sentiment-analysis” como el mejor en Subtask 3 en su setup, y reporta que con dataset balanceado se alcanzan macro‑F1 ≈ 0.889 (Subtask 2) y ≈ 0.836 (Subtask 3). citeturn25view1turn26view2turn32view2

**Zero-shot con NLI/transformers (sin entrenamiento):**  
El estudio 2025 en inglés explora zero-shot con modelos NLI (DeBERTa/BART) con prompts; el rango de macro‑F1 que reporta para zero‑shot es claramente más bajo que el fine‑tuning, tanto en Task 1 como (más marcado) en Task 2/3. citeturn24view2turn24view3

**Zero-shot / prompting con LLMs (GPT‑x):**  
En inglés, se reportan experimentos con GPT‑3/GPT‑4/GPT‑4o para las 3 tareas, con GPT‑4o como el más fuerte en Task 1 (F1 reportado alrededor de 0.78 en su tabla) y con degradación en Task 3 frente a tasks más simples. citeturn24view2turn12view0  
En español, GPT‑4o se reporta como el mejor para Subtask 1 con macro‑F1 ≈ 0.853 con dataset original (desbalanceado), mientras que el balance + transformer gana en Subtask 2 y 3. citeturn26view2turn25view0turn32view2

**Balanceo/augmentación (es casi obligatorio en Task 3):**  
Aquí está una de las lecciones más “de la vida real”: en Task 3 el desbalance puede ser extremo (p. ej., en inglés se muestran conteos de Women/Religion en decenas). citeturn24view2  
Los papers recientes prueban dos familias de soluciones:
- Balanceo por clustering/undersampling: en inglés se usa K‑means para balancear (reportado como parte del método) y se comparan resultados contra el dataset “normal”. citeturn24view2turn32view0  
- Oversampling por paráfrasis con LLM: en español se usa GPT‑4o para generar paráfrasis y así aumentar clases subrepresentadas y mejorar macro‑F1 en subtareas posteriores. citeturn26view2turn32view2

Una advertencia técnica importante, conectada a tu Subtask 3: en tareas “target identity” se ha documentado que los modelos tienden a **sobre-apoyarse** en términos identitarios (minority mentions) y que la representación desbalanceada por identidad afecta robustez y equidad; estudios de augmentación “target-aware” muestran mejoras grandes en F1 al aumentar targets infrarrepresentados, pero también señalan este riesgo de sesgo por identidad. citeturn30view0turn11view0

## Estrategias bilingües español–inglés que realmente se usan en la práctica
Como estás en es/en, tienes tres rutas técnicas plausibles (y combinables). La diferencia no es ideológica: es **tamaño de datos + costo + generalización**.

**Modelos monolingües por idioma (dos modelos):**  
- Español: usar un encoder entrenado en español (p. ej., BETO o RoBERTa-español), o un modelo ya probado en tu subtask (como robertuito en el paper de español). citeturn14search4turn25view1turn32view2  
- Inglés: si el dominio es estilo Twitter, modelos tipo BERTweet suelen ayudar por pretraining in-domain; si es YouTube/foros, RoBERTa-base aparece consistentemente fuerte como baseline de transformer. citeturn14search2turn24view3  

Ventaja: maximizas rendimiento por idioma. Desventaja: mantenimiento doble y peor transferencia si un idioma tiene pocos datos (que en español es exactamente tu caso: 679 supports en 3,189). citeturn26view0

**Un solo modelo multilingüe (transferencia cruzada):**  
XLM‑RoBERTa se diseñó explícitamente para representación cross‑lingual y suele superar a mBERT en benchmarks cross‑lingual, lo que lo vuelve un candidato natural para entrenar un solo SSD para es/en (o para preentrenar en inglés y adaptar a español). citeturn14search1turn14search9  
El propio estudio en español prueba un modelo basado en XLM‑RoBERTa dentro de su suite de transformers, lo que te da evidencia directa de que entra en el “set de herramientas” en este dominio. citeturn25view1

**Traducción + entrenamiento (translate-train / translate-test):**  
No siempre se reporta como la mejor opción en papers recientes de SSD, pero en escenarios de bajo recurso es una estrategia estándar: traducir español→inglés para aprovechar más datos/recursos o traducir ambos a un pivot para consistencia. Lo que sí está bien respaldado es que el cuello de botella suele ser **representación por target** y **sesgo por términos identitarios**, más que “la traducción per se”, así que si traduces, igual necesitas controlar ese sesgo con augmentación balanceada estricta por clase/target. citeturn30view0turn28view6

En tu caso (SSD-2026), una receta racional “sin fantasías” sería: usar un multilingüe (XLM‑R) como backbone compartido, y aplicar **augmentación controlada por subtarea** (especialmente en GroupClass) para evitar que la clase “Other” o “Nation” domine por cantidad. Esto es consistente con lo que ya hacen los trabajos en español (paráfrasis con GPT‑4o) y en tareas target-aware (augmentación por identidad). citeturn26view2turn30view0

## Evaluación end-to-end, errores típicos y riesgos de sesgo en Targeted Group
Tu set de subtareas es jerárquico/condicional: Subtask 2 y 3 solo aplican si Subtask 1 predice Support. En deployment, eso significa que el error se **propaga**: si fallas en “Support”, nunca llegas a clasificar target, aunque el target classifier sea buenísimo.

Implicaciones prácticas (y por qué muchos papers reportan macro‑F1 por tarea):
- En datasets con clases raras, accuracy puede verse “bonito” mientras el modelo ignora minorías; por eso macro‑F1 se usa para revelar desempeño por clase. citeturn28view23turn26view2  
- En Task 3, los conteos bajos por clase (p. ej., Women/Religion en el corpus inglés) hacen que el modelo aprenda atajos (“keyword spotting” por identidad) en vez de patrones de apoyo; esto está alineado con hallazgos en hate speech/target identity donde se advierte sobre la dependencia de términos identitarios y la mala generalización entre targets. citeturn24view2turn11view0turn30view0  

Errores que debes esperar (y diseñar análisis):
- **Apoyo implícito vs neutro**: mensajes breves o sarcásticos donde el tono no es obvio sin contexto. ENSYNET muestra lo difícil que es separar subtipos incluso con anotación dedicada (sympathy vs encouragement co-ocurren mucho). citeturn16view0  
- **Target sin mención explícita**: un comentario puede ser supportive (“qué orgullo”) pero no decir a quién; si solo tienes el comentario (sin video/tweet padre), Subtask 2/3 se vuelve una inferencia con alta incertidumbre. El estudio 2025 en español reconoce límites de generalización por estar restringido a comentarios de YouTube y sugiere ampliar a otras plataformas para robustez. citeturn13view0turn26view1  
- **Tu regla anti-violencia**: vas a necesitar ejemplos representativos de “aparente apoyo” + incitación a daño dentro de Not Support para que el modelo no confunda “defensa” con apoyo prosocial. En corpora de hate/target identity se observa que datasets incluyen posts “supportive/counter-speech” y que el rango de severidad (hasta genocidal) cambia la naturaleza lingüística; esto te sirve para tomar en serio el diseño de negativos. citeturn30view0  

Finalmente, si tu SSD-2026 pretende “evaluación realista end‑to‑end”, el punto más fuerte de la literatura reciente no es un modelo mágico, sino el combo:
- baseline sólido (SVM TF‑IDF + LIWC),  
- transformer fine‑tuned,  
- balanceo serio por clase/target, y  
- medición macro‑F1 + análisis de sesgo por identidad. citeturn24view0turn24view3turn26view2turn30view0
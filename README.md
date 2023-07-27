# AIr-port CV

Este repositorio forma parte de la realización de un [TFM](https://oa.upm.es/75308/) que presenta una solución que integra modelos de inteligencia artificial en diversas arquitecturas para abordar una variedad de tareas relacionadas con la detección de personas, caras y mascarillas, así como el cálculo de la detección de proximidad entre individuos, todo ello sin la necesidad de una calibración previa.

## Introducción

La detección de personas, caras y mascarillas, así como la medición de la proximidad entre personas, son problemas cruciales en diversas aplicaciones, como la seguridad, la salud pública y el monitoreo del distanciamiento social. Esta solución ofrece una plataforma unificada que aprovecha la inteligencia artificial, procesamiento batch y secuencial para afrontar estos desafíos de manera eficiente y precisa.

## Características clave

- Detección de personas: El modelo implementado en este proyecto es capaz de localizar y detectar personas en imágenes y videos en tiempo real, permitiendo diversas aplicaciones de seguimiento y análisis de multitudes.

- Detección de caras: La solución cuenta con un robusto modelo para detectar rostros humanos en imágenes con alta precisión.

- Detección de mascarillas: La detección de mascarillas en rostros es un componente crítico en el contexto de la salud pública y la prevención de enfermedades. Nuestro modelo especializado cumple esta función con elevada eficacia.

- Cálculo de detección de proximidad: Mediante matemáticas y algoritmia podemos calcular la distancia entre personas en una imagen o video, facilitando el monitoreo del cumplimiento del distanciamiento social en diferentes entornos.

## Pasos para su uso
1. Descomprimir los modelos que se ubican dentro de /models dentro de esta carpeta.
2. Insertar las imágenes deseadas dentro de data/input
3. Compilar el proyecto y ejecutar main.py
4. Si es necesario hacer alguna configuración, se puede hacer desde el archivo main.py:

4.1. Cambiar el modelo

4.2. Cambiar el modo de ejecución entre streaming y batch


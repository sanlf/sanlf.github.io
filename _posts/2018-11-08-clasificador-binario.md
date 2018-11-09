---
layout: post
title:  "Experiencias desarrollando clasificador binario"
date:   2018-11-08
---


## Breve introducción:
Tuve que desarrollar un proyecto de redes neuronales convolucionales y busque proyectos en los que me pudiese
basar. Después de una busqueda no tan profunda, encontre un proyecto que se me hizo interesante y pense que
sería simple. El proyecto es un clasificador binario de imagenes de anime, indicando si la imagen es o no anime
[el proyecto donde me base](https://github.com/antonpaquin/IsItAnime). Como es únicamente un clasificador binario
supuse que no habría mucha dificultad en entenderlo y me serviría como experiencia para hacer el tipo de red
convolucionar clásico.

## Proyecto original:
Como mencione, el proyecto en el que me basé es un clasificador binario de imagenes y este está en una página
web, con un diseño muy simple, donde puedes subir una imagen o un url de una imagen y clasifica si es o no anime.
También parece estar bien ordenado en carpetas: scraper en la cual obtiene las imagenes con las que entrnó su
red; cleaning donde limpia y prepara las imagenes para ser usadas en el entrenamiento; training en donde define
el modelo de su red y la entrena; models en donde guarda el modelo que su red utiliza; deploy en el que prepara
su modelo como una función lambda de AWS (link correspondiente); website en donde guarda el codigo pertiente de
su página web y por último test donde prueba su red.


Algo que cabe mencionar es que el proyecto original entrena su modelo usando AWS, el servicio de nube de Amazon,
cosa que por su precio yo no me puedo permitir por lo que mucho código, así como lo tiene, no lo puedo
simplemente copiar y pegar y esperar que funcione. Tengo que entenderlo y traducirlo para poder usarlo.

## Cambios objetivo:
Los objetivos que originalmente me plantee fueron:


- Encontrar datos de entrenamiento por mi mismo para mis modelos
- Reusar modelos ya entrenados para disminuir el esfuerzo computacional que mi red tendrá que aplicar.
- Comparar los resultados obtenidos con los de la red del proyecto original.
- Crear una aplicación web similar a la del proyecto original pero en heroku.


En la conlusión diré cuales objetivos logré y cuales no.

## Problemas encontrados
Basandome en un proyecto ya hecho y siendo solo un clasificador binario, pense que sería un trabajo sencillo
pero me encontre con varios problemas.


Un pequeño aviso. Creo que en lo siguiente empezaré a hablar principalmente de problemas que me encontré con
el proyecto original pero no creo que la mayoría se deban a que el autor original hizo las cosas especialmente
mal sino que la mayoría se deben a mis propias limitaciones y falta de conocimiento en el tema.


0. Que herramientas usar para entrenar la red. Como el proyecto donde me basé usaba keras con tensorflow de
    backend, decidí usar lo mismo. El problema es que para poder usar tensorflow con GPU necesité:

    
    - Un GPU (obviamente)

    
    - Instalar los drivers de cuda y cudnn. Pero oficialmente para conseguir los drivers de cudnn es necesario
    registrarse como desarrollador en la página de nvidia. Para evitar esto tuve que investigar hasta encontrar
    con el driver por otro lado. Encontré una versión más antigua pero funcional del driver [en un post de
    medium](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e).

    
    - Para instalar keras use anaconda creando un ambiente virtual. Pero al instalar el paquete de tensorflow
    con soporte para gpu, tensorflow-gpu, no parecía funcionar. Después de perder tiempo revisando cual era el
    problema e investigando soluciones encontre que bajando la version de tensorflow-gpu a la 1.5 hacia
    desaparecer ese problema y por fin pude hacer que funcionara mi GPU y pudiera entrenar.


1. El proyecto original practicamente usaba AWS para todo por lo que habían muy pocas partes que podia copiar
    y pegar y que funcionaran. Este terminó siendo un gran problema que me hizo hacer mucho más trabajo de lo
    que esperaba y terminé reescribiendo mucho código para que yo lo pudiese usar.


2. De donde conseguir las imagenes fue un problema que me tuvo pensando por algunos días. El proyecto original
    usaba scrapers y conseguía imagenes de imgur, imagenet, 4chan que parece que guardaba usando AWS. En lo que
    estaba intentando entender como lo hizo recibí una sugerencia que no se me había ocurrido: buscar bases de
    datos con imagenes de anime. Eso debió haber sido lo primero que hiciera, pues en caso de haber me ahorraría
    tiempo entendiedo que es lo que hacen los scrapers y con suerte me proveeran de suficientes imagenes de anime.
    
    
    Al buscar me encontré con una [base de datos de kaggle](https://www.kaggle.com/alamson/safebooru) (y
    necesite de una cuenta para obtener los links) y los links eran a safebooru, un sitio donde artistas y
    fanaticos pueden subir imagenes que creen o que les guste con numerosos tags.

    
    Para bajar las imagenes hice un pequeño script con el que descargue imagenes durante todo un día. Gracias a
    que mi internet no es muy bueno solo pude descargar alrededor de 20 GB que fueron casi 40,000 imagenes de
    anime. entre estas imagenes se encontraban también gifs pero son trivialmente removibles.

    Una vez obtenidas esas imagenes, lo siguiente fue conseguir imagenes que no fueran de anime para entrenar la
    red. Primero revisé los links de imagenet que el proyecto original tenía pero hay muchos rotos o que no
    responden, por lo que tuve que buscar en otro lado.

    
    En mi busqueda encontre lorempicsum, un sitio con imagenes de muchas dimensiones. Especialmente importante es
    que tienen un método para descargar una imagen aleatoria de diferentes dimensiones. Escribí un script para
    descargar unas 10000 imagenes de tamaños variados para probarlo y lo deje en la noche. En la mañana lo revisé
    y ciertamente lo había descargado pero revisé si habían duplicados y lamentablemente si los habían. Con eso
    me dí cuenta que no tendría sentido seguir descargando imagenes del sitio porque para llegar al mismo número
    de imagenes que las de anime que descargué antes irremediablemente también descargaría más duplicados y
    tuve que dejarlo así. Finalmente terminé con unas 8500 imagenes de no anime.

    Con las dos clases de imagenes listas, las separé en 70% de entrenamiento y 30% de validación (sin datos de
    prueba final).

3. Especificamente como entrenar la red. Gracias al proyecto original tengo un modelo de red neuronal que puedo
    utilizar pero su modo de entrenamiento no parecía aplicable para mi caso. Entonces busque ejemplos en keras
    de redes convolucionales de clasificación binaria y encontre dos ejemplos en los que me base principalmente
    para hacer el código de entrenamiento:
    [esta](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
    y [esta](https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8?gi=2e207f6d89c8).


    Después de entrenar la red me dí cuenta que me daba resultados muy sesgados, y ahí supe que nunca
    hice un balanceo de pesos de las clases debido a que no tengo el mismo número de imagenes para cada tipo de
    clase. Para solucionarlo use el método de cálculo de pesos de clases de scikit-learn y volví a entrenar la
    red y tuve resultados aun sesgados pero mucho más aceptables.


    Con ese modelo y esos datos entrené la red y obtuve un 95% de precisión con el conjunto de datos de entrenamiento
    y un 92% con el conjunto de validación con unos 30 epochs, lo cual se me hizó aceptable en ese momento.

4. Hacer transfer learning con una red ya preentrenada y que predijera en el ambito de mi problema. Para esto
    tome las redes vgg16 y vgg19 que tiene Keras ya incluidas y con el conjunto de imagenes de entrenamiento de
    imagenet. Lo probé así nomás y claramente me daba resultados sin mucho sentido pues la red no tenía incluidas
    imagenes de anime en su entrenamiento ni tampoco en la capa de salida así que cuando intentaba predecir
    imagenes que sabía que eran de anime, obtenía clases completamente diferentes. Aunque claro, esto era
    inevitable por el objetivo de mi red, las salidas de las vgg y el conjunto de entrenamiento de imagenet.
    Después quité las últimas capas de las redes vgg y agregé al tope las últimas de la red original y lo
    entrené. Las predicciones fueron terribles ya que estaba completamente sesgado a una clase y siempre con 100%
    de seguridad (aunque obviamente la imagen no pertenecía a esa clase). Fue entonces cuando, después de investigar
    porque sucedia, intenté hacer no entrenables todas las capas convolucionales y ya obtenía resultados mucho mejores.
    Con unos 5-10 epochs entrenando las redes vgg obtuve resultados similares que con mi red entrenada 30 epochs.

5. Al entrenar las redes vgg, mi sistema se ralentizaba considerablemente. Al investigar esto, encontré que es debido
    a que mi GPU no tiene suficiente VRAM y partes del modelo se tienen que guardar en la RAM principal. El único
    arreglo que esto tiene es reducir el tamaño del modelo o comprar una tarjeta con más VRAM. Ninguna de las opciones
    suena bien para mi así que solo tuve que aguantarme.

6. La página web. Encontré una [serie de video tutoriales](https://www.youtube.com/watch?v=SI1hVGvbbZ4) de como hacer
    una aplicación en flask que usando un modelo de keras el cual prácticamente copié y funcionaba
    para un sólo ejemplo. El problema es que para que funcionara para multiples imagenes tuve que aplicar un cambio
    en el código por un error que parece tener la versión de tensorflow que usaba. Para arreglar ese error tuve que
    limpiar la sesion de keras y volver a cargar el modelo. Esto es poco óptimo pero al menos funciona.

7. Un problema que sigo sin poder solucionar es poder guardar la página con el modelo que predice en heroku. Si he
    podido correr la aplicación en flask localmente pero no en heroku.

## Resultados encontrados

Después de toda la travesía, termine con 3 modelos. Un modelo igual al modelo del proyecto original, un modelo basado
en el vgg16 y otro basado en el vgg19 para poder comparar los resultados. Además de esto, veré como se desempeña
la red del proyecto original con un conjunto de prueba (conjunto con el cual calificare a todas).


# Conclusiones

De los objetivos que me plantié, los resultados fueron:

- Encontrar datos de entrenamiento por mi mismo para mis modelos


    Logré este objetivo con relativa facilidad gracias a la base de datos de kaggle de safebooru y a lorempicsum.

    
-Reusar modelos ya entrenados para disminuir el esfuerzo computacional que mi red tendrá que aplicar.


    Los modelos vgg16 y vgg19 cumplen este objetivo, pues en menos epochs lograron resultados equivalentes a la
    red base con más epochs. Aunque es de notarse que al entrenar las redes vgg16 y vgg19, no solo necesite de la
    VRAM de mi GPU sino también del RAM de mi computadora, ralentizandola.

    
-Comparar los resultados obtenidos con los de la red del proyecto original.


    Ciertamente fue logrado. Está en la sección de resultados enocntrados.

    
-Crear una aplicación web similar a la del proyecto original pero en heroku.


    Lastimosamente no lo pude lograr. No estoy seguro de las razones por las cuales sucede.

# deCU: deconvolución con CUDAdecon para DeltaVision y SpinningDisk
Este script, basado en las librerías pyCUDAdecon y mrc, permite realizar la deconvolución por GPU (dGPU) de imágenes de microscopía capturada con microscopios DeltaVision y SpinningDisk, probados en el Centro Andaluz de Biología del Desarrollo (CABD-CSIC-UPO).

## Antes de comenzar
En la carpeta raíz se encuentran las subcarpetas y archivos siguientes:
- PSFs, y las subcarpetas para cada tipo de microscopio utilizado, que contienen los archivos ***Point Spread Function*** que funcionan como ***kernel*** del filtro de deconvolución. Los archivos están en formato .tif, con la nomenclatura **psf_{big}_{nm}**, donde ***big***, de estar presente, indica el filtro de mayor calidad (mayor tiempo de procesamiento), y ***nm*** la longitud de onda de emisión para cada PSF.
- deconvolve.py, la aplicación para dGPU, ejecutable desde un terminal de comandos en el entorno Conda adecuado

## Cómo instalar
1. Debes asegurarte de contar con una GPU compatible: la deconvolución utiliza los núcleos CUDA de las GPU Nvidia modernas, habiéndose probado con éxito una Nvidia GT 1030 LP, que se considerará requisito mínimo. 
2. [Instala los drivers CUDA](https://developer.nvidia.com/cuda-downloads) compatibles con tu GPU
3. [Instala Python 3.7 o superior](https://www.python.org/downloads/)
4. [Instala Conda](https://docs.conda.io/en/latest/miniconda.html), en cualquiera de sus distribuciones
5. [Instala Git](https://git-scm.com/downloads)
6. [Instala pyCUDAdecon](https://github.com/tlambert03/pycudadecon)
7. Dentro del entorno Conda para pyCUDAdecon, clona el repositorio en tu equipo
```
conda activate decon_env
git clone https://github.com/danilexn/deCU
```
8. Instala las dependencias para deCU
```
pip install tqdm numpy tifffile mrc argparse
```
9. ¡deCU ya ha sido correctamente configurado!

## Deconvolución por GPU (dGPU)
**IMPORTANTE**: La dGPU sólo es válida para imágenes z-stack, no para proyecciones máximas. Sí son válidos archivos con un solo punto temporal.

### Deconvolución de archivos .dv (para DeltaVision)
1. Ejecutar ***Windows PowerShell***. El terminal está en el entorno ***base*** de Conda.
2. Cambiar el entorno Conda a ***decon_env*** 
```
(base) PS C:\Users\admincabd> conda activate decon_env
(decon_env) PS C:\Users\admincabd>
```
3. Navega a la carpeta donde se encuentre ***deconvolve.py***
```
(decon_env) PS C:\Users\admincabd> cd .\Desktop\Deconvolución\
```
4. El comando para ejecutar la aplicación de deconvolución es el siguiente:
```
python .\deconvolve.py --source [RUTA/ARCHIVO.dv] --psf [RUTA/PSF.tif]
```
Los archivos (entre []), deben ser provistos con ruta relativa o absoluta, separados por espacio. Para seleccionar varios archivos, el comando quedaría como:
```
python .\deconvolve.py --source (get-item [RUTA/*.dv]) --psf [RUTA/PSF.tif]
```
Cuando se especifica el *cmdlet* ***get-item***, se está especificando que la ruta vendrá dada por un modificador con *wildcards*. Si indicamos como *.dv, estaremos pasando el comando para leer todos los archivos con extensión .dv de una carpeta.
El orden de los canales debe ser respetado a la hora de indicar los PSF. Si un archivo tiene 2 canales (460nm y 618nm), hay que indicar los PSF, separados por espacio y en su ruta absoluta o relativa, en ese mismo orden. Para omitir uno o varios canales, hay que indicar la posición del correspondiente archivo PSF con barra baja (_) en el terminal. Pueden seleccionarse cualquiera de los PSF, estén marcados o no como ***big***. 
5. Una vez ejecutemos el comando, a la finalización del programa deben haberse generado los ficheros en el directorio donde nos encontremos (en este caso, el directorio ***Deconvolución***). **Asegúrese de seleccionar los PSF correctos para el tipo de microscopio**.

### Deconvolución de archivos tiff (para SpinningDisk u otros .tiff)
#### Para cualquier .tiff
Con los mismos pasos 1-3 anteriores, es posible realizar la deconvolución de archivos .tif/.tiff, con un orden de stacks T-Z-W-XY (tipo DeltaVision).
4. A la hora de introducir el comando para la ejecución del programa, hay que tener en cuenta algunas especificaciones:
```
python .\deconvolve.py --source [RUTA/ARCHIVO.tif(f)] --psf [RUTA/PSF.tif] --xyimage N --zimage N -w N1 N2 N3
```
El argumento *--xyimage* sirve para especificar el tamaño de pixel en micras de la imagen a deconvolucionar. El argumento *--zimage* indica la separación entre los Z-stacks, también en micras. El argumento *-w* viene seguido de tantos números como canales haya en la imagen, representando la longitud de onda de emisión para cada uno de ellos.
El resto de pasos (5) y de posibles modificaciones a parámetros son equivalentes en el caso DV.

#### Para SpinningDisk (estructura por carpetas)
Con los mismos pasos 1-3 anteriores, es posible realizar la deconvolución de datos obtenidos del SpinningDisk, con una estructura de archivos Z-stack individuales por tiempo, y en carpetas individuales para cada canal.
4. A la hora de introducir el comando para la ejecución del programa, hay que tener en cuenta algunas especificaciones:
```
python .\deconvolve.py --source [RUTA_CARPETA] --psf [RUTA/PSF.tif] --xyimage N --zimage N -w N1 N2 ... Nn --spinning M1 M2 ... Mn
```
Hay que tener en cuenta que, en este caso, *--source* no será un fichero de imagen, sino una carpeta o carpetas que contendrán dentro la estructura de datos típica para la microscopía por SpinningDisk. El argumento *--xyimage* sirve para especificar el tamaño de pixel en micras de la imagen a deconvolucionar. El argumento *--zimage* indica la separación entre los Z-stacks, también en micras. El argumento *-w* viene seguido de tantos números como canales haya en la imagen, representando la longitud de onda de emisión para cada uno de ellos. El argumento *--spinning* viene seguido de los nombres de los canales (coincidencia con el nombre de las carpetas) para el tipo de datos a tratar. **Asegúrese de seleccionar los PSF correctos para el tipo de microscopio**.
El resto de pasos (5) y de posibles modificaciones a parámetros son equivalentes en el caso DV y TIFF.

Como resultado, se generan los archivos como {nombre_orginal}_{longitud}_{D3D}.[dv|tiff] y {nombre_orginal}.[dv|tiff].log, registro de toda la actividad durante el proceso de deconvolución.

### Otros parámetros modificables (DV y TIFF)
Consultando el comando de ayuda *--help*, podrás encontrar una lista de todos los comandos así como sus valores por defecto:
- *--project*, realiza la proyección máxima de un Z-Stack, para todos los planos. Si se especifica el argumento *--planes min MAX*, donde min y MAX son planos N-1 (empezando por el 0), se indicará un rango de Z-stacks a proyectar
- *--na*, indica la apertura numérica del objetivo, que debe coincidir con la del PSF.
- *--refractive*, indica el índice de refracción del aceite o medio de inmersión utilizado.
- *-i N*, donde N es el número de iteraciones para el algoritmo de Richardson-Lucy implementado.
- *--zfilter* y *--xyfilter*, los tamaños o separación en micras del stack Z y de los píxeles XY, para los archivos PSF indicados.

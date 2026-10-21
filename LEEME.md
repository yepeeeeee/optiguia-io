# OPTIGUÍA IO — Guía para trabajar y entregar

Integrantes: Yeferson David Hernández Tax y Katherine Iovanna Melendez Olivares.
Asignatura: Investigación de Operaciones. Universidad Mariano Gálvez de Guatemala.
Entrega indicada en el enunciado: domingo 25 de octubre de 2026.

## 1. Qué incluye

- motor.py: fórmulas, validaciones y algoritmos, sin interfaz.
- app.py: pantallas y explicaciones en Streamlit.
- ejemplos.py: diez enunciados con datos que se pueden cambiar.
- requirements.txt: bibliotecas necesarias.
- verificar.py: comprobaciones reproducibles del motor.

No usa Gemini, OpenAI, claves de API, Supabase ni una base de datos externa.
No cobra por consultas. La publicación depende de los límites del alojamiento que elijas.
La práctica se conserva durante la sesión; no hay historial permanente ni cuentas de usuario.
No es un chat de IA: trabaja con números y formularios. El documento permite IA O programación.

## 2. Ejecutarlo en Windows / VS Code

1. Descarga y extrae el ZIP en una carpeta, por ejemplo Documentos/OPTIGUIA_IO.
2. En VS Code abre Archivo > Abrir carpeta y selecciona OPTIGUIA_IO.
3. Abre Terminal > Nueva terminal.
4. Instala las bibliotecas (recomendado Python 3.11 o 3.12):

```bash
python -m pip install -r requirements.txt
```

5. Ejecuta:

```bash
python -m streamlit run app.py
```

6. Abre la dirección Local URL que aparece en la terminal. Deja la terminal abierta.
   Si Windows no reconoce python pero sí el lanzador py, reemplaza python por py.
7. Para detenerlo presiona Ctrl+C en la terminal.

No pegues este código encima del anterior: ejecuta app.py desde la carpeta nueva,
manteniendo juntos todos los archivos.

## 3. Prueba inicial

En Resolver > Programación lineal selecciona la prueba del docente y pulsa Formular y resolver.
Debe dar x1=40, x2=20 y Z=2200.
Comprueba: 2(40)+20=100; 40+2(20)=80; 40(40)+30(20)=2200.

Luego cambia los datos para demostrar que no son respuestas fijas.
Para comprobar el motor desde la terminal:

```bash
python verificar.py
```

## 4. Cómo ingresar un problema nuevo

Objetivo 40x1+30x2: escribe 40,30.
Restricción 2x1+x2<=100: escribe 2,1,<=,100.
Segunda restricción x1+2x2<=80: en otra línea escribe 1,2,<=,80.
Usa <=, >= o =. Usa punto decimal. Las variables son no negativas.

En modelación describe lo que representa cada variable; el programa construye las
ecuaciones con los datos que ingresas. Tú debes traducir el enunciado a coeficientes.
No promete interpretar preguntas abiertas ni lenguaje natural.

## 5. Cobertura y límites para explicar al docente

| Módulo | Funcionamiento |
|---|---|
| Modelación | Plantillas de máximo y mínimo; nombres de variables, objetivo y restricciones desde coeficientes. |
| Programación lineal | Máximo y mínimo, restricciones <=, >= y =; hasta 10 variables continuas y 30 restricciones. Usa SciPy/HiGHS y verifica factibilidad. |
| Método gráfico | Dos variables; fronteras, sombreado de región factible, óptimo e intersecciones factibles. La ventana puede recortar regiones ilimitadas. |
| Simplex | Tablas y pivotes reales del simplex primal básico con holguras y regla de Bland. Requiere restricciones <= y límites no negativos; no incluye fase I, dos fases o gran M. Admite mínimo transformándolo a máximo de -Z. |
| Transporte | Optimización con igualdades de oferta/demanda; plan y comprobación. Balanceo ficticio opcional y explícito con costo cero. No desarrolla esquina noroeste, Vogel o MODI paso a paso. |
| Asignación | Matrices cuadradas hasta 30×30; mínimo costo o máximo beneficio y una tarea por persona. Usa SciPy; no desarrolla reducciones del método húngaro manual. |
| Inventarios | EOQ básico, ROP, pedidos/año y costo relevante. Demanda constante y unidades de tiempo consistentes. |
| Práctica | Datos nuevos de los seis módulos; evaluación numérica y procedimiento. No califica redacción ni demostraciones libres. |

En modelos de producción, las cantidades son continuas; si se exigen unidades enteras
no basta con redondear. Ese caso requiere programación entera, fuera de esta versión.
Las respuestas se calculan por algoritmos. No hay un modelo generativo ni entrenamiento de IA.
Confirma con el docente si espera traducción automática de texto o un método manual
específico adicional a las funciones básicas descritas en el enunciado.

## 6. Publicarlo mediante un enlace

Ruta sugerida: GitHub + Streamlit Community Cloud.

1. Crea un repositorio público en GitHub, por ejemplo optiguia-io.
2. En GitHub usa Add file > Upload files. Sube app.py, motor.py, ejemplos.py,
   requirements.txt y verificar.py en la raíz del repositorio. Confirma con Commit changes.
3. Entra a https://share.streamlit.io/ e inicia sesión para administrar tu publicación.
4. Elige Create app / Deploy an app, selecciona el repositorio y su rama.
5. En Main file path escribe app.py. Elige Python 3.11 o 3.12 si se solicita.
6. Publica con Deploy y espera a que instale requirements.txt.
7. Configura la aplicación como pública en Sharing / acceso.
8. Copia el enlace terminado en .streamlit.app y ábrelo en una ventana de incógnito.
   Debes poder resolver ejercicios sin iniciar sesión. El docente no debe usar tu cuenta.
9. Verifica el enlace antes de entregar y durante la revisión; un servicio gratuito
   puede suspender aplicaciones inactivas y necesitar un tiempo de arranque.

El acceso a GitHub/Streamlit lo hace quien publica; la app no pide cuenta al visitante.
Este paquete no está publicado todavía. localhost NO es el enlace de entrega.

Documentación de publicación: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
Documentación del motor: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

## 7. Evidencias e informe de máximo tres páginas

No se incluyen capturas inventadas ni resultados de una publicación inexistente.
Después de ejecutarlo, captura tus pantallas reales y prepara:

- Página 1: nombre OPTIGUÍA IO, integrantes, sección, objetivo, herramientas
  (Python, Streamlit, SciPy, NumPy, pandas, Matplotlib) y enlace público final.
- Página 2: funciones de los seis módulos y dos capturas legibles de funcionamiento.
- Página 3: tabla de pruebas con entrada, resultado esperado, resultado observado y
  estado, más una captura de gráfica o simplex y límites de la aplicación.

Pruebas mínimas recomendadas:
1. Prueba del docente: 40,20,2200.
2. Mínimo: Min 3x+2y, x+y>=10, x>=2: 2,8,22.
3. Simplex: Max 5x+4y, x+y<=5, 2x+y<=8: 3,2,23.
4. Transporte del ejemplo: costo 90.
5. Asignación del ejemplo: costo 7.
6. EOQ del ejemplo 9: 200 y ROP 20.
7. ROP del ejemplo 10: 130.
8. Inviable: x<=1 y x>=2: advertencia.
9. No acotado: Max x con -x<=0: advertencia.
10. Una práctica con respuesta correcta y otra incorrecta.
11. Enlace público probado en incógnito y con datos nuevos.

Ambos integrantes deben poder explicar la entrada de coeficientes, cómo se verifica
una restricción, el pivote simplex, la función de costo de transporte y EOQ/ROP.

## 8. Comprobación realizada al preparar el paquete

Se probaron los 10 ejemplos y 80 casos nuevos de máximo/mínimo comparando el simplex
educativo con HiGHS, además de tablas, balanceo y entradas inválidas.
El estado de las pruebas de interfaz y las versiones probadas se detallan en VALIDACION.txt.
La publicación real y sus capturas deben completarse en tu cuenta.

"""OPTIGUÍA IO: aplicación académica sin servicios de IA externos."""
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from motor import (
    numeros, matriz, leer_modelo, resolver_pl,
    simplex, cumple, transporte, asignacion, inventario
)
from ejemplos import MODULOS, EJEMPLOS



def aplicar_diseno():
    st.markdown("""
    <style>
    .stApp {
        background: #f5f6f8;
        color: #202b38;
    }

    [data-testid="stHeader"] {
        background: #f5f6f8;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.8rem;
        padding-bottom: 3.5rem;
    }

    h1, h2, h3 {
        color: #182636;
        letter-spacing: -.035em;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 650 !important;
    }

    h2 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    p, li {
        line-height: 1.65;
    }

    [data-testid="stCaptionContainer"] {
        color: #647181;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #dfe5ea;
        border-radius: 12px;
        padding: 1.6rem;
        box-shadow: 0 2px 6px #17263804;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #fafbfc;
        color: #202b38;
        border-radius: 6px;
    }

    [data-testid="stTextArea"] textarea {
        font-family: ui-monospace, Consolas, monospace;
        line-height: 1.7;
    }

    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background: #176659;
        border: 1px solid #176659;
        border-radius: 7px;
        font-weight: 600;
        min-height: 42px;
        box-shadow: none;
    }

    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: #125447;
        border-color: #125447;
    }

    [data-testid="stExpander"] {
        border-radius: 8px;
        background: #ffffff;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        background: #182a35;
        border-right: 1px solid #203743;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.6rem;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] summary {
        color: #dce4e9;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #9fb0bc;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 5px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .7rem .8rem;
        border-radius: 7px;
        background: #ffffff05;
        width: 100%;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #ffffff0d;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff04;
        border: 1px solid #ffffff20;
        margin-top: 2.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        color: #c3cfd8;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 1.9rem;
    }

    .brand-symbol {
        display: grid;
        place-items: center;
        width: 34px;
        height: 34px;
        border: 1px solid #8bada8;
        border-radius: 8px;
        color: #dcece6;
        font-size: 19px;
    }

    .brand-name {
        color: #f1f5f7;
        font-size: 21px;
        font-weight: 650;
        letter-spacing: -.6px;
    }

    .brand-sub {
        color: #99afbd;
        font-size: 11px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .eyebrow {
        color: #176659;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 650;
        text-transform: uppercase;
        margin: 0 0 10px;
    }

    .page-intro {
        color: #667382;
        font-size: 16px;
        margin: 0 0 25px;
        max-width: 660px;
    }

    .section-rule {
        border-top: 1px solid #dce3e8;
        margin: 10px 0 24px;
    }

    .problem-brief {
        background: #eef2f3;
        border-left: 3px solid #8ba59e;
        border-radius: 0 6px 6px 0;
        padding: 16px 20px;
        margin: 8px 0 24px;
        color: #384855;
        font-size: 14px;
    }

    .problem-brief strong {
        display: block;
        margin-bottom: 5px;
        color: #64757c;
        font-size: 10px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.5rem;
        }

        h1 {
            font-size: 2.05rem !important;
        }

        [data-testid="stForm"] {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def f(numero):
    return f"{float(numero):,.6f}".rstrip("0").rstrip(".")


def expresion(coeficientes):
    partes = []

    for i, valor in enumerate(coeficientes):
        if abs(valor) > 1e-12:
            signo = "-" if valor < 0 else ("+" if partes else "")
            partes.append(f"{signo} {abs(valor):g}x_{{{i+1}}}")

    return " ".join(partes) or "0"


def mostrar_modelo(c, A, b, signos, sentido, variables):
    st.subheader("Variables de decisión")
    etiquetas = [x.strip() for x in variables.split(",")]

    for i in range(len(c)):
        etiqueta = (
            etiquetas[i]
            if i < len(etiquetas) and etiquetas[i]
            else f"cantidad de la actividad {i+1}"
        )
        st.write(f"x{i+1}: {etiqueta}.")

    st.subheader("Función objetivo y restricciones")

    st.latex(
        (r"\max" if sentido == "Maximizar" else r"\min")
        + r"\ Z = "
        + expresion(c)
    )

    simbolos = {"<=": r"\le", ">=": r"\ge", "=": "="}

    for fila, limite, signo in zip(A, b, signos):
        st.latex(expresion(fila) + simbolos[signo] + f"{limite:g}")

    st.latex(r"x_i \ge 0\quad\text{para todas las variables}")

    st.caption(
        "Modelo continuo: las variables pueden ser fraccionarias. "
        "No se impone integralidad."
    )


def mostrar_solucion(c, A, b, signos, x, z):
    st.subheader("Procedimiento y comprobación")

    st.write(
        "Se optimiza el modelo sujeto a todas las restricciones "
        "y a la no negatividad. Después se sustituyen las cantidades "
        "obtenidas para verificar su factibilidad."
    )

    filas = []

    for i, (a, limite, signo) in enumerate(zip(A, b, signos), 1):
        valor = float(a @ x)

        if signo == "<=":
            margen = limite - valor
        elif signo == ">=":
            margen = valor - limite
        else:
            margen = abs(valor - limite)

        filas.append({
            "Restricción": f"R{i}",
            "Sustitución": " + ".join(
                f"({f(v)})({f(w)})" for v, w in zip(a, x)
            ),
            "Resultado": valor,
            "Signo": signo,
            "Límite": limite,
            "Holgura / exceso / error de igualdad": margen
        })

    st.dataframe(
        pd.DataFrame(filas),
        hide_index=True,
        width="stretch"
    )

    st.write(
        "En ≤ el margen indica recurso disponible; "
        "en ≥ indica exceso; en = se muestra el error numérico."
    )

    st.latex(
        "Z = "
        + " + ".join(f"({v:g})({w:.8g})" for v, w in zip(c, x))
        + f" = {z:.8g}"
    )

    st.subheader("Respuesta final e interpretación")

    st.success(
        "; ".join(f"x{i+1} = {f(v)}" for i, v in enumerate(x))
        + f"; Z = {f(z)}"
    )

    st.write(
        "Estas cantidades optimizan el objetivo del modelo ingresado. "
        "Si Z representa ganancia, es la ganancia óptima; "
        "si representa costo, es el costo óptimo. "
        "Puede haber otras soluciones óptimas equivalentes."
    )

    st.caption(
        "Las tablas simplex se desarrollan en su módulo. "
        "Este resultado se calculó con SciPy/HiGHS."
    )


def grafica(c, A, b, signos, x, z, ventana):
    limites = list(zip(A, b)) + [
        (np.array([1., 0.]), 0.),
        (np.array([0., 1.]), 0.)
    ]

    vertices = []

    for (a1, b1), (a2, b2) in itertools.combinations(limites, 2):
        sistema = np.array([a1, a2])

        if abs(np.linalg.det(sistema)) > 1e-10:
            punto = np.linalg.solve(sistema, np.array([b1, b2]))

            if (
                cumple(punto, A, b, signos)
                and not any(np.allclose(punto, p) for p in vertices)
            ):
                vertices.append(punto)

    escala = max([1., *x, *(v for p in vertices for v in p)]) * 1.3
    tope = float(ventana) if ventana > 0 else escala

    g = np.linspace(0, tope, 450)
    xx, yy = np.meshgrid(g, g)
    factible = np.ones_like(xx, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccd6dc")
    ax.tick_params(colors="#526371")

    ax.set_prop_cycle(color=[
        "#28675e", "#698399", "#ad8b60", "#7e7897", "#708c75"
    ])

    hay_igualdad = False

    for i, (fila, limite, signo) in enumerate(zip(A, b, signos), 1):
        valores = fila[0] * xx + fila[1] * yy

        if signo == "<=":
            factible &= valores <= limite + 1e-9
        elif signo == ">=":
            factible &= valores >= limite - 1e-9
        else:
            hay_igualdad = True
            factible &= np.abs(valores - limite) <= max(
                1e-8, tope * np.linalg.norm(fila) / 450
            )

        if abs(fila[1]) > 1e-12:
            ax.plot(
                g,
                (limite - fila[0] * g) / fila[1],
                label=(
                    f"R{i}: {fila[0]:g}x1 + "
                    f"{fila[1]:g}x2 {signo} {limite:g}"
                )
            )
        elif abs(fila[0]) > 1e-12:
            ax.axvline(
                limite / fila[0],
                label=f"R{i}",
                linestyle="--"
            )

    ax.contourf(
        xx, yy, factible.astype(int),
        levels=[0.5, 1.5],
        colors=["#b0e0be"],
        alpha=0.8
    )

    if np.linalg.norm(c) > 0:
        if abs(c[1]) > 1e-12:
            ax.plot(
                g,
                (z - c[0] * g) / c[1],
                ":",
                color="black",
                label="Objetivo en el óptimo"
            )
        else:
            ax.axvline(
                z / c[0],
                color="black",
                linestyle=":",
                label="Objetivo en el óptimo"
            )

    ax.scatter(
        [x[0]], [x[1]],
        color="#176659",
        edgecolors="white",
        linewidths=1.5,
        s=90,
        label=f"Óptimo ({f(x[0])}, {f(x[1])})",
        zorder=5
    )

    ax.set(
        xlim=(0, tope),
        ylim=(0, tope),
        xlabel="x1",
        ylabel="x2",
        title="Región factible y solución óptima"
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Verde: región factible dentro de la ventana mostrada. "
        "La gráfica puede recortar regiones ilimitadas. "
        + (
            "En igualdades el sombreado es una aproximación "
            "visual de una línea o punto."
            if hay_igualdad else ""
        )
    )

    if vertices:
        tabla = pd.DataFrame([
            {"x1": p[0], "x2": p[1], "Z": float(c @ p)}
            for p in vertices
        ])

        st.write(
            "Intersecciones factibles: se resuelven pares de fronteras "
            "y se descartan puntos que incumplen restricciones."
        )

        st.dataframe(tabla, hide_index=True, width="stretch")


def mostrar_simplex(c, A, b, signos, sentido):
    x, z, pasos = simplex(c, A, b, signos, sentido)
    n, m = len(c), len(b)

    nombres = (
        [f"x{i+1}" for i in range(n)]
        + [f"s{i+1}" for i in range(m)]
    )

    st.write(
        "1. Se agrega una holgura s a cada restricción ≤. "
        "Las holguras forman la base inicial. "
        "Para minimizar se maximiza -Z; "
        "al final se evalúa el objetivo original."
    )

    for k, paso in enumerate(pasos):
        titulo = f"Tabla {k}"

        if paso["pivote"] is None:
            titulo += " — óptima"

        with st.expander(titulo, expanded=True):
            indice = (
                [nombres[i] for i in paso["base"]]
                + ["Objetivo transformado"]
            )

            st.dataframe(
                pd.DataFrame(
                    paso["tabla"],
                    columns=nombres + ["LD"],
                    index=indice
                ).round(6),
                width="stretch"
            )

            if paso["pivote"] is not None:
                sale, entra = paso["pivote"]
                tabla = paso["tabla"]

                st.write(
                    f"Entra {nombres[entra]}: se elige la primera "
                    f"columna con costo reducido negativo "
                    f"(regla de Bland). "
                    f"Sale {nombres[paso['base'][sale]]}. "
                    f"Pivote = {f(tabla[sale, entra])}."
                )

                razones = [
                    {
                        "Fila": nombres[paso["base"][i]],
                        "LD / coeficiente":
                            tabla[i, -1] / tabla[i, entra]
                    }
                    for i in range(m)
                    if tabla[i, entra] > 1e-9
                ]

                st.dataframe(pd.DataFrame(razones), hide_index=True)

                st.write(
                    "Se usa la menor razón no negativa; "
                    "los empates se resuelven por índice de variable básica. "
                    "Se divide la fila saliente entre el pivote. "
                    "En cada otra fila: fila nueva = fila anterior "
                    "− coeficiente de la columna pivote "
                    "× fila pivote normalizada."
                )
            else:
                st.write(
                    "No quedan costos reducidos negativos: "
                    "la base es óptima para el objetivo transformado."
                )

    return x, z


def pantalla_pl(modulo, ejemplo, clave):
    st.caption(
        "Escribe datos nuevos. No se interpreta automáticamente "
        "un enunciado libre: tú defines sus coeficientes."
    )

    with st.form("pl_" + clave):
        sentido = st.selectbox(
            "Objetivo",
            ["Maximizar", "Minimizar"],
            index=0 if ejemplo.get("sentido") == "Maximizar" else 1
        )

        variables = st.text_input(
            "¿Qué representa cada variable? (separa con comas)",
            ejemplo.get("variables", "cantidad x, cantidad y")
        )

        objetivo = st.text_input(
            "Coeficientes de la función objetivo",
            ejemplo.get("c", "40,30")
        )

        st.caption(
            "40,30 significa Z = 40x1 + 30x2. "
            "Usa punto decimal, por ejemplo 0.1."
        )

        restricciones = st.text_area(
            "Restricciones: una por línea",
            ejemplo.get(
                "restricciones",
                "2,1,<=,100\n1,2,<=,80"
            ),
            height=140
        )

        st.caption(
            "2,1,<=,100 significa 2x1 + x2 ≤ 100. "
            "Para tres variables: 2,1,3,<=,100. "
            "Todas las variables son no negativas."
        )

        ventana = 0

        if modulo == "Método gráfico":
            ventana = st.number_input(
                "Límite de ejes de la gráfica (0 = automático)",
                min_value=0.,
                value=0.
            )

        enviado = st.form_submit_button(
            "Formular y resolver",
            type="primary"
        )

    if enviado:
        st.session_state.pop("resultado_pl", None)

        try:
            c, A, b, signos = leer_modelo(objetivo, restricciones)

            if modulo == "Método gráfico" and len(c) != 2:
                raise ValueError(
                    "El método gráfico requiere exactamente dos variables."
                )

            if modulo == "Método simplex":
                x, z, _ = simplex(c, A, b, signos, sentido)
            else:
                x, z = resolver_pl(c, A, b, signos, sentido)

            st.session_state["resultado_pl"] = (
                clave, c, A, b, signos,
                sentido, variables, x, z, ventana
            )

        except ValueError as error:
            st.error(str(error))

    resultado = st.session_state.get("resultado_pl")

    if resultado and resultado[0] == clave:
        (
            _, c, A, b, signos,
            sentido, variables, x, z, ventana
        ) = resultado

        mostrar_modelo(c, A, b, signos, sentido, variables)

        if modulo == "Modelación":
            st.write(
                "Traducción: cada fila contiene el consumo o aporte "
                "de una unidad de cada actividad. "
                "El signo ≤ expresa un máximo disponible; "
                "≥, un mínimo requerido; =, una cantidad exacta. "
                "Los coeficientes del objetivo indican costo o beneficio "
                "por unidad. La interpretación depende de los datos "
                "y nombres que ingresaste."
            )

        if modulo == "Método simplex":
            mostrar_simplex(c, A, b, signos, sentido)

        mostrar_solucion(c, A, b, signos, x, z)

        if modulo == "Método gráfico":
            grafica(c, A, b, signos, x, z, ventana)


def pantalla_distribucion(ejemplo, clave):
    tipo = st.radio(
        "Problema",
        ["Transporte", "Asignación"],
        index=0 if ejemplo.get("tipo", "Transporte") == "Transporte" else 1,
        horizontal=True
    )

    with st.form("distribucion_" + clave + "_" + tipo):
        entrada = st.text_area(
            "Matriz de costos: filas por origen/persona; "
            "columnas por destino/tarea",
            ejemplo.get("costos", "2,6\n5,1")
        )

        st.caption(
            "Separa columnas con comas y filas con Enter. "
            "Usa punto decimal."
        )

        if tipo == "Transporte":
            oferta = st.text_input(
                "Oferta por origen",
                ejemplo.get("oferta", "20,30")
            )

            demanda = st.text_input(
                "Demanda por destino",
                ejemplo.get("demanda", "25,25")
            )

            balancear = st.checkbox(
                "Autorizar balanceo ficticio con costo cero "
                "(oferta sin usar o demanda no cubierta)"
            )

            maximizar = False

        else:
            maximizar = st.checkbox(
                "Maximizar beneficios en vez de minimizar costos"
            )

        enviado = st.form_submit_button(
            "Calcular distribución óptima",
            type="primary"
        )

    identificador = clave + "_" + tipo

    if enviado:
        st.session_state.pop("resultado_distribucion", None)

        try:
            costos = matriz(entrada)

            if tipo == "Transporte":
                datos = transporte(
                    costos,
                    numeros(oferta),
                    numeros(demanda),
                    balancear
                )
            else:
                datos = (
                    costos,
                    maximizar,
                    asignacion(costos, maximizar)
                )

            st.session_state["resultado_distribucion"] = (
                identificador, datos
            )

        except ValueError as error:
            st.error(str(error))

    res = st.session_state.get("resultado_distribucion")

    if not res or res[0] != identificador:
        return

    st.subheader("Variables, función objetivo y restricciones")

    if tipo == "Transporte":
        plan, z, costos, oferta, demanda, artificial = res[1]

        st.write("xᵢⱼ = unidades enviadas del origen i al destino j.")

        st.latex(
            r"\min Z=\sum_i\sum_j c_{ij}x_{ij},"
            r"\quad \sum_j x_{ij}=O_i,"
            r"\quad \sum_i x_{ij}=D_j,"
            r"\quad x_{ij}\ge0"
        )

        st.subheader("Procedimiento y sustitución")

        st.write(
            f"1. Oferta total = {f(oferta.sum())}; "
            f"demanda total = {f(demanda.sum())}. "
            "2. Se plantea una igualdad por origen y por destino. "
            "3. Se minimiza la suma costo × cantidad mediante "
            "programación lineal (HiGHS). "
            "4. Se comprueban sumas por filas y columnas."
        )

        if artificial:
            st.warning(artificial)

        tabla = pd.DataFrame(
            plan,
            index=[f"Origen {i+1}" for i in range(len(oferta))],
            columns=[f"Destino {j+1}" for j in range(len(demanda))]
        )

        st.dataframe(tabla, width="stretch")

        st.write(
            "Enviado por origen:",
            dict(zip(tabla.index, map(float, plan.sum(1))))
        )

        st.write(
            "Recibido por destino:",
            dict(zip(tabla.columns, map(float, plan.sum(0))))
        )

        terminos = [
            f"({f(costos[i, j])} × {f(plan[i, j])})"
            for i, j in np.ndindex(plan.shape)
            if plan[i, j] > 1e-8
        ]

        st.write("Z = " + " + ".join(terminos) + " = " + f(z))

        st.success(
            f"Costo mínimo = {f(z)}. "
            "Utiliza los envíos de la tabla; "
            "cero significa que no se usa esa ruta."
        )

    else:
        costos, maximizar, (filas, columnas, z) = res[1]

        st.write(
            "xᵢⱼ = 1 si se asigna la persona i a la tarea j; "
            "0 en otro caso."
        )

        st.latex(
            (r"\max" if maximizar else r"\min")
            + r"\ Z=\sum_i\sum_j c_{ij}x_{ij},"
            r"\quad\sum_j x_{ij}=1,"
            r"\quad\sum_i x_{ij}=1,"
            r"\quad x_{ij}\in\{0,1\}"
        )

        st.subheader("Procedimiento y sustitución")

        st.write(
            "Se busca la combinación de menor costo o mayor beneficio "
            "con el algoritmo de asignación de SciPy. "
            "Se selecciona exactamente una tarea por persona "
            "y una persona por tarea. Esta pantalla muestra "
            "la asignación y su comprobación, no las iteraciones "
            "manuales del método húngaro."
        )

        st.dataframe(
            pd.DataFrame([
                {
                    "Persona": int(i + 1),
                    "Tarea": int(j + 1),
                    "Valor": costos[i, j]
                }
                for i, j in zip(filas, columnas)
            ]),
            hide_index=True
        )

        st.write(
            "Z = "
            + " + ".join(f(costos[i, j]) for i, j in zip(filas, columnas))
            + " = " + f(z)
        )

        st.success(
            f"Valor óptimo = {f(z)}. "
            "Cada persona y cada tarea aparecen exactamente una vez."
        )


def mostrar_inventario(datos):
    D, S, H, dias, plazo, seguridad = datos
    r = inventario(*datos)

    st.subheader("Fórmulas y sustitución")

    st.write(
        "D: demanda anual; S: costo por pedido; "
        "H: costo anual de mantener una unidad; "
        "L: plazo en días operativos; SS: stock de seguridad. "
        "Q es la cantidad por pedido."
    )

    st.latex(r"CT(Q)=\frac{D}{Q}S+\frac{Q}{2}H")

    st.write(
        "Se minimiza el costo anual relevante de ordenar y mantener. "
        "Derivando: CT′(Q) = −DS/Q² + H/2 = 0; "
        "por tanto Q² = 2DS/H."
    )

    st.latex(
        r"Q^*=\sqrt{\frac{2DS}{H}}=\sqrt{\frac{2("
        + f"{D:g})({S:g})"
        + "}{"
        + f"{H:g}"
        + "}}="
        + f"{r['q']:.8g}"
    )

    st.latex(
        r"d=\frac{D}{\text{días operativos}}="
        + f"\\frac{{{D:g}}}{{{dias:g}}}={r['demanda_diaria']:.8g}"
    )

    st.latex(
        r"ROP=dL+SS="
        + (
            f"({r['demanda_diaria']:.8g})({plazo:g})"
            f"+{seguridad:g}={r['rop']:.8g}"
        )
    )

    st.write(
        f"Pedidos/año = D/Q = {f(r['pedidos'])}; "
        f"días entre pedidos = días × Q/D = {f(r['ciclo'])}."
    )

    st.write(
        f"Ordenar: (D/Q)S = {f(r['ordenar'])}; "
        f"mantener: (Q/2)H = {f(r['mantener'])}; "
        f"total anual relevante = {f(r['costo'])}."
    )

    st.subheader("Respuesta final e interpretación")

    st.success(
        f"EOQ = {f(r['q'])} unidades por pedido. "
        f"ROP = {f(r['rop'])} unidades de posición de inventario."
    )

    st.write(
        "Realiza un pedido cuando la posición de inventario "
        "(disponible + en tránsito − pedidos pendientes) alcance el ROP. "
        "El modelo EOQ supone demanda constante, reposición instantánea "
        "al llegar el pedido, sin descuentos ni faltantes planificados. "
        "El costo mostrado excluye compra y mantenimiento del stock "
        "de seguridad. Q* es continuo; para unidades indivisibles "
        "conviene comparar enteros adyacentes."
    )


def pantalla_inventario(ejemplo, clave):
    with st.form("inv_" + clave):
        a, b = st.columns(2)

        with a:
            D = st.number_input(
                "Demanda anual D (unidades/año)",
                min_value=0.,
                value=float(ejemplo.get("D", 1200.))
            )

            S = st.number_input(
                "Costo por pedido S",
                min_value=0.,
                value=float(ejemplo.get("S", 50.))
            )

            H = st.number_input(
                "Mantenimiento H (por unidad/año)",
                min_value=0.,
                value=float(ejemplo.get("H", 3.))
            )

        with b:
            dias = st.number_input(
                "Días operativos al año",
                min_value=0.,
                value=float(ejemplo.get("dias", 300.))
            )

            plazo = st.number_input(
                "Plazo de entrega (días operativos)",
                min_value=0.,
                value=float(ejemplo.get("plazo", 5.))
            )

            seguridad = st.number_input(
                "Stock de seguridad (unidades)",
                min_value=0.,
                value=float(ejemplo.get("seguridad", 0.))
            )

        enviado = st.form_submit_button(
            "Calcular EOQ y ROP",
            type="primary"
        )

    if enviado:
        st.session_state.pop("resultado_inventario", None)

        try:
            datos = (D, S, H, dias, plazo, seguridad)
            inventario(*datos)

            st.session_state["resultado_inventario"] = (
                clave, datos
            )

        except ValueError as error:
            st.error(str(error))

    res = st.session_state.get("resultado_inventario")

    if res and res[0] == clave:
        mostrar_inventario(res[1])


def nueva_practica(modulo):
    azar = random.SystemRandom()

    if modulo in MODULOS[:4]:
        c = np.array([
            azar.randint(2, 9),
            azar.randint(2, 9)
        ], dtype=float)

        A = np.array([[2., 1.], [1., 2.]])

        b = np.array([
            3. * azar.randint(10, 25),
            3. * azar.randint(10, 25)
        ])

        signos = ["<=", "<="]
        x, z = resolver_pl(c, A, b, signos)

        if modulo == "Modelación":
            texto = (
                f"Producto A deja {c[0]:g} y B deja {c[1]:g}. "
                "A consume 2 h de corte y 1 h de armado; "
                "B consume 1 h de corte y 2 h de armado. "
                f"Hay {b[0]:g} h de corte y {b[1]:g} h de armado. "
                "Se permite producción fraccionaria. "
                "Define x1=A y x2=B. "
                "Formula maximización y no negatividad. "
                "Ingresa, en orden: beneficio de A, beneficio de B, "
                "horas disponibles de corte, "
                "horas disponibles de armado."
            )

            esperado = np.r_[c, b]

        else:
            texto = (
                f"Max Z={c[0]:g}x1+{c[1]:g}x2; "
                f"2x1+x2<={b[0]:g}; "
                f"x1+2x2<={b[1]:g}; x1,x2>=0. "
                "Ingresa x1, x2 y Z óptimos, separados por comas."
            )

            esperado = np.r_[x, z]

        return dict(
            modulo=modulo,
            texto=texto,
            esperado=esperado,
            tipo="pl",
            datos=(c, A, b, signos, x, z)
        )

    if modulo == "Inventarios":
        datos = (
            float(azar.randrange(1000, 8000, 100)),
            float(azar.randint(20, 80)),
            float(azar.randint(2, 8)),
            300.,
            float(azar.randint(2, 10)),
            float(azar.randint(0, 40))
        )

        r = inventario(*datos)

        texto = (
            f"D={datos[0]:g} unidades/año; "
            f"S={datos[1]:g}/pedido; "
            f"H={datos[2]:g}/unidad/año; "
            f"300 días operativos; L={datos[4]:g} días; "
            f"SS={datos[5]:g}. "
            "Demanda constante sin descuentos ni faltantes. "
            "Ingresa EOQ, ROP."
        )

        return dict(
            modulo=modulo,
            texto=texto,
            esperado=np.array([r["q"], r["rop"]]),
            tipo="inv",
            datos=datos
        )

    if azar.choice([True, False]):
        costos = np.array([
            [azar.randint(1, 9) for _ in range(3)]
            for _ in range(3)
        ])

        filas, columnas, z = asignacion(costos)

        return dict(
            modulo=modulo,
            texto=(
                "Asignación: minimiza el costo; "
                "cada persona debe realizar una tarea distinta. "
                "Ingresa solo el costo mínimo."
            ),
            esperado=np.array([z]),
            tipo="asig",
            datos=(costos, filas, columnas),
            matriz=costos
        )

    costos = np.array([
        [azar.randint(1, 9) for _ in range(2)]
        for _ in range(2)
    ])

    oferta = np.array([
        float(azar.randint(10, 30)),
        float(azar.randint(10, 30))
    ])

    demanda = np.array([oferta.sum() / 2, oferta.sum() / 2])
    plan, z, *_ = transporte(costos, oferta, demanda)

    texto = (
        f"Transporte: oferta {oferta.tolist()}; "
        f"demanda {demanda.tolist()}. "
        "Minimiza el costo usando la matriz. "
        "Ingresa solo el costo mínimo."
    )

    return dict(
        modulo=modulo,
        texto=texto,
        esperado=np.array([z]),
        tipo="trans",
        datos=(costos, oferta, demanda, plan),
        matriz=costos
    )


def pantalla_practica():
    st.header("Prácticas con retroalimentación")
    modulo = st.selectbox("Tema de práctica", MODULOS)

    st.caption(
        "Los datos se generan al pulsar el botón "
        "y las respuestas se calculan. "
        "La evaluación comprueba los números solicitados; "
        "no califica explicaciones escritas."
    )

    if st.button("Generar una práctica nueva", type="primary"):
        st.session_state["practica"] = nueva_practica(modulo)
        st.session_state["respuesta_practica"] = ""
        st.session_state.pop("evaluacion", None)

    p = st.session_state.get("practica")

    if not p:
        return

    st.caption("Tema del ejercicio: " + p["modulo"])
    st.info(p["texto"])

    if "matriz" in p:
        st.dataframe(pd.DataFrame(
            p["matriz"],
            index=[
                f"Fila {i+1}"
                for i in range(len(p["matriz"]))
            ],
            columns=[
                f"Columna {j+1}"
                for j in range(p["matriz"].shape[1])
            ]
        ))

    with st.form("respuesta"):
        respuesta = st.text_input(
            "Respuesta numérica "
            "(usa punto decimal y comas entre valores)",
            key="respuesta_practica"
        )

        enviar = st.form_submit_button("Evaluar mi respuesta")

    if enviar:
        st.session_state.pop("evaluacion", None)

        try:
            valores = numeros(respuesta)
            esperado = p["esperado"]

            if len(valores) != len(esperado):
                raise ValueError(
                    f"Se esperan {len(esperado)} números, "
                    "en el orden indicado."
                )

            if p["tipo"] == "pl" and p["modulo"] != "Modelación":
                c, A, b, signos, _, z = p["datos"]

                correcto = (
                    cumple(valores[:2], A, b, signos)
                    and np.isclose(
                        c @ valores[:2], z,
                        atol=.01, rtol=1e-5
                    )
                    and np.isclose(
                        valores[2], z,
                        atol=.01, rtol=1e-5
                    )
                )

            else:
                correcto = bool(np.allclose(
                    valores, esperado,
                    atol=.01, rtol=1e-5
                ))

            st.session_state["evaluacion"] = (
                bool(correcto), valores
            )

        except ValueError as error:
            st.error(str(error))

    ev = st.session_state.get("evaluacion")

    if ev:
        correcto, valores = ev

        if correcto:
            st.success(
                "Respuesta correcta dentro de la tolerancia numérica."
            )
        else:
            st.warning(
                "Revisa tu resultado. Compara los datos "
                "y el procedimiento de abajo."
            )

        st.write(
            "Tu respuesta evaluada: "
            + ", ".join(f(v) for v in valores)
        )

        st.write(
            "Una respuesta esperada: "
            + ", ".join(f(v) for v in p["esperado"])
        )

        st.caption(
            "Tolerancia de comparación: 0.01 absoluta "
            "más 0.001% relativa. "
            "La factibilidad usa una tolerancia más estricta."
        )

        if p["tipo"] == "pl":
            c, A, b, signos, x, z = p["datos"]

            mostrar_modelo(
                c, A, b, signos,
                "Maximizar", "producto A, producto B"
            )

            if p["modulo"] == "Método simplex":
                mostrar_simplex(c, A, b, signos, "Maximizar")

            mostrar_solucion(c, A, b, signos, x, z)

            if p["modulo"] == "Método gráfico":
                grafica(c, A, b, signos, x, z, 0)

        elif p["tipo"] == "inv":
            mostrar_inventario(p["datos"])

        elif p["tipo"] == "asig":
            costos, filas, columnas = p["datos"]

            st.write(
                "Se elige una tarea distinta por persona "
                "y se suman sus costos:"
            )

            st.write(" + ".join(
                f"Persona {i+1} → tarea {j+1}: {f(costos[i, j])}"
                for i, j in zip(filas, columnas)
            ))

        else:
            costos, oferta, demanda, plan = p["datos"]

            st.write(
                "Se imponen sumas por fila iguales a oferta "
                "y por columna iguales a demanda. "
                "Un plan óptimo es:"
            )

            st.dataframe(pd.DataFrame(plan))

            st.write(
                "Costo = suma de cada envío × su costo = "
                + f(np.sum(costos * plan))
            )


def main():
    from html import escape

    st.set_page_config(
        page_title="Optiguía | Investigación de operaciones",
        page_icon="∑",
        layout="wide"
    )

    aplicar_diseno()

    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            '<div class="brand-symbol">∑</div>'
            '<div>'
            '<div class="brand-name">optiguía</div>'
            '<div class="brand-sub">Operaciones</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        pagina = st.radio(
            "Navegación",
            ["Resolver", "10 ejemplos", "Práctica", "Ayuda"],
            label_visibility="collapsed"
        )

        with st.expander("Acerca del proyecto", expanded=False):
            st.caption("Proyecto académico")
            st.write("**Universidad Mariano Gálvez de Guatemala**")
            st.write("Facultad de Ingeniería")
            st.write("**Asignatura:** Investigación de Operaciones")
            st.write("**Integrantes**")
            st.write("Yeferson David Hernández Tax")
            st.write("Katherine Iovanna Melendez Olivares")
         

        st.caption("Optiguía · Herramientas de decisión")

    titulos = {
        "Resolver": (
            "Espacio de trabajo",
            "Resuelve tu próximo problema.",
            "Define un modelo, introduce los datos "
            "y revisa el resultado paso a paso."
        ),
        "10 ejemplos": (
            "Biblioteca",
            "Aprende con un ejemplo.",
            "Diez casos para explorar modelos, "
            "comparar resultados y empezar a practicar."
        ),
        "Práctica": (
            "Tu aprendizaje",
            "Pon a prueba lo aprendido.",
            "Genera un ejercicio, calcula tu respuesta "
            "y comprueba el procedimiento."
        ),
        "Ayuda": (
            "Guía de uso",
            "Un buen modelo empieza con los datos.",
            "Consulta cómo ingresar un problema "
            "y qué puedes resolver en cada módulo."
        )
    }

    categoria, titulo, descripcion = titulos[pagina]

    st.markdown(
        f'<div class="eyebrow">{categoria}</div>',
        unsafe_allow_html=True
    )

    st.title(titulo)

    st.markdown(
        f'<p class="page-intro">{descripcion}</p>'
        '<div class="section-rule"></div>',
        unsafe_allow_html=True
    )

    if pagina == "Ayuda":
        st.header("Cómo usar el asistente")

        st.write(
            "1. Escoge un módulo. 2. Carga un ejemplo o ingresa "
            "datos nuevos. 3. Pulsa calcular. "
            "4. Revisa fórmulas, comprobación e interpretación."
        )

        st.write(
            "En modelación tú identificas los datos "
            "y describes las variables; el asistente construye "
            "su representación matemática. "
            "No convierte automáticamente cualquier texto en ecuaciones."
        )

        st.write(
            "Programación lineal: 1–10 variables, "
            "hasta 30 restricciones, <=, >= o = "
            "y variables continuas no negativas. "
            "Método gráfico: dos variables. "
            "Simplex explicado: restricciones <= "
            "y límites no negativos; no incluye fase I. "
            "Transporte: hasta 30×30 con balanceo ficticio opcional. "
            "Asignación: matriz cuadrada hasta 30×30. "
            "Inventarios: modelo EOQ básico "
            "y ROP con seguridad dada por el usuario."
        )

        st.write(
            "Los problemas fuera de estos alcances deben reformularse "
            "o resolverse con otra herramienta. "
            "Un resultado fraccionario de producción no se debe "
            "redondear sin comprobar nuevamente las restricciones."
        )

        return

    if pagina == "Práctica":
        pantalla_practica()
        return

    if pagina == "10 ejemplos":
        st.header("10 ejercicios de ejemplo")

        st.write(
            "Los ejemplos se pueden cargar y modificar en Resolver. "
            "Los resultados allí se calculan nuevamente."
        )

        for e in EJEMPLOS:
            with st.expander(e["titulo"] + " · " + e["modulo"]):
                st.write(e["enunciado"])
                st.info("Resultado de referencia: " + e["esperado"])

        return

    modulo = st.selectbox("Qué quieres resolver", MODULOS)

    candidatos = [
        e for e in EJEMPLOS
        if e["modulo"] == modulo
    ]

    seleccion = st.selectbox(
        "Ejemplo de partida · puedes modificar sus datos",
        [e["titulo"] for e in candidatos]
    )

    ejemplo = next(
        e for e in candidatos
        if e["titulo"] == seleccion
    )

    st.markdown(
        '<div class="problem-brief">'
        '<strong>El problema</strong>'
        + escape(ejemplo["enunciado"])
        + "</div>",
        unsafe_allow_html=True
    )

    clave = modulo + "_" + seleccion
    st.header(modulo)

    if modulo in MODULOS[:4]:
        pantalla_pl(modulo, ejemplo, clave)
    elif modulo == "Inventarios":
        pantalla_inventario(ejemplo, clave)
    else:
        pantalla_distribucion(ejemplo, clave)


if __name__ == "__main__":
    main()
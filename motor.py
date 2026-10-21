"""Cálculos independientes de la interfaz. No requiere claves ni servicios externos."""
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment


def numeros(texto):
    try:
        valores = np.array([float(x) for x in texto.replace(';', ',').split(',')], dtype=float)
    except (ValueError, AttributeError):
        raise ValueError('Escribe números separados por comas; usa punto para decimales.')
    if not valores.size or not np.isfinite(valores).all():
        raise ValueError('Todos los datos deben ser números finitos.')
    return valores


def matriz(texto):
    filas = [numeros(f) for f in texto.strip().splitlines() if f.strip()]
    if not filas or len({len(f) for f in filas}) != 1:
        raise ValueError('La matriz necesita filas con la misma cantidad de columnas.')
    resultado = np.array(filas)
    if max(resultado.shape) > 30:
        raise ValueError('Usa como máximo 30 filas y 30 columnas.')
    return resultado


def leer_modelo(objetivo, restricciones):
    c = numeros(objetivo)
    if not 1 <= len(c) <= 10:
        raise ValueError('Usa entre 1 y 10 variables.')
    A, b, signos = [], [], []
    for numero, fila in enumerate(restricciones.strip().splitlines(), 1):
        if not fila.strip():
            continue
        campos = [x.strip() for x in fila.replace('≤', '<=').replace('≥', '>=').split(',')]
        if len(campos) != len(c) + 2 or campos[-2] not in ('<=', '>=', '='):
            raise ValueError(f'Restricción {numero}: escribe coeficientes, signo y límite separados por comas.')
        A.append(numeros(','.join(campos[:-2])))
        b.append(numeros(campos[-1])[0])
        signos.append(campos[-2])
    if not A or len(A) > 30:
        raise ValueError('Ingresa entre 1 y 30 restricciones.')
    return c, np.array(A), np.array(b), signos


def resolver_pl(c, A, b, signos, sentido='Maximizar'):
    aub, bub, aeq, beq = [], [], [], []
    for fila, limite, signo in zip(A, b, signos):
        if signo == '=':
            aeq.append(fila); beq.append(limite)
        else:
            factor = -1 if signo == '>=' else 1
            aub.append(fila * factor); bub.append(limite * factor)
    res = linprog(-c if sentido == 'Maximizar' else c,
                  A_ub=np.array(aub) if aub else None, b_ub=bub or None,
                  A_eq=np.array(aeq) if aeq else None, b_eq=beq or None,
                  bounds=(0, None), method='highs')
    mensajes = {2: 'El modelo es infactible: no existe una solución que cumpla todas las restricciones.',
                3: 'El objetivo no está acotado: no existe un óptimo finito en la dirección solicitada.'}
    if not res.success:
        raise ValueError(mensajes.get(res.status, 'No se obtuvo una solución confiable. Revisa la escala de los datos.'))
    if not cumple(res.x, A, b, signos):
        raise ValueError('No se pudo verificar la solución con la tolerancia numérica. Revisa la escala de los datos.')
    return res.x, float(c @ res.x)


def cumple(x, A, b, signos):
    if np.any(x < -1e-7):
        return False
    for valor, limite, signo in zip(A @ x, b, signos):
        tol = 1e-6 * max(1, abs(valor), abs(limite))
        if signo == '<=' and valor > limite + tol:
            return False
        if signo == '>=' and valor < limite - tol:
            return False
        if signo == '=' and abs(valor-limite) > tol:
            return False
    return True


def simplex(c, A, b, signos, sentido='Maximizar'):
    """Simplex primal básico: base inicial de holguras. Regla de Bland."""
    if any(s != '<=' for s in signos) or np.any(b < 0):
        raise ValueError('El simplex explicado admite restricciones <= y límites no negativos. '
                         'Para >=, = o límites negativos usa Programación lineal; aquí no se implementa fase I.')
    m, n = A.shape
    tabla = np.zeros((m+1, n+m+1))
    tabla[:m, :n] = A
    tabla[:m, n:n+m] = np.eye(m)
    tabla[:m, -1] = b
    tabla[-1, :n] = -c if sentido == 'Maximizar' else c
    base = list(range(n, n+m))
    pasos = []
    for _ in range(250):
        candidatos = np.flatnonzero(tabla[-1, :-1] < -1e-9)
        if not len(candidatos):
            pasos.append(dict(tabla=tabla.copy(), base=base.copy(), pivote=None))
            x = np.zeros(n+m)
            x[base] = tabla[:m, -1]
            return x[:n], float(c @ x[:n]), pasos
        entra = int(candidatos[0])
        positivas = np.flatnonzero(tabla[:m, entra] > 1e-9)
        if not len(positivas):
            raise ValueError('Problema no acotado: no existe fila saliente para la columna entrante.')
        razones = tabla[positivas, -1] / tabla[positivas, entra]
        minimo = razones.min()
        empates = positivas[np.isclose(razones, minimo, atol=1e-9, rtol=1e-9)]
        sale = int(min(empates, key=lambda i: base[i]))
        pasos.append(dict(tabla=tabla.copy(), base=base.copy(), pivote=(sale, entra)))
        tabla[sale] /= tabla[sale, entra]
        for fila in range(m+1):
            if fila != sale:
                tabla[fila] -= tabla[fila, entra] * tabla[sale]
        base[sale] = entra
    raise ValueError('Se alcanzó el límite de iteraciones del simplex educativo.')


def transporte(costos, oferta, demanda, balancear=False):
    costos = np.array(costos, dtype=float)
    oferta = np.array(oferta, dtype=float)
    demanda = np.array(demanda, dtype=float)
    if costos.shape != (len(oferta), len(demanda)):
        raise ValueError('La matriz debe tener una fila por origen y una columna por destino.')
    if not all(np.isfinite(a).all() for a in (costos, oferta, demanda)):
        raise ValueError('Los datos deben ser finitos.')
    if np.any(oferta < 0) or np.any(demanda < 0) or oferta.sum() <= 0 or demanda.sum() <= 0:
        raise ValueError('Ofertas y demandas deben ser no negativas y tener totales positivos.')
    diferencia = oferta.sum() - demanda.sum()
    artificial = ''
    if abs(diferencia) > 1e-7:
        if not balancear:
            raise ValueError('Oferta y demanda no coinciden. Corrige los datos o autoriza un origen/destino ficticio.')
        if diferencia > 0:
            demanda = np.append(demanda, diferencia)
            costos = np.column_stack([costos, np.zeros(len(oferta))])
            artificial = 'Último destino ficticio: representa oferta sin utilizar, con costo cero supuesto.'
        else:
            oferta = np.append(oferta, -diferencia)
            costos = np.row_stack([costos, np.zeros(len(demanda))])
            artificial = 'Último origen ficticio: representa demanda no cubierta, con penalización cero supuesta; no es un envío real.'
    m, n = costos.shape
    A = []
    for i in range(m):
        f = np.zeros((m, n)); f[i, :] = 1; A.append(f.ravel())
    for j in range(n):
        f = np.zeros((m, n)); f[:, j] = 1; A.append(f.ravel())
    res = linprog(costos.ravel(), A_eq=A, b_eq=np.r_[oferta, demanda], bounds=(0, None), method='highs')
    if not res.success:
        raise ValueError('No se pudo resolver el transporte con los datos ingresados.')
    plan = res.x.reshape(m, n)
    if not np.allclose(plan.sum(1), oferta) or not np.allclose(plan.sum(0), demanda):
        raise ValueError('La solución no superó la comprobación de oferta y demanda.')
    return plan, float(np.sum(plan * costos)), costos, oferta, demanda, artificial


def asignacion(costos, maximizar=False):
    costos = np.array(costos, dtype=float)
    if costos.ndim != 2 or costos.shape[0] != costos.shape[1] or not np.isfinite(costos).all():
        raise ValueError('Este módulo básico requiere una matriz cuadrada y finita: igual número de personas y tareas.')
    filas, columnas = linear_sum_assignment(costos, maximize=maximizar)
    return filas, columnas, float(costos[filas, columnas].sum())


def inventario(D, S, H, dias, plazo, seguridad):
    datos = np.array([D, S, H, dias, plazo, seguridad], dtype=float)
    if not np.isfinite(datos).all() or min(D, S, H, dias) <= 0 or min(plazo, seguridad) < 0:
        raise ValueError('D, S, H y días deben ser positivos; plazo y seguridad no pueden ser negativos.')
    q = float(np.sqrt(2*D*S/H))
    return dict(q=q, pedidos=D/q, ciclo=dias*q/D, demanda_diaria=D/dias,
                rop=D/dias*plazo+seguridad, ordenar=D/q*S,
                mantener=q/2*H, costo=D/q*S+q/2*H)

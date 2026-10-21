"""Pruebas de los cálculos: ejecutar python verificar.py."""
import numpy as np
from motor import leer_modelo, resolver_pl, simplex, transporte, asignacion, inventario, matriz
from ejemplos import EJEMPLOS


def falla(accion):
    try:
        accion()
    except ValueError:
        return
    raise AssertionError('Se esperaba advertencia de entrada o caso no resoluble.')


def verificar():
    for e, esperado in zip(EJEMPLOS[:6], [2200,400,2200,22,10,23]):
        c,A,b,s = leer_modelo(e['c'], e['restricciones'])
        x,z = resolver_pl(c,A,b,s,e['sentido'])
        assert np.isclose(z,esperado), (e['titulo'],z)
    c,A,b,s = leer_modelo('40,30','2,1,<=,100\n1,2,<=,80')
    x,z = resolver_pl(c,A,b,s)
    assert np.allclose(x,[40,20]) and z == 2200
    assert np.isclose(simplex(c,A,b,s)[1],2200)
    # Variaciones nuevas: comparación independiente con HiGHS, max y min.
    rng = np.random.default_rng(42)
    for _ in range(40):
        A = np.vstack([rng.integers(1,10,size=(3,3)), np.eye(3)])
        b = rng.integers(5,40,size=6).astype(float)
        c = rng.integers(-5,10,size=3).astype(float)
        signos = ['<=']*6
        for sentido in ['Maximizar','Minimizar']:
            x,z = resolver_pl(c,A,b,signos,sentido)
            xs,zs,pasos = simplex(c,A,b,signos,sentido)
            assert np.isclose(z,zs) and np.all(A@xs <= b+1e-7)
            for paso in pasos:
                T = paso['tabla']
                assert np.allclose(T[:-1,paso['base']],np.eye(len(b)))
                assert np.all(T[:-1,-1]>=-1e-7)
    plan,z,*_ = transporte([[2,6],[5,1]],[20,30],[25,25])
    assert np.isclose(z,90) and np.allclose(plan,[[20,0],[5,25]])
    plan,z,C,O,D,nota = transporte([[2,6],[5,1]],[30,30],[25,25],True)
    assert nota and plan.shape == (2,3) and np.allclose(plan.sum(1),O)
    plan,z,C,O,D,nota = transporte([[2,6],[5,1]],[20,20],[25,25],True)
    assert nota and plan.shape == (3,2) and np.allclose(plan.sum(0),D)
    assert asignacion([[4,7],[6,3]])[2] == 7
    assert asignacion([[4,7],[6,3]],True)[2] == 13
    r = inventario(1200,50,3,300,5,0)
    assert np.isclose(r['q'],200) and r['rop']==20 and np.isclose(r['costo'],600)
    assert inventario(6000,50,3,300,5,30)['rop']==130
    # Inviable, no acotado y entradas incompletas.
    falla(lambda: resolver_pl(*leer_modelo('1','1,<=,1\n1,>=,2')))
    falla(lambda: resolver_pl(*leer_modelo('1','-1,<=,0')))
    falla(lambda: simplex(*leer_modelo('1','-1,<=,0')))
    falla(lambda: simplex(*leer_modelo('1','1,>=,2')))
    falla(lambda: leer_modelo('1,2','1,<=,4'))
    falla(lambda: leer_modelo('nan,2','1,1,<=,4'))
    falla(lambda: matriz('1,2\n3'))
    falla(lambda: asignacion([[1,2,3],[4,5,6]]))
    falla(lambda: transporte([[1]],[3],[2]))
    falla(lambda: inventario(1200,50,0,300,5,0))
    print('CORRECTO: 10 ejemplos, 80 comparaciones simplex/HiGHS, tablas y casos de error.')


if __name__ == '__main__':
    verificar()

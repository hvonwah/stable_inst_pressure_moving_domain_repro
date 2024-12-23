"""
    __author__ = H. v. Wahl
"""
# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from CutFEMMovingDomainTH import *
from other_methods.CutFEMMovingDomainSV import *

from math import pi, log2
import argparse
import pickle
from xfem import ngsxfemglobals

ngsglobals.msg_level = 0
SetHeapSize(10000000)
SetNumThreads(4)

ngsxfemglobals.simd_eval = False

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='TH', choices=['TH', 'SV'])
parser.add_argument('--bdf', type=int, default='1', choices=[1, 2],)
parser.add_argument('-Lx', '--mesh_levels', type=int, default=1)
parser.add_argument('-Lt', '--time_levels', type=int, default=1)
parser.add_argument('-k', '--order', type=int, default=2)
parser.add_argument('-nu', '--viscosity', type=float, default=0.01)
parser.add_argument('-gp_v', '--ghostpenalty_vel', type=float, default=1.0)
parser.add_argument('-gp_p', '--ghostpenalty_pre', type=float, default=1.0)
parser.add_argument('-stab', '--stabil', type=str, default='sufficient', choices=['sufficient', 'global'])
parser.add_argument('-ho', '--isoparametric', type=int, default=0, choices=[0, 1])
parser.add_argument('-diag', '--diagonal', type=int, default=0)
options = vars(parser.parse_args())
print(options)


# -------------------------------- PARAMETERS ---------------------------------
method = options['method']                      # Spatial discretisation
bdf = options['bdf']                            # BDF time-stepping scheme
k = options['order']                            # Order of velocity space
Lx = options['mesh_levels']                     # Mesh refinements
Lt = options['time_levels']                     # Time step refinements
h0 = 0.2                                        # Mesh diameter
diagonal = options['diagonal']                  # compute lx = lt + diagonal
dt0 = 0.1                                       # Inverse Time step

nu = options['viscosity']                       # Viscosity
gamma_v = options['ghostpenalty_vel']           # Vel. ghost-penalty parameter
gamma_p = options['ghostpenalty_pre']           # Pre. ghost-penalty parameter
gamma_n = 40                                    # Nitsche parameter
ho_geom = bool(options['isoparametric'])        # Use isoparametric mapping
stabil = options['stabil']             # Ghost-penalty facet choice 

filename = f'results/Convergence{method}{k}ho{int(ho_geom)}h{h0}bdf{bdf}'
filename += f'dt{dt0}nu{nu}gpv{gamma_v}gpp{gamma_p}facets{stabil}.pickle'


# ----------------------------------- DATA ------------------------------------
tend = 1.0                                     # End time
t = Parameter(0.0)                             # Time variable

r2_t = (x - t)**2 + y**2
uex = CF((2 * pi * y * cos(pi * r2_t), -2 * pi * (x - t) * cos(pi * r2_t)))
pex = sin(pi * r2_t) - 1 / (pi * 1 / 2)
grad_uex = CF((uex.Diff(x), uex.Diff(y)), dims=(2, 2)).trans

f = uex.Diff(t) - nu * uex.Diff(x).Diff(x) - nu * uex.Diff(y).Diff(y)
f += grad_uex * uex
f += CF((pex.Diff(x), pex.Diff(y)))

ubnd = CF((0, 0))
velmax = 1
levelset = sqrt(r2_t) - sqrt(1 / 2)


# ----------------------------- BACKGROUND DOMAIN -----------------------------
domain = SplineGeometry()
domain.AddRectangle((-1, -1), (2, 1), bcs=['bottom', 'right', 'top', 'left'])


# ----------------------------- CONVERGENCE STUDY -----------------------------
try:
    errors_raw, stabil_raw = pickle.load(open(filename, 'rb'))
    print('loaded the following error levels:\n', errors_raw.keys())
except OSError:
    errors_raw, stabil_raw = {}, {}


if method == 'TH':
    if bdf == 1:
        solver = cutfem_navier_stokes_moving_BDF1TH
    elif bdf == 2:
        solver = cutfem_navier_stokes_moving_BDF2TH
elif method == 'SV':
    if bdf == 1:
        solver = cutfem_navier_stokes_moving_BDF1SV
        print('hallo')
    elif bdf == 2:
        solver = cutfem_navier_stokes_moving_BDF2SV


for lx in range(Lx):
    print(f'Mesh level {lx + 1} / {Lx}')
    hmax = h0 / 2**lx
    with TaskManager():
        ngmesh = domain.GenerateMesh(maxh=h0, quad_dominated=False)
        for i in range(lx):
            ngmesh.Refine()
        mesh = Mesh(ngmesh)

    for lt in range(max(0, lx - diagonal), Lt):
        print(f'Time level {lt + 1} / {Lt}')
        if (lx, lt) in errors_raw.keys():
            continue
        else:
            print(f'computing lx={lx}, lt={lt}')

        dt = dt0 / 2**lt
        with TaskManager():
            errs, stab, gfu = solver(
                mesh, hmax, k, levelset, t, dt, tend, nu, velmax, f, ubnd,
                uex, pex, gamma_v, gamma_p, gamma_n, ho_geom, stabil)
        errors_raw[(lx, lt)] = errs
        stabil_raw[(lx, lt)] = stab
        pickle.dump([errors_raw, stabil_raw], open(filename, 'wb'))


# --------------------------- POST-PROCESS RESULTS ----------------------------
for data, type in [(errors_raw, 'error'), (stabil_raw, 'stability')]:
    vals = {(lx, lt): {} for lx in range(Lx) for lt in range(Lt)}
    for err in data[(0, 0)].keys():
        for lx in range(Lx):
            for lt in range(Lt):
                dt = dt0 / 2**lt
                try:
                    e = sqrt(sum([dt * _e**2 for _e in data[(lx, lt)][err]]))
                    vals[(lx, lt)][f'L2({err})'] = e
                    vals[(lx, lt)][f'Linf({err})'] = max(data[(lx, lt)][err])
                except:
                    KeyError

    for norm in ['L2', 'Linf']:
        for err in data[(0, 0)].keys():
            print(f'{norm}({err}) - {type}')
            err = f'{norm}({err})'
            print('Lt \\ Lx ', end='')
            [print(f'{i}        ', end='') for i in range(Lx)]
            print('eoc_t')
            for lt in range(Lt):
                print(f'{lt}       ', end='')
                for lx in range(Lx):
                    try:
                        print(f'{vals[(lx, lt)][err]:4.2e} ', end='')
                    except KeyError:
                        pass
                if lt == 0 and diagonal == Lx:
                    print('----')
                else:
                    try:
                        rate = log2(vals[(Lx - 1, lt - 1)][err] / vals[(Lx - 1, lt)][err])
                        print(f'{rate:4.2f}')
                    except KeyError:
                        print('')

            print('eoc_x   ----     ', end='')
            for lx in range(1, Lx):
                rate = log2(vals[(lx - 1, Lt - 1)][err] / vals[(lx, Lt - 1)][err])
                print(f'{rate:4.2f}     ', end='')
            print('\neoc_xt  ----     ', end='')
            for lx in range(1, min(Lx, Lt)):
                rate = log2(vals[(lx - 1, lx - 1)][err] / vals[(lx, lx)][err])
                print(f'{rate:4.2f}     ', end='')
            print('\n')

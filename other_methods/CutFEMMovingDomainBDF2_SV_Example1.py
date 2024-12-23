"""
    __author__ = H. v. Wahl
"""
# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve import dx as dx_ngs
from xfem import *
import sys
sys.path.append("..")
from newton import quasi_newton_solve
from barycentric import *
from math import pi, ceil
import time
import argparse
import os

ngsglobals.msg_level = 0
SetHeapSize(10000000)
SetNumThreads(4)
ngsxfemglobals.simd_eval = False

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--order', type=int, default=2)
parser.add_argument('-hmax', '--mesh_size', type=float, default=0.2)
parser.add_argument('-dt', '--time_step', type=float, default=0.14)
parser.add_argument('-mu', '--viscosity', type=float, default=1.0)
parser.add_argument('-gp_v', '--ghostpenalty_vel', type=float, default=0.1)
parser.add_argument('-gp_p', '--ghostpenalty_pre', type=float, default=1.0)
parser.add_argument('-vtk', '--vtk', type=int, default=0, choices=[0, 1])
options = vars(parser.parse_args())
print(options)

start_time = time.time()


# -------------------------------- PARAMETERS ---------------------------------
lowerleft, upperright = (-10, -5), (10, 5)      # Corners of background domain
h_max = options['mesh_size']                    # Mesh diameter
k = options['order']                            # Order of velocity space

t_end = 10.0                                    # End time
dt = options['time_step']                       # Inverse Time step
t = Parameter(0.0)                              # Time variable

rho = 1                                         # Fluid density
nu = options['viscosity'] / rho                 # Viscosity
gamma_v = options['ghostpenalty_vel']           # Vel. ghost-penalty parameter
gamma_p = options['ghostpenalty_pre']           # Pre. ghost-penalty parameter
gamma_n = 40                                    # Nitsche parameter

pReg = 1e-8                                     # Pressure block regularization
inverse = "pardiso"                             # Sparse direct linear solver

vtk_out = bool(options['vtk'])                  # Write VTK files of solution
vtk_dir = 'vtk'                                 # VTK directory
vtk_frequency = 1                               # VTK every nth time step

output_file = f'Example1_SV{k}hmax{h_max}BDF2dt{dt}nu{nu}gpv{gamma_v}gpp{gamma_p}'


# ----------------------------------- DATA ------------------------------------
uin = CF((1, 0)).Compile()

D = 2.0
f0 = 0.2237
X0 = 0.25 * D
U0 = 2 * pi * f0 * X0
velmax = U0


def d2(t):
    """Displacement of the cylinder in the x-direction"""
    return X0 * cos(2 * pi * f0 * t)


def levelset_func(t):
    """Level set function of the cylinder"""
    return D / 2 - sqrt((x + 5)**2 + (y - d2(t))**2)


def v2(t):
    """Velocity of the cylinder in the x-direction"""
    return - U0 * sin(2 * pi * f0 * t)


def circ_speed(t):
    """Velocity of the cylinder"""
    return CoefficientFunction((0.0, v2(t)))


# ------------------------------ BACKGROUND MESH ------------------------------
background_domain = SplineGeometry()
background_domain.AddRectangle(lowerleft, upperright,
                               bcs=['bottom', 'right', 'top', 'left'])
mesh_noref = Mesh(background_domain.GenerateMesh(maxh=h_max, quad_dominated=False))

ngmesh, mapping = alfeld_split_2d(mesh_noref)
mesh = Mesh(ngmesh)


# --------------------------- FINITE ELEMENT SPACE ----------------------------
V = VectorH1(mesh, order=k, dirichletx='left', dirichlety='left|top|bottom')
Q = L2(mesh, order=k - 1)
X = FESpace([V, Q], dgjumps=True)

active_dofs = BitArray(X.ndof)

gfu, gfu_last, gfu_last2 = GridFunction(X), GridFunction(X), GridFunction(X)
vel, pre = gfu.components
vel_last = gfu_last.components[0]
vel_last2 = gfu_last2.components[0]
gfu.vec[:] = 0.0


# ---------------------------- LEVELSET & CUT-INFO ----------------------------
lsetmeshadap = NoDeformation(mesh, levelset_func(0.0), warn=False)
lsetp1 = lsetmeshadap.lset_p1

lsetp1_ring = GridFunction(H1(mesh_noref, order=1))
InterpolateToP1(levelset_func(0.0), lsetp1_ring)

ci_main, ci_ring = CutInfo(mesh, lsetp1), CutInfo(mesh_noref, lsetp1_ring)


# ------------------------------ ELEMENT MARKERS ------------------------------
els_hasneg, els_if = BitArray(mesh.ne), BitArray(mesh.ne)
els_outer = BitArray(mesh.ne)
facets_gp, els_gp = BitArray(mesh.nedge), BitArray(mesh.ne)

els_outer_noref = BitArray(mesh_noref.ne)
els_inner_noref = BitArray(mesh_noref.ne)
els_ring_noref = BitArray(mesh_noref.ne)

els_outer_old, els_test = BitArray(mesh.ne), BitArray(mesh.ne)
els_outer_old2 = BitArray(mesh.ne)
facets_none = BitArray(mesh.nedge)
facets_none[:] = False
els_hasneg_last, els_hasneg_last2 = BitArray(mesh.ne), BitArray(mesh.ne)


# -------------------------------- INTEGRATORS --------------------------------
dx_act = dx_ngs(definedonelements=els_outer)
dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg)
ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if)
dw = dFacetPatch(definedonelements=facets_gp)


# ----------------------------- (BI)LINEAR FORMS ------------------------------
(u, p), (v, q) = X.TnT()


h = specialcf.mesh_size                         # Mesh size coefficient func.
n_levelset = 1.0 / Norm(grad(lsetp1)) * grad(lsetp1)  # Level set normal vector
delta = 2 * dt * velmax                         # Ghost strip-width
K_tilde = int(ceil(delta / h_max))              # Strip width in elements

bdf1_steps = ceil(dt / (dt**(4 / 3)))           # Nr. BDF1 time steps
dt_bdf1 = dt / bdf1_steps                       # BDF1 time step

mass = u * v
mass_1, mass_2 = InnerProduct(vel_last, v), InnerProduct(vel_last2, v)
stokes = nu * InnerProduct(Grad(u), Grad(v)) - p * div(v) - q * div(u)
stab_pre = - pReg * p * q

convect = InnerProduct(Grad(u) * u, v)

nitsche = -nu * InnerProduct(Grad(u) * n_levelset, v)
nitsche += -nu * InnerProduct(Grad(v) * n_levelset, u)
nitsche += nu * (gamma_n * k**2 / h) * InnerProduct(u, v)
nitsche += p * InnerProduct(v, n_levelset)

ghost_penalty = gamma_v * 3 * K_tilde * (nu + 1 / nu) * (1 / h**2) * (u - u.Other()) * (v - v.Other())
ghost_penalty += -gamma_p * 3 * K_tilde * (div(u) - div(u.Other())) * (q - q.Other())
ghost_penalty += -gamma_p * 3 * K_tilde * (div(v) - div(v.Other())) * (p - p.Other())

nitsche_rhs = -nu * InnerProduct(Grad(v) * n_levelset, circ_speed(t))
nitsche_rhs += nu * (gamma_n * k**2 / h) * InnerProduct(circ_speed(t), v)


# -------------------------------- INTEGRATORS --------------------------------
comp_opts = {'realcompile': False, 'wait': False}

a = RestrictedBilinearForm(X, element_restriction=els_outer,
                           facet_restriction=facets_gp, check_unused=False)
a += stokes.Compile(**comp_opts) * dx
a += nitsche.Compile(**comp_opts) * ds
a += ghost_penalty.Compile(**comp_opts) * dw
a += stab_pre.Compile(**comp_opts) * dx_act


a1 = RestrictedBilinearForm(X, element_restriction=els_outer,
                            facet_restriction=facets_gp, check_unused=False)
a1 += (mass + dt_bdf1 * (stokes + convect)).Compile(**comp_opts) * dx
a1 += (dt_bdf1 * nitsche).Compile(**comp_opts) * ds
a1 += (dt_bdf1 * ghost_penalty).Compile(**comp_opts) * dw
a1 += (dt_bdf1 * stab_pre).Compile(**comp_opts) * dx_act

f1 = LinearForm(X)
f1 += mass_1.Compile(**comp_opts) * dx
f1 += (dt_bdf1 * nitsche_rhs).Compile(**comp_opts) * ds

a2 = RestrictedBilinearForm(X, element_restriction=els_outer,
                            facet_restriction=facets_gp, check_unused=False)
a2 += (3 / 2 * mass + dt * (stokes + convect)).Compile(**comp_opts) * dx
a2 += (dt * nitsche).Compile(**comp_opts) * ds
a2 += (dt * ghost_penalty).Compile(**comp_opts) * dw
a2 += (dt * stab_pre).Compile(**comp_opts) * dx_act

f2 = LinearForm(X)
f2 += (2 * mass_1 - 1 / 2 * mass_2).Compile(**comp_opts) * dx
f2 += (dt * nitsche_rhs).Compile(**comp_opts) * ds


# -------------------------------- FUNCTIONALS --------------------------------
stress = InnerProduct(-nu * Grad(u) * n_levelset + p * n_levelset, v)
stress += nu * (gamma_n * k**2 / h) * InnerProduct(u - circ_speed(t), v)

a_stress = RestrictedBilinearForm(X, element_restriction=els_if,
                                  facet_restriction=facets_none,
                                  check_unused=False)
a_stress += stress.Compile(**comp_opts) * ds

a_pre = RestrictedBilinearForm(X, element_restriction=els_if,
                               facet_restriction=facets_none,
                               check_unused=False)
a_pre += InnerProduct(p * n_levelset, v).Compile(**comp_opts) * ds

drag_x_test, drag_y_test = GridFunction(X), GridFunction(X)
drag_x_test.components[0].Set(CoefficientFunction((1.0, 0.0)))
drag_y_test.components[0].Set(CoefficientFunction((0.0, 1.0)))
res = gfu.vec.CreateVector()
div2_cf = (div(vel)**2).Compile(**comp_opts)
area = []

with open(f'{output_file}.txt', 'w') as fid:
    fid.write('time\tdrag\tlift\tdragP1\tdragP2\telsnew\telsnewext\tdiv\tmeanP\tdiffA\n')


def CompOutput():
    # Drag and lift
    a_stress.Apply(gfu.vec, res)

    drag_x = 2 / (U0**2 * rho * D) * InnerProduct(res, drag_x_test.vec)
    drag_y = 2 / (U0**2 * rho * D) * InnerProduct(res, drag_y_test.vec)

    a_pre.Apply(gfu.vec, res)
    drag_p1 = 2 / (rho * D**3 * f0**2) * InnerProduct(res, drag_x_test.vec)
    drag_p2 = 2 / (rho * D**3 * f0**2) * InnerProduct(res, drag_y_test.vec)

    if t.Get() < 1.1 * dt:
        nr_els_new = sum(els_hasneg & ~els_hasneg_last)
        nr_els_new_ext = sum(els_outer & ~els_outer_old)
    else:
        nr_els_new = sum(els_hasneg & ~els_hasneg_last & ~els_hasneg_last2)
        nr_els_new_ext = sum(els_outer & ~els_outer_old & ~els_outer_old2)
    div_err = sqrt(Integrate(div2_cf * dx_act, mesh))
    pre_mean = Integrate(pre * dx, mesh)
    area.append(Integrate(CF(1) * dx, mesh))
    if len(area) > 1:
        diff_area = area[-1] - area[-2]
    else:
        diff_area = 0.0

    out = f'{t.Get():10.8f}\t{drag_x:10.8f}\t{drag_y:10.8f}\t{drag_p1:10.8f}'
    out += f'\t{drag_p2:10.8f}\t{nr_els_new}\t{nr_els_new_ext}\t{div_err:5.3e}'
    out += f'\t{pre_mean:.5e}\t{diff_area:.5e}\n'
    with open(f'{output_file}.txt', "a") as fid:
        fid.write(out)


# ------------------------------- VISUALISATION -------------------------------
try:
    data_dir = os.environ['DATA']
except KeyError:
    print('DATA environment variable does not exist')
    data_dir = '..'
comp_dir_name = os.getcwd().split('/')[-1]

if vtk_out:
    vtk_dir_abs = f'{data_dir}/{comp_dir_name}/{vtk_dir}'
    if not os.path.isdir(vtk_dir_abs):
        os.makedirs(vtk_dir_abs)

    coefs = [vel.Compile(), div(vel).Compile(), pre.Compile(), lsetp1.Compile()]
    names = ['vel', 'div', 'pre', 'lsetp1']

    vtk = VTKOutput(ma=mesh, coefs=coefs, names=names,
                    filename=f'{vtk_dir_abs}/{output_file}', subdivision=1)
    vtk_mesh = VTKOutput(ma=mesh, coefs=[lsetp1], names=['lset'],
                         filename=f'{vtk_dir_abs}/{output_file}_Mesh',
                         subdivision=0)
    vtk_mesh.Do()

Draw(BitArrayCF(els_outer) * vel, mesh, 'vel')
Draw(BitArrayCF(els_outer) * div(vel), mesh, 'div')
Draw(BitArrayCF(els_outer) * pre, mesh, 'pre')


# ------------------------------- MARKER UPDATES ------------------------------
def update_lset_and_markers(t):
    # Update physical domain, mesh adaptation updates gfu_last1,2
    lsetmeshadap.CalcDeformation(levelset_func(t))
    ci_main.Update(lsetp1)
    els_hasneg[:] = ci_main.GetElementsOfType(HASNEG)
    els_if[:] = ci_main.GetElementsOfType(IF)

    # Update strip element markers
    InterpolateToP1(levelset_func(t) - delta, lsetp1_ring)
    ci_ring.Update(lsetp1_ring)
    els_outer_noref[:] = ci_ring.GetElementsOfType(HASNEG)
    InterpolateToP1(levelset_func(t) + delta, lsetp1_ring)
    ci_ring.Update(lsetp1_ring)
    els_inner_noref[:] = ci_ring.GetElementsOfType(NEG)
    els_ring_noref[:] = els_outer_noref & ~els_inner_noref
    facets_gp_noref = GetFacetsWithNeighborTypes(
        mesh_noref, a=els_outer_noref, b=els_ring_noref, use_and=True, bnd_val_b=False)
    els_gp_noref = GetElementsWithNeighborFacets(mesh_noref, facets_gp_noref)

    els_gp.Clear(), els_outer.Clear()
    for i in range(mesh_noref.ne):
        if els_gp_noref[i]:
            els_gp[3 * i: 3 * (i + 1)] = True
        if els_outer_noref[i]:
            els_outer[3 * i: 3 * (i + 1)] = True
    facets_gp[:] = GetFacetsWithNeighborTypes(mesh, a=els_gp, b=els_gp,
                                              use_and=True)

    # Update dofs
    active_dofs[:] = GetDofsOfElements(X, els_outer) & X.FreeDofs()
    return None


# ----------------------------- INITIAL CONDITION -----------------------------
with TaskManager():
    t.Set(0)

    # Update level set and marker BitArrays
    update_lset_and_markers(t)

    # Solve linear problem
    res = gfu.vec.CreateVector()
    res.data[:] = 0.0
    a.Assemble()
    inv = a.mat.Inverse(freedofs=active_dofs, inverse=inverse)

    gfu.components[0].Set(uin, definedon=mesh.Boundaries('left'))
    res.data -= a.mat * gfu.vec
    gfu.vec.data += inv * res
    if vtk_out:
        vtk.Do(time=t.Get(), drawelems=els_outer)
    Redraw()


# ------------------------------- TIME STEPPING -------------------------------
with TaskManager():
    # Copy data for first BDF2 time-step
    gfu_last2.vec.data = gfu.vec
    els_outer_old2[:] = els_outer
    els_hasneg_last2[:] = els_hasneg

    # Copy last velocity for time-stepping
    gfu_last.vec.data = gfu.vec
    els_hasneg_last[:] = els_hasneg
    els_outer_old[:] = els_outer

    # First BDF1 time-steps
    for it in range(1, bdf1_steps + 1):
        gfu_last.vec.data = gfu.vec
        els_outer_old[:] = els_outer

        t.Set(it * dt_bdf1)

        # Update level set and marker BitArrays
        update_lset_and_markers(t)

        # Check whether all active elements have the necessary history
        els_test[:] = ~els_outer_old & els_hasneg
        assert sum(els_test) == 0, 'Some elements have no history'
        els_outer_old[:] = els_outer

        # Solve non-linear problem
        f1.Assemble()
        quasi_newton_solve(a1, gfu, f=f1.vec, freedofs=active_dofs, inverse=inverse)
        CompOutput()
        Redraw()
        print(f't={t.Get():010.6f}, active_els={sum(els_outer)}, K={K_tilde} - 1')

    if vtk_out and (1 % vtk_frequency == 0):
        vtk.Do(time=t.Get(), drawelems=els_outer)

    # Copy initial condition back
    gfu_last.vec.data = gfu_last2.vec
    els_outer_old[:] = els_outer_old2
    els_hasneg_last[:] = els_hasneg_last2

    # BDF2 time-stepping
    for it in range(2, int(t_end / dt + 1)):
        # Copy last velocities for time-stepping
        gfu_last2.vec.data = gfu_last.vec
        gfu_last.vec.data = gfu.vec
        els_hasneg_last2, els_outer_old2[:] = els_hasneg_last, els_outer_old
        els_hasneg_last[:], els_outer_old[:] = els_hasneg, els_outer

        t.Set(it * dt)

        # Update level set and marker BitArrays
        update_lset_and_markers(t)

        # Check whether all active elements have the necessary history
        els_test[:] = ~els_outer_old2 & ~els_outer_old & els_hasneg
        assert sum(els_test) == 0, 'Some elements have no history'

        # Solve non-linear problem
        f2.Assemble()
        quasi_newton_solve(a2, gfu, f=f2.vec, freedofs=active_dofs, inverse=inverse)
        CompOutput()

        if vtk_out and (it % vtk_frequency == 0):
            vtk.Do(time=t.Get(), drawelems=els_outer)
        Redraw()
        print(f't={t.Get():010.6f}, active_els={sum(els_outer)}, K={K_tilde}')


# ------------------------------ POST-PROCESSING ------------------------------
end_time = time.time() - start_time

print('\n----------- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f}'
      ' ----------'.format(end_time // (24 * 60 * 60),
                           end_time % (24 * 60 * 60) // (60 * 60),
                           end_time % 3600 // 60,
                           end_time % 60))
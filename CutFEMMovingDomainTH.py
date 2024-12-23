"""
    __author__ = H. v. Wahl
"""
# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from ngsolve import dx as dx_ngs
from xfem import *
from xfem.lsetcurv import *
from newton import quasi_newton_solve
from math import ceil

__all__ = ['cutfem_navier_stokes_moving_BDF2TH', 'cutfem_navier_stokes_moving_BDF1TH']

comp_opts = {'realcompile': False, 'wait': True}


def cutfem_navier_stokes_moving_BDF2TH(mesh, hmax, k, levelset, t, dt, tend, nu, velmax, forcing, ubnd, uex, pex, gamma_v, gamma_p, gamma_n, ho_geom=True, facet_choice='sufficient', inverse='pardiso'):
    t.Set(0.0)

    # ------------------------- FINITE ELEMENT SPACE --------------------------
    V = VectorH1(mesh, order=k)
    Q = H1(mesh, order=k - 1)
    X = FESpace([V, Q], dgjumps=True)

    active_dofs = BitArray(X.ndof)

    gfu, gfu_last, gfu_last2 = GridFunction(X), GridFunction(X), GridFunction(X)
    vel, pre = gfu.components
    vel_last = gfu_last.components[0]
    vel_last2 = gfu_last2.components[0]
    gfu.vec[:] = 0.0

    # -------------------------- LEVELSET & CUT-INFO --------------------------
    # Higher order discrete level set approximation
    if ho_geom is True:
        print('ho deformation')
        lsetmeshadap = LevelSetMeshAdaptation(mesh, order=k, threshold=0.1,
                                              discontinuous_qn=True,
                                              levelset=levelset)
        lsetmeshadap.ProjectOnUpdate(vel_last)
        lsetmeshadap.ProjectOnUpdate(vel_last2)
    else:
        lsetmeshadap = NoDeformation(mesh, levelset, warn=False)
    deform = lsetmeshadap.deform
    lsetp1 = lsetmeshadap.lset_p1

    lsetp1_ring = GridFunction(H1(mesh, order=1))
    InterpolateToP1(levelset, lsetp1_ring)

    ci_main, ci_ring = CutInfo(mesh, lsetp1), CutInfo(mesh, lsetp1_ring)

    # ---------------------------- ELEMENT MARKERS ----------------------------
    els_hasneg, els_if = BitArray(mesh.ne), BitArray(mesh.ne)
    els_outer, els_inner = BitArray(mesh.ne), BitArray(mesh.ne)
    els_ring = BitArray(mesh.ne)

    facets_gp_v, facets_gp_p = BitArray(mesh.nedge), BitArray(mesh.nedge)
    facets_gp = BitArray(mesh.nedge)

    els_outer_old, els_test = BitArray(mesh.ne), BitArray(mesh.ne)
    els_outer_old2 = BitArray(mesh.ne)

    # ------------------------------ INTEGRATORS ------------------------------
    dx_act = dx_ngs(definedonelements=els_outer, deformation=deform)
    dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg, deformation=deform)
    dx2k = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg, deformation=deform, order=2 * k)
    ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if, deformation=deform)
    dw_v = dFacetPatch(definedonelements=facets_gp_v, deformation=deform)
    dw_p = dFacetPatch(definedonelements=facets_gp_p, deformation=deform)

    # --------------------------- (BI)LINEAR FORMS ----------------------------
    (u, p), (v, q) = X.TnT()
    
    h = specialcf.mesh_size                               # Mesh size coefficient func.
    n_levelset = 1.0 / Norm(grad(lsetp1)) * grad(lsetp1)  # Level set normal vector
    delta = 2 * dt * velmax                               # Ghost strip-width
    K_tilde = int(ceil(delta / hmax))                     # Strip width in elements
    if facet_choice == 'global':
        K_tilde = 1

    bdf1_steps = ceil(dt / (dt**(4 / 3)))
    dt_bdf1 = dt / bdf1_steps

    mass = u * v
    mass_1, mass_2 = InnerProduct(vel_last, v), InnerProduct(vel_last2, v)
    stokes = nu * InnerProduct(Grad(u), Grad(v)) - p * div(v) - q * div(u)
    stokes += -1e-8 * p * q

    convect = InnerProduct(Grad(u) * u, v)

    nitsche = -nu * InnerProduct(Grad(u) * n_levelset, v)
    nitsche += -nu * InnerProduct(Grad(v) * n_levelset, u)
    nitsche += nu * (gamma_n * k**2 / h) * InnerProduct(u, v)
    nitsche += p * InnerProduct(v, n_levelset)
    nitsche += q * InnerProduct(u, n_levelset)

    ghost_penalty_v = gamma_v * K_tilde * (nu + 1 / nu) * (1 / h**2) * (u - u.Other()) * (v - v.Other())
    ghost_penalty_p = -gamma_p / nu * (p - p.Other()) * (q - q.Other())

    rhs = forcing * v

    nitsche_rhs = -nu * InnerProduct(Grad(v) * n_levelset, ubnd)
    nitsche_rhs += nu * (gamma_n * k**2 / h) * InnerProduct(ubnd, v)
    nitsche_rhs += q * InnerProduct(ubnd, n_levelset)

    # ------------------------------ INTEGRATORS ------------------------------
    a1 = RestrictedBilinearForm(X, element_restriction=els_outer,
                                facet_restriction=facets_gp, check_unused=False)
    a1 += (mass + dt_bdf1 * (stokes + convect)).Compile(**comp_opts) * dx
    a1 += (dt_bdf1 * nitsche).Compile(**comp_opts) * ds
    a1 += (dt_bdf1 * ghost_penalty_v).Compile(**comp_opts) * dw_v
    a1 += (dt_bdf1 * ghost_penalty_p).Compile(**comp_opts) * dw_p

    f1 = LinearForm(X)
    f1 += (mass_1 + dt_bdf1 * rhs).Compile(**comp_opts) * dx
    f1 += (dt_bdf1 * nitsche_rhs).Compile(**comp_opts) * ds

    a2 = RestrictedBilinearForm(X, element_restriction=els_outer,
                                facet_restriction=facets_gp, check_unused=False)
    a2 += (3 / 2 * mass + dt * (stokes + convect)).Compile(**comp_opts) * dx
    a2 += (dt * nitsche).Compile(**comp_opts) * ds
    a2 += (dt * ghost_penalty_v).Compile(**comp_opts) * dw_v
    a2 += (dt * ghost_penalty_p).Compile(**comp_opts) * dw_p

    f2 = LinearForm(X)
    f2 += (2 * mass_1 - 1 / 2 * mass_2 + dt * rhs).Compile(**comp_opts) * dx
    f2 += (dt * nitsche_rhs).Compile(**comp_opts) * ds

    # ----------------------------- MARKER UPDATES ----------------------------
    def update_lset_and_markers():
        # Update physical domain
        lsetmeshadap.CalcDeformation(levelset)
        ci_main.Update(lsetp1)
        els_hasneg[:] = ci_main.GetElementsOfType(HASNEG)
        els_if[:] = ci_main.GetElementsOfType(IF)
        # Update strip element markers
        InterpolateToP1(levelset - delta, lsetp1_ring)
        ci_ring.Update(lsetp1_ring)
        els_outer[:] = ci_ring.GetElementsOfType(HASNEG)
        InterpolateToP1(levelset + delta, lsetp1_ring)
        ci_ring.Update(lsetp1_ring)
        els_inner[:] = ci_ring.GetElementsOfType(NEG)
        els_ring[:] = els_outer & ~els_inner
        # Select facets appropriately
        if facet_choice == 'sufficient':
            facets_gp_v[:] = GetFacetsWithNeighborTypes(mesh, a=els_outer, b=els_ring, use_and=True)
            facets_gp_p[:] = GetFacetsWithNeighborTypes(mesh, a=els_hasneg, b=els_if, use_and=True)
            facets_gp[:] = facets_gp_v | facets_gp_p
            # Update dofs
            active_dofs[:] = CompoundBitArray([GetDofsOfElements(V, els_outer),
                                               GetDofsOfElements(Q, els_hasneg)])
        elif facet_choice == 'global':
            facets_gp_v.Set(), facets_gp_p.Set(), facets_gp.Set()
            active_dofs.Set()

        return None

    # --------------------------- ERROR COMPUTATION ---------------------------
    errs = {'l2v': [], 'h1v': [], 'l2p': [], 'h1p': [], 'l2div': [], 'l2dtv': []}

    uex_t = uex.Diff(t)
    grad_uex = CF((uex.Diff(x), uex.Diff(y)), dims=(2, 2)).trans
    grad_pex = CF((pex.Diff(x), pex.Diff(y)))

    err_l2v_cf = (vel - uex)**2
    err_h1v_cf = InnerProduct(Grad(vel) - grad_uex, Grad(vel) - grad_uex).Compile(**comp_opts)
    err_l2p_cf = ((pre - pex)**2).Compile(**comp_opts)
    err_h1p_cf = InnerProduct(Grad(pre) - grad_pex, Grad(pre) - grad_pex).Compile(**comp_opts)
    err_l2div_cf = (div(vel)**2).Compile(**comp_opts)
    err_l2dtv_1 = (((vel - vel_last) / dt - uex_t)**2).Compile(**comp_opts)
    err_l2dtv_2 = (((3 / 2 * vel - 2 * vel_last + 1 / 2 * vel_last2) / dt - uex_t)**2).Compile(**comp_opts)

    def comp_errors():
        errs['l2v'].append(sqrt(Integrate(err_l2v_cf * dx2k, mesh)))
        errs['h1v'].append(sqrt(Integrate(err_h1v_cf * dx2k, mesh)))
        errs['l2p'].append(sqrt(Integrate(err_l2p_cf * dx2k, mesh)))
        errs['h1p'].append(sqrt(Integrate(err_h1p_cf * dx2k, mesh)))
        errs['l2div'].append(sqrt(Integrate(err_l2div_cf * dx_act, mesh)))
        if t.Get() <= 1.1 * dt:
            errs['l2dtv'].append(sqrt(Integrate(err_l2dtv_1 * dx2k, mesh)))
        else:
            errs['l2dtv'].append(sqrt(Integrate(err_l2dtv_2 * dx2k, mesh)))
        return None

    stab = {'l2v': [], 'h1v': [], 'l2p': [], 'h1p': [], 'l2dtv': []}
    stab_l2v_cf = vel**2
    stab_h1v_cf = InnerProduct(grad_uex, grad_uex).Compile(**comp_opts)
    stab_l2p_cf = (pre**2).Compile(**comp_opts)
    stab_h1p_cf = (Grad(pre)**2).Compile(**comp_opts)
    stab_l2dtv_1 = (((vel - vel_last) / dt)**2).Compile(**comp_opts)
    stab_l2dtv_2 = (((3 / 2 * vel - 2 * vel_last + 1 / 2 * vel_last2) / dt)**2).Compile(**comp_opts)

    def comp_stabil():
        stab['l2v'].append(sqrt(Integrate(stab_l2v_cf * dx2k, mesh)))
        stab['h1v'].append(sqrt(Integrate(stab_h1v_cf * dx2k, mesh)))
        stab['l2p'].append(sqrt(Integrate(stab_l2p_cf * dx2k, mesh)))
        stab['h1p'].append(sqrt(Integrate(stab_h1p_cf * dx2k, mesh)))
        if t.Get() <= 1.1 * dt:
            stab['l2dtv'].append(sqrt(Integrate(stab_l2dtv_1 * dx2k, mesh)))
        else:
            stab['l2dtv'].append(sqrt(Integrate(stab_l2dtv_2 * dx2k, mesh)))
        return None

    # ----------------------------- TIME STEPPING -----------------------------
    # Initial condition
    t.Set(0)
    update_lset_and_markers()
    vel.Set(uex, definedonelements=els_outer)

    # Copy data for first BDF2 time-step
    gfu_last2.vec.data = gfu.vec
    els_outer_old2[:] = els_outer

    # First BDF1 time-steps
    for it in range(1, bdf1_steps + 1):
        gfu_last.vec.data = gfu.vec
        els_outer_old[:] = els_outer

        t.Set(it * dt_bdf1)

        # Update level set and marker BitArrays
        update_lset_and_markers()

        # Check whether all active elements have the necessary history
        els_test[:] = ~els_outer_old & els_hasneg
        assert sum(els_test) == 0, 'Some elements have no history'
        els_outer_old[:] = els_outer

        # Solve non-linear problem
        f1.Assemble()
        quasi_newton_solve(a1, gfu, f=f1.vec, freedofs=active_dofs,
                           inverse=inverse, printing=False, reiterate=True)

        _err_l2 = sqrt(Integrate(err_l2v_cf.Compile() * dx, mesh))
        print(f't={t.Get():010.6f}, err(l2v)={_err_l2:4.2e}, active_els={sum(els_outer)}, K={K_tilde} - 1')

    # Copy initial condition back
    gfu_last.vec.data = gfu_last2.vec
    els_outer_old[:] = els_outer_old2
    comp_errors()
    comp_stabil()

    # BDF2 time-stepping
    for it in range(2, int(tend / dt + 1)):
        # Copy last velocities for time-stepping
        gfu_last2.vec.data = gfu_last.vec
        gfu_last.vec.data = gfu.vec
        els_outer_old2[:] = els_outer_old
        els_outer_old[:] = els_outer

        t.Set(it * dt)

        # Update levelset and marker BitArrays
        update_lset_and_markers()

        # Check whether all active elements have the necessary history
        els_test[:] = ~els_outer_old2 | ~els_outer_old
        els_test &= els_hasneg
        assert sum(els_test) == 0, 'Some elements have no history'

        # Solve non-linear problem
        f2.Assemble()
        quasi_newton_solve(a2, gfu, f=f2.vec, freedofs=active_dofs,
                           inverse=inverse, printing=False, reiterate=True)
        comp_errors()
        comp_stabil()
        print(f't={t.Get():010.6f}, err(l2v)={errs['l2v'][-1]:4.2e}, active_els={sum(els_outer)}, K={K_tilde}')

    return errs, stab, gfu


def cutfem_navier_stokes_moving_BDF1TH(mesh, hmax, k, levelset, t, dt, tend, nu, velmax, forcing, ubnd, uex, pex, gamma_v, gamma_p, gamma_n, ho_geom=True, facet_choice='sufficient', inverse='pardiso'):
    t.Set(0.0)

    # ------------------------- FINITE ELEMENT SPACE --------------------------
    V = VectorH1(mesh, order=k)
    Q = H1(mesh, order=k - 1)
    X = FESpace([V, Q], dgjumps=True)

    active_dofs = BitArray(X.ndof)

    gfu, gfu_last = GridFunction(X), GridFunction(X)
    vel, pre = gfu.components
    vel_last = gfu_last.components[0]
    gfu.vec[:] = 0.0

    # -------------------------- LEVELSET & CUT-INFO --------------------------
    # Higher order discrete level set approximation
    if ho_geom is True:
        print('ho deformation')
        lsetmeshadap = LevelSetMeshAdaptation(mesh, order=k, threshold=0.1,
                                              discontinuous_qn=True,
                                              levelset=levelset)
        lsetmeshadap.ProjectOnUpdate(vel_last)
    else:
        lsetmeshadap = NoDeformation(mesh, levelset, warn=False)
    deform = lsetmeshadap.deform
    lsetp1 = lsetmeshadap.lset_p1

    lsetp1_ring = GridFunction(H1(mesh, order=1))
    InterpolateToP1(levelset, lsetp1_ring)

    ci_main, ci_ring = CutInfo(mesh, lsetp1), CutInfo(mesh, lsetp1_ring)

    # ---------------------------- ELEMENT MARKERS ----------------------------
    els_hasneg, els_if = BitArray(mesh.ne), BitArray(mesh.ne)
    els_outer, els_inner = BitArray(mesh.ne), BitArray(mesh.ne)
    els_ring = BitArray(mesh.ne)

    facets_gp_v, facets_gp_p = BitArray(mesh.nedge), BitArray(mesh.nedge)
    facets_gp = BitArray(mesh.nedge)

    els_outer_old, els_test = BitArray(mesh.ne), BitArray(mesh.ne)

    # ------------------------------ INTEGRATORS ------------------------------
    dx_act = dx_ngs(definedonelements=els_outer, deformation=deform)
    dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg, deformation=deform)
    dx2k = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg, deformation=deform, order=2 * k)
    ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if, deformation=deform)
    dw_v = dFacetPatch(definedonelements=facets_gp_v, deformation=deform)
    dw_p = dFacetPatch(definedonelements=facets_gp_p, deformation=deform)

    # --------------------------- (BI)LINEAR FORMS ----------------------------
    (u, p), (v, q) = X.TnT()

    h = specialcf.mesh_size                               # Mesh size coefficient func.
    n_levelset = 1.0 / Norm(grad(lsetp1)) * grad(lsetp1)  # Level set normal vector
    delta = dt * velmax                                   # Ghost strip-width
    K_tilde = int(ceil(delta / hmax))                     # Strip width in elements
    if facet_choice == 'global':
        K_tilde = 1

    mass = u * v
    mass_1 = InnerProduct(vel_last, v)
    stokes = nu * InnerProduct(Grad(u), Grad(v)) - p * div(v) - q * div(u)
    stokes += -1e-8 * p * q

    convect = InnerProduct(Grad(u) * u, v)

    nitsche = -nu * InnerProduct(Grad(u) * n_levelset, v)
    nitsche += -nu * InnerProduct(Grad(v) * n_levelset, u)
    nitsche += nu * (gamma_n * k**2 / h) * InnerProduct(u, v)
    nitsche += p * InnerProduct(v, n_levelset)
    nitsche += q * InnerProduct(u, n_levelset)

    ghost_penalty_v = gamma_v * K_tilde * (nu + 1 / nu) * (1 / h**2) * (u - u.Other()) * (v - v.Other())
    ghost_penalty_p = -gamma_p / nu * (p - p.Other()) * (q - q.Other())

    rhs = InnerProduct(forcing, v)

    nitsche_rhs = -nu * InnerProduct(Grad(v) * n_levelset, ubnd)
    nitsche_rhs += nu * (gamma_n * k**2 / h) * InnerProduct(ubnd, v)
    nitsche_rhs += q * InnerProduct(ubnd, n_levelset)

    # ------------------------------ INTEGRATORS ------------------------------
    a1 = RestrictedBilinearForm(X, element_restriction=els_outer,
                                facet_restriction=facets_gp, check_unused=False)
    a1 += (mass + dt * (stokes + convect)).Compile(**comp_opts) * dx
    a1 += (dt * nitsche).Compile(**comp_opts) * ds
    a1 += (dt * ghost_penalty_v).Compile(**comp_opts) * dw_v
    a1 += (dt * ghost_penalty_p).Compile(**comp_opts) * dw_p

    f1 = LinearForm(X)
    f1 += (mass_1 + dt * rhs).Compile(**comp_opts) * dx
    f1 += (dt * nitsche_rhs).Compile(**comp_opts) * ds

    # ----------------------------- MARKER UPDATES ----------------------------
    def update_lset_and_markers():
        # Update physical domain
        lsetmeshadap.CalcDeformation(levelset)
        ci_main.Update(lsetp1)
        els_hasneg[:] = ci_main.GetElementsOfType(HASNEG)
        els_if[:] = ci_main.GetElementsOfType(IF)
        # Update strip element markers
        InterpolateToP1(levelset - delta, lsetp1_ring)
        ci_ring.Update(lsetp1_ring)
        els_outer[:] = ci_ring.GetElementsOfType(HASNEG)
        InterpolateToP1(levelset + delta, lsetp1_ring)
        ci_ring.Update(lsetp1_ring)
        els_inner[:] = ci_ring.GetElementsOfType(NEG)
        els_ring[:] = els_outer & ~els_inner
        # Select facets appropriately
        if facet_choice == 'sufficient':
            facets_gp_v[:] = GetFacetsWithNeighborTypes(mesh, a=els_outer, b=els_ring, use_and=True)
            facets_gp_p[:] = GetFacetsWithNeighborTypes(mesh, a=els_hasneg, b=els_if, use_and=True)
            facets_gp[:] = facets_gp_v | facets_gp_p
            # Update dofs
            active_dofs[:] = CompoundBitArray([GetDofsOfElements(V, els_outer),
                                               GetDofsOfElements(Q, els_hasneg)])
        elif facet_choice == 'global':
            facets_gp_v.Set(), facets_gp_p.Set(), facets_gp.Set()
            active_dofs.Set()

        return None

    # --------------------------- ERROR COMPUTATION ---------------------------
    errs = {'l2v': [], 'h1v': [], 'l2p': [], 'h1p': [], 'l2div': [], 'l2dtv': []}

    uex_t = uex.Diff(t)
    grad_uex = CF((uex.Diff(x), uex.Diff(y)), dims=(2, 2)).trans
    grad_pex = CF((pex.Diff(x), pex.Diff(y)))

    err_l2v_cf = (vel - uex)**2
    err_h1v_cf = InnerProduct(Grad(vel) - grad_uex, Grad(vel) - grad_uex).Compile(**comp_opts)
    err_l2p_cf = ((pre - pex)**2).Compile(**comp_opts)
    err_h1p_cf = InnerProduct(Grad(pre) - grad_pex, Grad(pre) - grad_pex).Compile(**comp_opts)
    err_l2div_cf = (div(vel)**2).Compile(**comp_opts)
    err_l2dtv_1 = (((vel - vel_last) / dt - uex_t)**2).Compile(**comp_opts)

    def comp_errors():
        errs['l2v'].append(sqrt(Integrate(err_l2v_cf * dx2k, mesh)))
        errs['h1v'].append(sqrt(Integrate(err_h1v_cf * dx2k, mesh)))
        errs['l2p'].append(sqrt(Integrate(err_l2p_cf * dx2k, mesh)))
        errs['h1p'].append(sqrt(Integrate(err_h1p_cf * dx2k, mesh)))
        errs['l2div'].append(sqrt(Integrate(err_l2div_cf * dx_act, mesh)))
        errs['l2dtv'].append(sqrt(Integrate(err_l2dtv_1 * dx2k, mesh)))
        return None

    stab = {'l2v': [], 'h1v': [], 'l2p': [], 'h1p': [], 'l2dtv': []}
    stab_l2v_cf = vel**2
    stab_h1v_cf = InnerProduct(grad_uex, grad_uex).Compile(**comp_opts)
    stab_l2p_cf = (pre**2).Compile(**comp_opts)
    stab_h1p_cf = (Grad(pre)**2).Compile(**comp_opts)
    stab_l2dtv_1 = (((vel - vel_last) / dt)**2).Compile(**comp_opts)

    def comp_stabil():
        stab['l2v'].append(sqrt(Integrate(stab_l2v_cf * dx2k, mesh)))
        stab['h1v'].append(sqrt(Integrate(stab_h1v_cf * dx2k, mesh)))
        stab['l2p'].append(sqrt(Integrate(stab_l2p_cf * dx2k, mesh)))
        stab['h1p'].append(sqrt(Integrate(stab_h1p_cf * dx2k, mesh)))
        stab['l2dtv'].append(sqrt(Integrate(stab_l2dtv_1 * dx2k, mesh)))
        return None

    # ----------------------------- TIME STEPPING -----------------------------
    # Initial condition
    t.Set(0)
    update_lset_and_markers()
    vel.Set(uex, definedonelements=els_outer)

    # BDF1 time-steps
    for it in range(1, int(tend / dt + 1)):
        gfu_last.vec.data = gfu.vec
        els_outer_old[:] = els_outer

        t.Set(it * dt)

        # Update level set and marker BitArrays
        update_lset_and_markers()

        # Check whether all active elements have the necessary history
        els_test[:] = ~els_outer_old & els_hasneg
        assert sum(els_test) == 0, 'Some elements have no history'
        els_outer_old[:] = els_outer

        # Solve non-linear problem
        f1.Assemble()
        quasi_newton_solve(a1, gfu, f=f1.vec, freedofs=active_dofs,
                           inverse=inverse, printing=False, reiterate=True)
        comp_errors()
        comp_stabil()
        print(f't={t.Get():010.6f}, err(l2v)={errs['l2v'][-1]:4.2e}, active_els={sum(els_outer)}, K={K_tilde} - 1')

    return errs, stab, gfu

from ngsolve import Norm, Projector
from ngsolve.solvers import PreconditionedRichardson as PreRic


def quasi_newton_solve(a, u, alin=None, f=None, freedofs=None, maxit=100, maxerr=1e-11, inverse="umfpack", dampfactor=1, jacobi_update_tol=0.1, reuse=False, printing=True, reiterate=False, **kwargs):
    """
    Based on Newton() from NGSolve by J. Schöberl and NGSolve iTutorial 
    on non-linear problems by Christoph Lehrenfeld: 
    Uses the (quasi) Newton method to solve a non-linear system A(u)=f. 
    The Jacobian is updated, only if the residual did not decrease 
    sufficiently in the last iteration  

    Parameters
    ----------
    a : BilinearForm
      The BilinearForm of the non-linear variational problem. It does 
      not have to be assembled.
    u : GridFunction
      The GridFunction where the solution is saved. The values are used 
      as initial guess for Newton's method.
    f : Vector
      The right-hand side vector of the non-linear variational problem. 
    freedofs : BitArray
      The FreeDofs on which the assembled matrix is inverted. If 
      argument is 'None' then the FreeDofs of the underlying FESpace 
      is used.
    maxit : int
      Number of maximal iteration for Newton. If the maximal number is 
      reached before the maximal error Newton might no converge and a 
      warning is displayed.
    maxerr : float
      The maximal error which Newton should reach before it stops. The 
      error is computed by the square root of the inner product of the 
      residuum and the correction.
    inverse : string
      A string of the sparse direct solver which should be solved for 
      inverting the assembled Newton matrix.
    dampfactor : float
        Damping factor for the quasi Newton method. Damping is applied 
        in the form min(1,dampfactor*numit)
    jacobi_update_tol: float
      If the residual decreases by less than the factor 
      'jacobi_update_tol' compared to the last residual, then the 
      Jacobian is updated.
    reuse: bool
      If True, inv_jacobian is defined as a global variable and the 
      method checks in the Jacobian is available from a previous Newton 
      iteration.
    printing : bool
      Set if Newton's method should print informations about the 
      iterations like the error or if the Jacobian was updated in this 
      iteration.
    reiterate : bool
      Use direct solver as preconditioner for two preconditioned
      Richardson iterations.
    Returns
    -------
    (int, int)
      List of two integers. The first one is 0 if Newton's method did 
      converge, -1 otherwise. The second one gives the number of Newton 
      iterations needed.
    """

    # Test for unknown keyword arguments
    for key in kwargs:
        print("WARNING: Unknown keyword argument \"{}\"".format(key))

    res = u.vec.CreateVector()
    res_freedofs = u.vec.CreateVector()
    du = u.vec.CreateVector()
    u_old = u.vec.CreateVector()
    numit = 0
    err, errLast = float("NaN"), float("NaN")
    Updated = "n/a "

    if reuse and "inv_jacobian" not in globals():
        global inv_jacobian
        JacobianAvailable = False
    if reuse and "inv_jacobian" in globals():
        JacobianAvailable = True
    else:
        JacobianAvailable = False

    if freedofs is None:
        freedofs = u.space.FreeDofs(coupling=a.condense)
    projector = Projector(freedofs, True)

    if printing:
        print("\tNumit\tUpd.J\t||res||_l2")
        print("\t--------------------------")

    a.Apply(u.vec, res)
    if f:
        res.data -= f
    res_freedofs.data = projector * res
    err = Norm(res_freedofs)
    omega = 1

    for it in range(maxit):

        if printing:
            str_out = "    {:}\t\t{:}\t{:1.4e}".format(numit, Updated, err)
            if omega < 1:
                str_out += "  LS: {:}".format(omega)
            print(str_out)
        if err < maxerr:
            break
        elif not JacobianAvailable or err > errLast * jacobi_update_tol:
            UpdateJacobian = True
        else:
            UpdateJacobian = False

        numit += 1

        if UpdateJacobian:
            if alin is None:
                a.AssembleLinearization(u.vec, reallocate=True)
            else:
                alin.Assemble(reallocate=True)

            if inverse == "umfpack":
                new_factorisation = False
                try:
                    inv_jacobian.Update()
                except NameError:
                    new_factorisation = True
            if inverse != "umfpack" or new_factorisation:
                try:
                    del inv_jacobian
                except NameError:
                    pass
                finally:
                    if alin is None:
                        inv_jacobian = a.mat.Inverse(freedofs, inverse=inverse)
                    else:
                        inv_jacobian = alin.mat.Inverse(freedofs, inverse=inverse)
            Updated = True
            JacobianAvailable = True
        else:
            Updated = False

        if a.condense:
            res.data += a.harmonic_extension_trans * res
        if reiterate is False:
            du.data = inv_jacobian * res
        elif alin is None:
            du.data = PreRic(a=a, rhs=res, pre=inv_jacobian, freedofs=freedofs,
                             maxit=2, printing=False)
        else:
            du.data = PreRic(a=alin, rhs=res, pre=inv_jacobian,
                             freedofs=freedofs, maxit=2, printing=False)
        if a.condense:
            du.data += a.harmonic_extension * du
            du.data += a.inner_solve * res

        do_linesearch = True
        errLast = err
        u_old.data = u.vec
        omega = 1
        while do_linesearch and omega > 0.01:
            u.vec.data = u_old - omega * du

            a.Apply(u.vec, res)
            if f:
                res.data -= f
            res_freedofs.data = projector * res
            err = Norm(res_freedofs)
            if err < errLast:
                do_linesearch = False
            else:
                omega *= 0.5

    else:
        print(f"\tWarning: Newton might not have converged: ||res|| = {err:4.2e}")
        if not reuse:
            try:
                del inv_jacobian
            except NameError:
                pass
        return (-1, numit)
    if not reuse:
        try:
            del inv_jacobian
        except NameError:
            pass
    return (0, numit)


def del_jacobean():
    global inv_jacobian
    del inv_jacobian
    return None

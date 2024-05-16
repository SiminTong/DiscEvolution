# viscous_evolution.py
#
# Author: R. Booth
# Date: 16 - Nov - 2016
#
# Contains classes for solving the viscous evolution equations.
################################################################################
from __future__ import print_function
import numpy as np
import copy
from scipy.interpolate import interp1d

class ViscousEvolution(object):
    """Solves the 1D viscous evolution equation.

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.
    """

    def __init__(self, tol=0.5, boundary='Zero', in_bound='Mdot', boundary_dw='Zero', in_bound_dw = 'Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound
        self._bound_dw = boundary_dw
        self._in_bound_dw = in_bound_dw

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}



    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rc = grid.Rc
        self._Re = grid.Re
        self._X = 2 * np.sqrt(grid.Rc)
        self._dXe = 2 * np.diff(np.sqrt(grid.Re))
        self._dXc = 2 * np.diff(np.sqrt(grid.Rce))
        self._RXdXe = grid.Rc * self._X * self._dXe

    def _init_fluxes(self, disc):
        """Cache the important variables needed for the fluxes"""
        nuX = disc.nu * self._X

        S = np.zeros(len(nuX) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuX

        # Inner boundary
        if self._in_bound == 'Zero':            # Zero torque
            S[0] = 0
        elif self._in_bound == 'Mdot':          # Constant flux (appropriate for power law)
            #S[0] = S[1] * self._X[0] / self._X[1]
            S[0] = S[1] * 10 ** (np.log10(self._X[0]/self._X[1]) * np.log10(S[1]/S[2])/ np.log10(self._X[1]/self._X[2]))
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary
        if self._bound == 'Zero':               # Zero torque
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2] ** 2 / S[-3]
        elif self._bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S[-1] = S[-2] * self._X[-2] / self._X[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S[-1] = S[-2] * self._X[-1] / self._X[-2]
        else:
            raise ValueError("Error boundary type not recognised")

        self._dS = np.diff(S) / self._dXc

    def _init_fluxes_dw(self, disc):

        # term for wind advection term
        #S_dw = np.zeros(len(nuX) + 1, dtype='f8')
        #if disc.alpha_dw != 0: 
        x = self._Rc
        y = disc.Sigma_G * disc.nu_dw
        f = interp1d(x, y, fill_value='extrapolate')
        S_dw = f(self._Re)


        # Inner boundary (for wind advection term)
        if self._in_bound_dw == 'Zero':            # Zero torque
            S_dw[0] = 0
        elif self._in_bound_dw == 'Mdot':          # Constant flux (appropriate for power law)
            #S_dw[0] = S_dw[1] * self._X[0] / self._X[1]
            S_dw[0] = S_dw[1] * 10 ** (np.log10(self._X[0]/self._X[1]) * np.log10(S_dw[1]/S_dw[2])/ np.log10(self._X[1]/self._X[2]))
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary (for wind advection term)
        if self._bound_dw == 'Zero':               # Zero torque
            S_dw[-1] = 0
        elif self._bound_dw == 'power_law':
            S_dw[-1] = S_dw[-2] ** 2 / S_dw[-3]
        elif self._bound_dw == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S_dw[-1] = S_dw[-2] * self._X[-2] / self._X[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S_dw[-1] = S_dw[-2] * self._X[-1] / self._X[-2]
        else:
            raise ValueError("Error boundary type not recognised")
        #elif disc.alpha_dw == 0:
        #    S_dw = np.zeros_like(grid.Re)

        self._S_dw = S_dw

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return 3. * np.diff(self._dS) / self._RXdXe

    def _fluxes_dw(self):

        return 3. * np.diff(self._S_dw) / self._RXdXe

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return 3. * np.diff(ds) / self._RXdXe

    def viscous_velocity(self, disc, Sigma=None):
        """Compute the radial velocity due to viscosity and magnetised winds"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc) # for self._dS

        # Can accept other density specifications (e.g. gas only to match specification of dust drift by Dipierro+18)
        if Sigma is None:
            Sigma = disc.Sigma
               
        RS = Sigma * disc.R

        vvisc = np.nan_to_num(- 3 * self._dS[1:-1] / (RS[1:] + RS[:-1]))
        vwind = np.nan_to_num(-3 * disc.nu_dw[1:] /(2*disc.R[1:]))
        # question: why we use RS[1:]+RS[:-1]?
        return vvisc+vwind # Avoid nans when working with Sigma_G -> 0

    def _wind_ext(self, sigma, disc):
        '''compute the wind surface density extraction term '''
        nu_dw = disc.nu_dw
        ext = -12 * nu_dw * sigma / (disc.leverarm-1) / self._X ** 4
        return ext


    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu
        nu_dw = disc.nu_dw

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2
        if disc.alpha != 0.:
            tc_diff = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min() #* grid.Rc
        else:
            tc_diff = 1e5
        if disc.alpha_dw !=0:
            tc_adv = ((dXe2 * grid.Rc)/ (6*nu_dw)).min()
        else:
            tc_adv = 1e5

        tc = np.min((tc_diff, tc_adv))

        return self._tol * tc

    def __call__(self, dt, disc, tracers=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)
        self._init_fluxes_dw(disc)


        f_ss = self._fluxes()
        f_dw = self._fluxes_dw()
        f = f_ss + f_dw
        Sigma1 = disc.Sigma + dt * f
        e = self._wind_ext(Sigma1, disc)
        Sigma_new = Sigma1 + dt * e
        
        for t in tracers:
            if t is None: continue
            tracer_density = t*disc.Sigma
            t[:] = (dt*self._tracer_fluxes(t) + tracer_density) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new

class ViscousEvolutionFV(object):
    """Solves the 1D viscous evolution equation via a finite-volume method

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.
    """

    def __init__(self, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}

    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rh  = np.sqrt(grid.Rc)
        self._dR3 = np.diff(np.sqrt(grid.Rce)**3)

        self._dA = grid.Re
        self._dV = 0.5 * np.diff(grid.Re**2)

    def _init_fluxes(self, disc):
        """Cache the important variables needed for the fluxes"""
        nuRh = disc.nu * self._Rh

        S = np.zeros(len(nuRh) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuRh

        # Inner boundary
        if self._in_bound == 'Zero':            # Zero torque
            S[0] = 0
        elif self._in_bound == 'Mdot':          # Constant flux (appropriate for power law)
            S[0] = S[1] * self._Rh[0] / self._Rh[1]
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary
        if self._bound == 'Zero':
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2] ** 2 / S[-3]
        elif self._bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S[-1] = S[-2] * self._Rh[-2] / self._Rh[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S[-1] = S[-2] * self._Rh[-1] / self._Rh[-2]
        else:
            raise ValueError("Error boundary type not recognised")

        self._dS = 4.5 * np.diff(S) / self._dR3

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return np.diff(self._dA * self._dS) / self._dV

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return np.diff(self._dA * ds) / self._dV

    def viscous_velocity(self, disc, S = None):
        """Compute the radial velocity due to viscosity"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        if S is None:
            S = disc.Sigma 
        return - 0.5 * self._dS[1:-1] / (S[1:] + S[:-1])

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc

    def __call__(self, dt, disc, tracers=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        f = self._fluxes()
        Sigma_new = disc.Sigma + dt * f

        for t in tracers:
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*f) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new

class LBP_Solution(object):
    """Analytical solution for the evolution of an accretion disc,

    Lynden-Bell & Pringle, (1974).

    args:
        M     : Disc mass
        rc    : Critical radius at t=0
        n_c   : viscosity at rc
        gamma : radial dependence of nu, default=1
    """

    def __init__(self, M, rc, nuc, gamma=1):
        self._rc = rc
        self._nuc = nuc
        self._tc = rc * rc / (3 * (2 - gamma) ** 2 * nuc)

        self._Sigma0 = M * (2 - gamma) / (2 * np.pi * rc ** 2)
        self._gamma = gamma

    def __call__(self, R, t):
        """Surface density at R and t"""
        tt = t / self._tc + 1
        X = R / self._rc
        Xg = X ** - self._gamma
        ft = tt ** ((-2.5 + self._gamma) / (2 - self._gamma))

        return self._Sigma0 * ft * Xg * np.exp(- Xg * X * X / tt)

class Tabone22(object):
    '''Analytical solution to the viscous + wind disc, Tabone+2022
    args:
        M: initial disc mass
        rc: characteristic radius at t =0
        alpha_dw: alpha for disc wind
        alpha_ss: alpha for turbulence
        leverarm: lever arm of the disc wind
        aspect ratio at r = r_c
        sound speed at r = r_c
    '''
    def __init__(self, M, rc, alpha, alpha_dw, leverarm=3, aspect=0.01, sound_speed=1):

        self._Md = M
        self._rc = rc
        self._alpha_ss = alpha
        self._alpha_dw = alpha_dw
        self._alpha_tot = self._alpha_dw+self._alpha_ss
        self._la = leverarm
        # we adopt different thermal structures, the proportionality in previous thermal assumptions is not valid any more
        self._aspect_rc = aspect # aspect ratio (h/r) at r_c
        self._sp_rc = sound_speed # sound speed at r_c
        self._tacc0 = self._rc/(3*self._sp_rc*self._aspect_rc*self._alpha_tot)
        if self._alpha_ss ==0:
            self._ksi = 1/ (2*(self._la-1))
        elif self._alpha_ss !=0:
            self._psi = self._alpha_dw/self._alpha_ss
            self._ksi = 1/4 * (self._psi+1) * (np.sqrt(1+4*self._psi/((self._la-1)*(self._psi+1)**2))-1)

    def __call__(self, R, t):
        """Surface density at R and t"""
        sigma0  = self._Md / (2*np.pi*self._rc**2)
        if self._alpha_ss == 0:
            sigma_ct = sigma0 * np.exp(-t/(2*self._tacc0)) 
            sigma = sigma_ct * (R/self._rc)**(-1+self._ksi) * np.exp(-R/self._rc)
        elif self._alpha_ss !=0:
            rc_t = (1 + t/self._tacc0 /(1+self._psi)) * self._rc
            sigma_ct = sigma0 * (1+t/self._tacc0 /(1+self._psi))** (-0.5*(self._psi+2*self._ksi+5))
            sigma = sigma_ct * (R/rc_t)**(-1+self._ksi) * np.exp(-R/rc_t)
        
        return sigma # (divided by AU**2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .disc import AccretionDisc
    from .grid import Grid
    from .constants import AU, Msun
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar

    alpha = 5e-3

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2 * np.pi)

    grid = Grid(0.1, 1000, 1000, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)
    sol = LBP_Solution(M, Rd, nud, 1)

    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = ViscousEvolutionFV()

    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6]) * T0

    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        l, = plt.loglog(grid.Rc, disc.Sigma / AU ** 2)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t) / AU ** 2,
                        c=l.get_color(), ls='--')

    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma\,[\mathrm{g\,cm}]^{-2}$')
    plt.show()

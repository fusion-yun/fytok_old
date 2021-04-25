import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.numerical.Function import Function
from spdm.util.logger import logger

from .nclass_mod import nclass_mod


def transport_nclass(equilibrium: Equilibrium, core_profiles: CoreProfiles, core_transport: CoreTransport):

    # ----------------------------------------------------------------------
    # Model options
    #  kboot           : internal option for metrics
    #                     = 1 -> Hirshman, Sigmar model: fm[k]=Lck^*,           xngrth=<(n.gr(B))^2>/<B^2>.
    #                     = 2 -> Kessel model:           fm[1]=Rq/eps**3/2.     xngrth not used.
    #                     =   -> Shaing, et al model: fm[k]=Fk,                 xngrth=<n.gr(theta)>
    # ----------------------------------------------------------------------
    # kboot   = core_transport.codeparam.parameters[1]
    kboot = 0
    #    by default :     kboot = 0nMaxSpec

    # -------------------------------------------------------
    #  k_order        : order of v moments to be solved (-)
    #                   = 2 -> u and q
    #                   = 3 -> u, q, and u2
    #                   =   -> else error
    # -------------------------------------------------------
    k_order = 3

    # ----------------------------------------------------
    #  l_potato       :  option to include potato orbits (-)
    #                   = 0 -> off
    #                   =   -> else on
    l_potato = 1
    # ------------------------------------------------
    #  l_banana       : option to include banana viscosity (-)
    #                 = 0 -> off
    #                 =   -> else on
    l_banana = 1
    # ------------------------------------------------------------------
    #  l_pfirsch      : option to include Pfirsch-Schluter viscosity (-)
    #                 = 0 -> off
    #                 =   -> else on
    l_pfirsch = 1

    m_i = 1+len(core_profiles.ion)
    m_z = max([ion.z_ion for ion in core_profiles.ion])
    # ------------------------------------------------------------------------
    #  c_den          : density cutoff below which species is ignored (/m**3)
    # ------------------------------------------------------------------------
    c_den = 1.0e10
    # -----------------------------------------------------------------------------
    #  p_grphi        : radial electric field Phi' (V/rho)
    #  p_gr2phi       : radial electric field gradient Psi'(Phi'/Psi')' (V/rho**2)
    # -----------------------------------------------------------------------------

    psi_norm = equilibrium.profiles_1d.psi_norm
    rho_tor_norm = core_profiles.grid.rho_tor_norm
    rho_tor = core_profiles.grid.rho_tor
    rho_tor_bdry = rho_tor[-1]

    bt0_pr = equilibrium.profiles_1d.fpol * equilibrium.profiles_1d.gm1 / equilibrium.profiles_1d.gm9
    gph_pr = equilibrium.profiles_1d.gm1*equilibrium.profiles_1d.vprime * rho_tor_bdry

    b0 = equilibrium.vacuum_toroidal_field.b0
    r0 = equilibrium.vacuum_toroidal_field.r0

    r_inboard = equilibrium.profiles_1d.r_inboard
    r_outboard = equilibrium.profiles_1d.r_outboard

    xr0_pr = (r_inboard+r_outboard)*0.5

    #  external poloidal current has a flux surface (A)|
    xfs_pr = 2.0*scipy.constants.pi*xr0_pr*bt0_pr/scipy.constants.mu_0

    # Electron,Ion densities, temperatures and mass

    temperature = [core_profiles.electrons.temperature, *[ion.temperature for ion in core_profiles.ion]]

    dTemperature = [t.derivative for t in temperature]

    density = [core_profiles.electrons.density, *[ion.density for ion in core_profiles.ion]]

    dPressure = [core_profiles.electrons.pressure.derivative*equilibrium.profiles_1d.dpsi_drho_tor_norm,
                 *[ion.pressure.derivative*equilibrium.profiles_1d.dpsi_drho_tor_norm for ion in core_profiles.ion]]

    amu = [scipy.constants.m_e/scipy.constants.m_u,   # Electron mass in amu
           * [ion.z_ion for ion in core_profiles.ion]]

    q = equilibrium.profiles_1d.q
    dq_drho_tor = equilibrium.profiles_1d.q.derivative*equilibrium.profiles_1d.dpsi_drho_tor
    dphi_drho_tor = equilibrium.profiles_1d.dphi_dpsi*equilibrium.profiles_1d.dpsi_drho_tor
    fpol = equilibrium.profiles_1d.fpol
    kappa = equilibrium.profiles_1d.elongation

    c_potb = kappa[0]*b0/2.0/q[0]
    c_potl = r0*q[0]

    psi_norm = Function(core_transport.grid_d.rho_tor_norm, core_transport.grid_d.psi_norm)

    # Set input for NCLASS

    for ipr, x in enumerate(core_transport.grid_d.rho_tor_norm):
        x_psi = psi_norm(x)
        xqs = q(x_psi)
        # Geometry and electrical field calculations

        dpsi_drho_tor = equilibrium.profiles_1d.dpsi_drho_tor(x_psi)
        jparallel = equilibrium.profiles_1d.j_parallel(x_psi)*b0
        dphi_dpsi = equilibrium.profiles_1d.dphi_dpsi(x_psi)
        grad_rho_tor2 = equilibrium.profiles_1d.gm3(x_psi)

        #-------------#
        #Call NCLASS  #
        #-------------#
        qsf = q(x_psi)
        fex_iz = np.ndarray([3, 3])
        fex_iz[0, 1] = 0.0
        fex_iz[1, 1] = 0.0
        fex_iz[2, 1] = 0.0
        dencut = 1e10
        xr0 = xr0_pr(x_psi)
        rminx = x*rho_tor_bdry

        xeps = rminx/xr0
        xgph = gph_pr(x_psi)/4/(scipy.constants.pi**2)
        xfs = xfs_pr(x_psi)

        if (xgph == 0):
            xgph = 0.01

        p_fm = [0.0,  0.0, 0.0]

        if (kboot == 1):
            # -----------------
            #         Hirshman
            # -----------------
            p_ngrth = (xeps/(xqs*xr0))**2/2.0
            p_fm[0] = xqs*xr0
        elif (kboot == 2):
            # ---------------
            #         Kessel
            # ---------------
            p_ngrth = 0.0
            p_fm[0] = xqs*xr0/xeps**1.5
        else:
            # ---------------
            #         Shaing
            # ---------------
            p_ngrth = 1.0/(xqs*xr0)
            if (xeps > 0.0):
                eps2 = xeps**2
                b = np.sqrt(1.0 - eps2)
                a = (1.0 - b)/xeps
                c = (b**3.0)*(xqs*xr0)**2

                p_fm[0] = 1*a**(2*1)*(1.0 + 1*b)/c
                p_fm[2] = 2*a**(2*2)*(1.0 + 2*b)/c
                p_fm[1] = 3*a**(2*3)*(1.0 + 3*b)/c

        (
            iflag,  # " int"
            p_etap,     # - parallel electrical resistivity [Ohm*m]
            p_jbbs,     # - <J_bs.B> [A*T/m**2]
            p_jbex,     # - <J_ex.B> current response to fex_iz [A*T/m**2]
            p_jboh,     # - <J_OH.B> Ohmic current [A*T/m**2]
            m_s,        # - number of species [ms>1]
            jm_s,       # - isotope number of s [-]
            jz_s,       # - charge state of s [-]
            bsjbp_s,    # - <J_bs.B> driven by unit p'/p of s [A*T*rho/m**2]
            bsjbt_s,    # - <J_bs.B> driven by unit T'/T of s [A*T*rho/m**2]
            glf_s,      # - radial particle flux comps of s [rho/m**3/s]
                        #   m=1, banana-plateau, p' and T'
                        #   m=2, Pfirsch-Schluter
                        #   m=3, classical
                        #   m=4, banana-plateau, <E.B>
                        #   m=5, banana-plateau, external parallel force fex_iz
            dn_s,       # -diffusion coefficients (diag comp) [rho**2/s]
            vnnt_s,     # -convection velocity (off diag p',T' comps) [rho/s]
            vneb_s,     # -<E.B> particle convection velocity [rho/s]
            vnex_s,     # -external force particle convection velocity [rho/s]
            dp_ss,      # -diffusion coefficient of s2 on p'/p of s1 [rho**2/s]
            dt_ss,      # -diffusion coefficient of s2 on T'/T of s1 [rho**2/s]
                        # ---------------------------------------------------------
                        #          Momentum equation
            upar_s,     # UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]
                        #                m=1, p', T', Phi'
                        #                m=2, <E.B>
                        #                m=3, fex_iz
            utheta_s,   # UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]
                        #                  m=1, p', T'
                        #                  m=2, <E.B>
                        #                  m=3, fex_iz
                        # ---------------------------------------------------------
                        #           Energy equation
            qfl_s,      # -radial heat conduction flux comps of s [W*rho/m**3]
                        #             m=1, banana-plateau, p' and T'
                        #             m=2, Pfirsch-Schluter
                        #             m=3, classical
                        #             m=4, banana-plateau, <E.B>
                        #             m=5, banana-plateau, external parallel force fex_iz
            chi_s,      # -conduction coefficients (diag comp) [rho**2/s]
            vqnt_s,     # -conduction velocity (off diag p',T' comps) [rho/s]
            vqeb_s,     # -<E.B> heat convection velocity [rho/s]
            vqex_s,     # -external force heat convection velocity [rho/s]
            chip_ss,    # -heat cond coefficient of s2 on p'/p of s1 [rho**2/s]
            chit_ss,    # -heat cond coefficient of s2 on T'/T of s1 [rho**2/s]
                        # ---------------------------------------------------------
                        #           Friction coefficients
            calm_i,     # test particle (tp) friction matrix [-]
            caln_ii,    # field particle (fp) friction matrix [-]
            capm_ii,    # tp eff friction matrix [kg/m**3/s]
            capn_ii,    # fp eff friction matrix [kg/m**3/s]
                        # ---------------------------------------------------------
                        #           Viscosity coefficients
            ymu_s,      # normalized viscosity for s [kg/m**3/s]
                        # ---------------------------------------------------------
                        #           Miscellaneous
            sqz_s,      # orbit squeezing factor for s [-]
            xi_s,       # charge weighted density factor of s [-]
            tau_ss,     # 90 degree scattering time [s]
        ) = nclass_mod.nclass(
            m_i,                                              # number of isotopes (> 1) [-]
            m_z,                                              # highest charge state [-]
            equilibrium.profiles_1d.gm5(x_psi),               # <B**2> [T**2]
            equilibrium.profiles_1d.gm4(x_psi),               # <1/B**2> [1/T**2]
            core_profiles.e_field.parallel(x_psi)*b0,         # <E.B> [V*T/m]
            scipy.constants.mu_0*fpol(x_psi) / dpsi_drho_tor,  # mu_0*F/(dPsi/dr) [rho/m]
            p_fm,                                             # poloidal moments of drift factor for PS [/m**2]
            equilibrium.profiles_1d.trapped_fraction(x_psi),  # trapped fraction [-]
            equilibrium.profiles_1d.gm6(x_psi),               # <grad(rho)**2/B**2> [rho**2/m**2/T**2]
            dphi_drho_tor(x_psi),                             # potential gradient Phi' [V/rho]
            ((dpsi_drho_tor**2) * dq_drho_tor(x_psi)),        # Psi'(Phi'/Psi')' [V/rho**2]
            p_ngrth,                                          # <n.grad(Theta)> [/m]
            amu,                                              # atomic mass number [-]
            [dT(x_psi) for dT in dTemperature],               # temperature gradient [keV/rho]
            [T(x_psi) for T in temperature],                  # temperature [keV]
            [n(x_psi) for n in density],                      # density [/m**3]
            fex_iz[:, :],                                     # moments of external parallel force [T*n/m**3]
            [dp(x_psi) for dp in dPressure],                  # pressure gradient [keV/m**3/rho]
            ipr,                                              #
            l_banana,                                         # option to include banana viscosity [logical]
            l_pfirsch,                                        # option to include Pfirsch-Schluter viscosity [logical]
            l_potato,                                         # option to include potato orbits [logical]
            k_order,                                          # order of v moments to be solved [-]
                                                              #            =2 u and q (default)
                                                              #            =3 u, q, and u2
                                                              #            =else error
            # density cutoff below which species is ignored (default 1.e10) [/m**3]
            c_den,
            c_potb,                                           # kappa(0)*Bt(0)/[2*q(0)**2] [T]
            c_potl,                                           # q(0)*R(0) [m]
        )

        # Update utheta with edotb rescaling

        utheta_s[:, 2, :] = p_etap*(jparallel - p_jbbs)*utheta_s[:, 2, :]

        # Ion heatfluxes
        for k, sp in enumerate(core_transport.ion):
            # Ion heat flux
            sp.energy.d[ipr] += chi_s[k + 1]/grad_rho_tor2
            sp.energy.v[ipr] += vqnt_s[k + 1] + vqeb_s[k + 1]*p_etap*(jparallel - p_jbbs)
            sp.energy.flux[ipr] += np.sum(qfl_s[:, k + 1])
            # sp.chieff[ipr] += chi_s[k + 1]/grad_rho_tor2

            # ion particle fluxes
            sp.particle.d[ipr] += dn_s[k + 1]/grad_rho_tor2
            sp.particle.v[ipr] += vnnt_s[k + 1] + vneb_s[k + 1]*p_etap*(jparallel - p_jbbs)
            sp.particle.flux[ipr] += np.sum(glf_s[:, k + 1])

            # sp.deff[ipr] += dn_s[k + 1]/grad_rho_tor2

        # Ionic rotational  momentum transport
            sp.momentum_tor.d[ipr] += 0.0  # Need to set

        # update poloidal velocities
        # rotation_frequency_tor_sonic
        # profiles_rm.vpol[ipr] = np.sum(utheta_s[1, 1:3, k + 1])*fpol[i] / r0/fhat

        # Update toroidal velocities
        # if k != 1:
            # profiles_rm.vtor[ipr] = r0*p_fpolhat[i]/p_fpol[i]*profiles_rm.e_rad[i] +\
            #     p_fpolhat[i]*profiles_rm.vpol[ipr] - r0 * \
            #     p_fpolhat[i]/p_fpol[i]*e_r[3]

        # Add in electrons last in array
        core_transport.electrons.energy.d[ipr] += chi_s[0]/grad_rho_tor2
        core_transport.electrons.energy.v[ipr] += vqnt_s[0] + vqeb_s[0]*p_etap*(jparallel - p_jbbs)
        core_transport.electrons.energy.flux[ipr] += np.sum(qfl_s[:, 1])

        core_transport.chieff[ipr] += chi_s[1]/grad_rho_tor2  # need to set 0.0

        core_transport.electrons.particles.d[ipr] += dn_s[0]/grad_rho_tor2
        core_transport.electrons.particles.v[ipr] += vnnt_s[0] + vneb_s[1]*p_etap*(jparallel - p_jbbs)
        core_transport.electrons.particles.flux[ipr] += np.sum(glf_s[:, 1])

        # core_transport.electrons.particles.d.eff[ipr] += dn_s[1]/grad_rho_tor2  # need to set

        # rotational momentum transport
        core_transport.momentum_tor.d[ipr] = 0.0

        # resistivity and <j dot B>
        core_transport.conductivity_parallel[ipr] = 1.0 / p_etap
        core_profiles.j_bootstrap[ipr] = p_jbbs/b0

        # Recalculate E_r for storage
        # core_profiles.e_field.radial[ipr] = NotImplemented  # p_fpol[i]/r0/fhat*profiles_rm.vtor[i, 1]
        # core_profiles.e_field.poloidal[ipr] = NotImplemented  # -p_fpol[i]/r0*profiles_rm.vpol[i, 1]
        # core_profiles.e_field.toroidal[ipr] = NotImplemented  # p_fpol[i]/r0/fhat*profiles_rm.vtor[i, 1]

    # Extend to edge values
    # core_transport.chi[-1, :] = core_transport.chi[-1 - 1, :]
    # core_transport.Vt[-1, :] = core_transport.Vt[-1 - 1, :]
    # core_transport.heat_fluxes[-1, :] = core_transport.heat_fluxes[-1 - 1, :]
    # core_transport.chieff[-1, :] = core_transport.chieff[-1 - 1, :]

    # core_transport.d[-1, :] = core_transport.d[-1 - 1, :]
    # core_transport.Vp[-1, :] = core_transport.Vp[-1 - 1, :]
    # core_transport.particle_fluxes[-1, :] = core_transport.particle_fluxes[-1 - 1, :]
    # core_transport.deff[-1, :] = core_transport.deff[-1 - 1, :]

    # # rotational  momentum core_transport
    # core_transport.chimom[-1, :] = core_transport.chimom[-1 - 1, :]

    # core_transport.resistivity[-1] = core_transport.resistivity[-1 - 1]
    # core_transport.jboot[-1] = core_transport.jboot[-1 - 1]

    # Copy local values to profiles
    # core_profiles.vloop[ipr] = profiles_rm.vpol[1, :]

    return core_transport

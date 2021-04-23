import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium

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
    m_z = max([ion.z_charge for ion in core_profiles.ion])
    # ------------------------------------------------------------------------
    #  c_den          : density cutoff below which species is ignored (/m**3)
    # ------------------------------------------------------------------------
    c_den = 1.0e10
    # -----------------------------------------------------------------------------
    #  p_grphi        : radial electric field Phi' (V/rho)
    #  p_gr2phi       : radial electric field gradient Psi'(Phi'/Psi')' (V/rho**2)
    # -----------------------------------------------------------------------------

    n0 = np.sum([ion.density for ion in core_profiles.ion])

    psi_norm = equilibrium.profiles_1d.psi_norm
    rho_tor_norm = core_profiles.grid.rho_tor_norm

    amu_i = None
    grt_i = None

    q0 = equilibrium.profiles_1d.q
    bt0 = equilibrium.profiles_1d.fpol * equilibrium.profiles_1d.gm1 / equilibrium.profiles_1d.gm9

    b0 = equilibrium.vacuum_toroidal_field.b0
    r0 = equilibrium.vacuum_toroidal_field.r0

    rkappa0 = equilibrium.boundary.elongation

    p_eb_pr = core_profiles.e_field.parallel

    p_eb_pr = p_eb_pr * b0
    c_potb = rkappa0*bt0/2/q0/q0
    c_potl = q0*xr0

    n_ion = len(core_profiles.ion)
    # Get some control variables from the control structure
    m_i = 1 + n_ion  # m_i should really be nions and 1+ nions = ms

    # Allocate mapping for NCLASS quantities (only valid when no multiple charge state ion present)
    m_s = len(core_profiles.ion) + 1

    jif_s = np.zeros([m_s], dtype=float)
    jzf_s = np.zeros([m_s], dtype=float)

    k = 1
    jzf_s[k] = 1  # electrons
    jif_s[k] = 1

    # DO i = 1, profiles.nmain
    for i, ion in enumerate(core_profiles.ion):
        k = k + 1
        jif_s[k] = i + 1
        jzf_s[k] = 1  # single charge state species

    # for i, ion in enumerate(core_profiles.ion):
    #    do j = 1, profiles.nzmap[profiles.nmain + i]

    #        k = k + 1
    #         jif_s[k] = profiles.nmain + 1 + i
    #         jzf_s[k] = j  # multiple charge states
    #     # END DO
    # # END DO

    m_z = maxval(jzf_s[:])

    # Note dimensioning of d_n, v_eb, v_nt should now be valid also for mcs
    # if[istat /= 0) THEN
    #    #    message='DEA_LOAD_TRANSPORT:: Error allocating densities'
    #    #   call setStateToFailure(Error,message)
    #    RETURN
    # END if

    temp_i = np.vstack([core_profiles.electrons.temperature, *[ion.temperature for ion in core_profiles.ion]])
    density_i = np.vstack([core_profiles.electrons.temperature, *[ion.temperature for ion in core_profiles.ion]])

    # Electron density, temperature and mass

    amu_i = [scipy.constants.m_e/scipy.constants.m_u     # Electron mass in amu
             * [ion.z for ion in core_profiles.ion]]

    # Ion densities, temperatures and mass
    # DO I = 1, profiles_rm.nmain

    #     den_riz[:, i + 1, jzf_s[i + 1]] = profiles_rm.ni[:, i]
    #     den_r[:, i + 1, jzf_s[i + 1]] = profiles.ni[:, i]
    #     temp_ri[:, i + 1] = profiles_rm.ti[:, i]
    #     temp_r[:, i + 1] = profiles.ti[:, i]
    #     amu_i[i + 1] = profiles_rm.z_mass[i]

    # # END DO
    # k = profiles.nmain + 1
    # DO I = 1, profiles.nimp
    # DO J = 1, profiles.nzmap[i]:
    #     k = k + 1
    #     den_riz[:, i + 1 + profiles_rm.nmain, jzf_s[k]] = profiles_rm.ni[:, k - 1]
    #     den_r[:, i + 1 + profiles_rm.nmain, jzf_s[k]] = profiles.ni[:, k - 1]
    #     temp_ri[:, i + 1 + profiles_rm.nmain] = profiles_rm.ti[:, k - 1]
    #     temp_r[:, i + 1 + profiles_rm.nmain] = profiles.ti[:, k - 1]
    #     amu_i[i + 1 + profiles_rm.nmain] = profiles_rm.z_mass[k - 1]
    # END DO
    # END DO
    # m_s = m_i # Only valid for scs ions

    a1 = equilibrium.boundary.minor_radius
    kappa = equilibrium.boundary.elongation
    c_potb = kappa*b0/2.0/equilibrium.profiles_1d.q[0]
    c_potl = r0*equilibrium.profiles_1d.q[0]

    # Set input for NCLASS

    for i, x in enumerate(core_transport.grid_d.rho_tor_norm):
        x_psi = core_transport.grid_d.psi_norm(x)

        # Geometry and electrical field calculations
        grad_rho_tor2 = equilibrium.profiles_1d.gm3(x_psi)
        fpol = equilibrium.profiles_1d.fpol(x_psi)
        dpsi_drho_tor = equilibrium.profiles_1d.dpsi_drho_tor(x_psi)
        jparallel = equilibrium.profiles_1d.j_parallel(x_psi)*b0
        # Parallel and radial electric field
        # Local variables for transport coefficients
        #-------------#
        #Call NCLASS  #
        #-------------#

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
            m_i,                                                       # number of isotopes (> 1) [-]
            m_z,                                                       # highest charge state [-]
            equilibrium.profiles_1d.gm5(x_psi),                        # <B**2> [T**2]
            equilibrium.profiles_1d.gm4(x_psi),                        # <1/B**2> [1/T**2]
            p_eb[i],                                                   # <E.B> [V*T/m]
            scipy.constants.mu_0*fpol / dpsi_drho_tor,               # mu_0*F/(dPsi/dr) [rho/m]
            p_fm[i],                                                   # poloidal moments of drift factor for PS [/m**2]
            equilibrium.profiles_1d.trapped_fraction(x_psi),           # trapped fraction [-]
            equilibrium.profiles_1d.gm6(x_psi),                        # <grad(rho)**2/B**2> [rho**2/m**2/T**2]
            equilibrium.profiles_1d.dphi_dpsi(x_psi) * dpsi_drho_tor,  # potential gradient Phi' [V/rho]
            ((dpsi_drho_tor**2) *                                      # second potential gradient Psi'(Phi'/Psi')' [V/rho**2]
                equilibrium.profiles_1d.dphi_dpsi.derivative(x_psi)),  #
            p_ngrth[i],                                                # <n.grad(Theta)> [/m]
            amu_i[:],                                                  # atomic mass number [-]
            grt_i[i, :],                                               # temperature gradient [keV/rho]
            temp_i[i, :],                                              # temperature [keV]
            den_iz[i, :],   # density [/m**3]
            fex_iz[:, i, :],  # moments of external parallel force [T*n/m**3]
            grp_iz[i, :],   # pressure gradient [keV/m**3/rho]
            ipr,            #
            l_banana,       # option to include banana viscosity [logical]
            l_pfirsch,      # option to include Pfirsch-Schluter viscosity [logical]
            l_potato,       # option to include potato orbits [logical]
            k_order,        # order of v moments to be solved [-]
                            #            =2 u and q (default)
                            #            =3 u, q, and u2
                            #            =else error
            c_den[i],       # C_DEN-density cutoff below which species is ignored (default 1.e10) [/m**3]
            c_potb[i],      # C_POTB-kappa(0)*Bt(0)/[2*q(0)**2] [T]
            c_potl[i],      # C_POTL-q(0)*R(0) [m]
        )

        # Update utheta with edotb rescaling

        utheta_s[:, 2, :] = p_etap*(jparallel - p_jbbs)*utheta_s[:, 2, :]

        # Ion heatfluxes
        for k, sp in enumerate(core_transport.ion):
            # Ion heat flux
            sp.chi[i, k] += chi_s[k + 1]/grad_rho_tor2

            sp.Vt[i, k] += vqnt_s[k + 1] + vqeb_s[k + 1]*p_etap*(jparallel - p_jbbs)

            sp.heat_fluxes[i, k] += np.sum(qfl_s[:, k + 1])

            sp.chieff[i, k] += chi_s[k + 1]/grad_rho_tor2

            # ion particle fluxes
            sp.d[i, k] += + dn_s[k + 1]/grad_rho_tor2

            sp.vp[i, k] += vnnt_s[k + 1] + vneb_s[k + 1]*p_etap*(jparallel - p_jbbs)

            sp.particle_fluxes[i, k] += np.sum(glf_s[:, k + 1])

            sp.deff[i, k] += dn_s[k + 1]/grad_rho_tor2

        # Ionic rotational  momentum transport
        sp.chimom[i, k] += 0.0  # Need to set

        # update poloidal velocities

        profiles_rm.vpol[i, k] = np.sum(utheta_s[1, 1:3, k + 1])*fpol[i] / r0/fhat

        # Update toroidal velocities
        if k != 1:  # THEN
            profiles_rm.vtor[i, k] = r0*p_fpolhat[i]/p_fpol[i]*profiles_rm.e_rad[i] +\
                p_fpolhat[i]*profiles_rm.vpol[i, k] - r0 * \
                p_fpolhat[i]/p_fpol[i]*e_r[3]
            # end if
        # END DO

        # Add in electrons last in array
        k = k + 1
        core_transport.chi[i, k] += chi_s[0]/grad_rho_tor2
        core_transport.Vt[i, k] += vqnt_s[0] + vqeb_s[0]*p_etap*(jparallel - p_jbbs)
        core_transport.heat_fluxes[i, k] += np.sum(qfl_s[:, 1])

        core_transport.chieff[i, k] = core_transport.chieff[i, k] + chi_s[1]/grad_rho_tor2  # need to set 0.0

        core_transport.d[i, k] += dn_s[0]/grad_rho_tor2
        core_transport.Vp[i, k] += vnnt_s[0] + vneb_s[1]*p_etap*(jparallel - p_jbbs)
        core_transport.particle_fluxes[i, k] += np.sum(glf_s[:, 1])

        core_transport.deff[i, k] += dn_s[1]/grad_rho_tor2  # need to set

        # rotational momentum transport
        core_transport.chimom[i, k] = 0.0

        # resistivity and <j dot B>
        core_transport.resistivity[i] = p_etap
        core_transport.jboot[i] = p_jbbs/b0

        # Recalculate E_r for storage
        e_r[1] = p_fpol[i]/r0/fhat*profiles_rm.vtor[i, 1]
        e_r[2] = -p_fpol[i]/r0*profiles_rm.vpol[i, 1]
        profiles_rm.e_rad[i] = e_r[1] + e_r[2] + e_r[3]
    # END DO

    # Extend to edge values
    core_transport.chi[-1, :] = core_transport.chi[-1 - 1, :]
    core_transport.Vt[-1, :] = core_transport.Vt[-1 - 1, :]
    core_transport.heat_fluxes[-1, :] = core_transport.heat_fluxes[-1 - 1, :]
    core_transport.chieff[-1, :] = core_transport.chieff[-1 - 1, :]

    core_transport.d[-1, :] = core_transport.d[-1 - 1, :]
    core_transport.Vp[-1, :] = core_transport.Vp[-1 - 1, :]
    core_transport.particle_fluxes[-1, :] = core_transport.particle_fluxes[-1 - 1, :]
    core_transport.deff[-1, :] = core_transport.deff[-1 - 1, :]

    # rotational  momentum core_transport
    core_transport.chimom[-1, :] = core_transport.chimom[-1 - 1, :]

    core_transport.resistivity[-1] = core_transport.resistivity[-1 - 1]
    core_transport.jboot[-1] = core_transport.jboot[-1 - 1]

    # Copy local values to profiles

    profiles.vpol[1, :] = profiles_rm.vpol[1, :]
    # DO I = 2, profiles.nr:
    #     profiles.vpol[i, :] = 2*profiles_rm.vpol[i, :] - profiles.vpol[i - 1, :]
    #     profiles.e_rad[i] = 2*profiles_rm.e_rad[i] - profiles.e_rad[i - 1]
    #     profiles.vtor[i, :] = 2*profiles_rm.vtor[i, :] - profiles.vtor[i - 1, :]
    # END DO

    # Free temporary memory holders
    # DEALLOCATE(den_riz, temp_ri, den_r, temp_r, grn_riz, grp_riz, &
    #            grt_ri, v_eb, v_nt, amu_i, cur_rm, jdotb_rm, &
    #            glf_s, jz_s, utheta_s, STAT=istat)

    # Note dimensioning of d_n, v_eb, v_nt is only valid when no mcs ions are present
    # if istat != 0:  # THEN
    #     #      message='DEA_LOAD_NCLASS:: Error deallocating local densities'
    #     #      call setStateToFailure(Error,message)
    #     RETURN

    # END if

    return core_transport

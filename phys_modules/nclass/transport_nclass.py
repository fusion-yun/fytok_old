import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium

from .nclass_mod import nclass_mod


def transport_nclass(equilibrium: Equilibrium, core_profiles: CoreProfiles, core_transport: CoreTransport):

    rho_tor_norm = core_profiles.grid.rho_tor_norm

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
    # ----------------------------------------------------
    #  kpotato       :  option to include potato orbits (-)
    #                   = 0 -> off
    #                   =   -> else on
    # -----------------------------------------------------
    # kpotato =  core_transport.codeparam.parameters[2]
    kpotato = 1
    #     by default :     kpotato = 1
    # -------------------------------------------------------
    #  k_order        : order of v moments to be solved (-)
    #                   = 2 -> u and q
    #                   = 3 -> u, q, and u2
    #                   =   -> else error
    # -------------------------------------------------------
    k_order = 3
    #      by default :     k_order = 3
    # ------------------------------------------------
    #  kbanana       : option to include banana viscosity (-)
    #                 = 0 -> off
    #                 =   -> else on
    # ------------------------------------------------
    kbanana = 1
    #      by default :     kbanana = 1
    # ------------------------------------------------------------------
    #  kpfirsch      : option to include Pfirsch-Schluter viscosity (-)
    #                 = 0 -> off
    #                 =   -> else on
    # ------------------------------------------------------------------
    kpfirsch = 1

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

    p_b2 = equilibrium.profiles_1d.gm5
    p_bm2 = equilibrium.profiles_1d.gm4
    p_eb = None
    p_fpol = equilibrium.profiles_1d.fpol
    p_fhat = (scipy.constants.mu_0*equilibrium.profiles_1d.fpol * equilibrium.profiles_1d.dpsi_drho_tor)
    p_fm = np.ndarray([3], dtype=float)
    p_grbm2 = equilibrium.profiles_1d.gm6
    p_grphi = (equilibrium.profiles_1d.dphi_dpsi * equilibrium.profiles_1d.dpsi_drho_tor)
    p_gr2phi = ((equilibrium.profiles_1d.dpsi_drho_tor**2) * equilibrium.profiles_1d.dphi_dpsi.derivative)
    p_ngrth = None

    amu_i = None
    grt_i = None

    q0 = equilibrium.profiles_1d.q
    bt0 = equilibrium.profiles_1d.fpol * equilibrium.profiles_1d.gm1 / equilibrium.profiles_1d.gm9

    b0 = equilibrium.vacuum_toroidal_field.b0
    r0 = equilibrium.vacuum_toroidal_field.r0

    rkappa0 = equilibrium.profiles_1d.elongation

    p_eb_pr = core_profiles.e_field.parallel

    p_eb_pr = p_eb_pr * b0
    c_potb = rkappa0*bt0/2/q0/q0
    c_potl = q0*xr0

    gr2phi = psidrho * gr2phi

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

    den_riz = 0.0
    den_r = 0.0

    # Electron density, temperature and mass
    den_riz[:, 1, 1] = core_profiles.electrons.density
    den_r[:, 1, 1] = profiles.ne[:]
    temp_ri[:, 1] = core_profiles.electrons.temperature
    temp_r[:, 1] = profiles.te[:]
    amu_i[1] = z_electronmass  # Electron mass in amu

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
    a1 = geometry_rm.a0

    # DO k = 1, m_s  # Over species

    #    im = jif_s[k]
    #     iza = jzf_s[k]

    #     DO i = 1, profiles_rm.nr  # Over radial grid

    #        if[i == 1] THEN

    #            grt_ri[i, im] = 0.0
    #             grn_riz[i, im, iza] = 0.0
    #             grp_riz[i, im, iza] = 0.0

    #         ELSE

    #            dr = (profiles.rho[i] - profiles.rho[i - 1])
    #             grt_ri[i, im] = (temp_r[i, im] - temp_r[i - 1, im])/dr
    #             grn_riz[i, im, iza] = (den_r[i, im, iza] - den_r[i - 1, im, iza])/dr
    #             grp_riz[i, im, iza] = (den_r[i, im, iza]*temp_r[i, im] &
    #                                    - den_r[i - 1, im, iza]*temp_r[i - 1, im])/dr

    #         END if

    #     # END DO  # Over radial grid

    # END DO  # Over species

    c_potb = geometry_rm.kappa[1]*b0/2.0/geometry_rm.q[1]
    c_potl = r0*geometry_rm.q[1]

    jdotb_rm = geometry_rm.jdotb[:]*b0

    # pis  jdotb_rm[1] = jdotb_rm[2]
    # pis  jdotb_rm(geometry_rm.nr) = jdotb_rm(geometry_rm.nr-1]

    # Set input for NCLASS
    # DO I = 1, profiles_rm.nr
    for i, x in enumerate(core_transport.grid_d.rho_tor_norm):

        # Geometry and electrical field calculations
        # p_b2 = geometry_rm.b2[i]
        # p_bm2 = geometry_rm.bm2[i]
        # p_fhat = geometry_rm.fhat[i]
        # p_fm  = geometry_rm.fm[i, :]
        # p_grbm2 = geometry_rm.gr2bm2[i]
        # p_ngrth = geometry_rm.grth[i]
        # p_ft = max(geometry_rm.f_t[i], epsilon[1.0]]

        # Parallel and radial electric field
        # p_eb = 1.0  # rescale with total inductive current after call to NCLASS optin note taken here

        e_r[1] = p_fpol[i]/r0/p_fhat[i]*profiles_rm.vtor[i, 1]
        e_r[2] = -geometry_rm.f[i]/r0*profiles_rm.vpol[i, 1]  # pistemp check fr ninimp version
        e_r[3] = grp_riz[i, jif_s[2], jzf_s[2]]*z_kev2joule/jzf_s[2]/z_electroncharge/den_riz[i, jif_s[2], jzf_s[2])

        profiles_rm.e_rad[i] = e_r[1] + e_r[2] + e_r[3]

        # p_grphi = -profiles_rm.e_rad[i]
        # p_gr2phi = 0.0

        # Local variables for transport coefficients
        d_n[:] = 0.0
        v_eb[:] = 0.0
        v_nt[:] = 0.0

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
            gfl_s,      # - radial particle flux comps of s [rho/m**3/s]
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
            qfl_s,      # " rank-2 array('d') with bounds (5,mxms)"
            chi_s,      # " rank-1 array('d') with bounds (mxms)"
            vqnt_s,     # " rank-1 array('d') with bounds (mxms)"
            vqeb_s,     # " rank-1 array('d') with bounds (mxms)"
            vqex_s,     # " rank-1 array('d') with bounds (mxms)"
            chip_ss,    # " rank-2 array('d') with bounds (mxms,mxms)"
            chit_ss,    # " rank-2 array('d') with bounds (mxms,mxms)"
                        # ---------------------------------------------------------
                        #           Friction coefficients
            calm_i,     # " rank-3 array('d') with bounds (3,3,m_i)"
            caln_ii,    # " rank-4 array('d') with bounds (3,3,m_i,m_i)"
            capm_ii,    # " rank-4 array('d') with bounds (3,3,m_i,m_i)"
            capn_ii,    # " rank-4 array('d') with bounds (3,3,m_i,m_i)"
                        # ---------------------------------------------------------
                        #           Viscosity coefficients
            ymu_s,      # " rank-3 array('d') with bounds (3,3,mxms)"
                        # ---------------------------------------------------------
                        #           Miscellaneous
            sqz_s,      # " rank-1 array('d') with bounds (mxms)"
            xi_s,       # " rank-1 array('d') with bounds (mxms)"
            tau_ss,     # " rank-2 array('d') with bounds (mxms,mxms)")
        ) = nclass_mod.nclass(
            m_i,            # number of isotopes (> 1) [-]
            m_z,            # highest charge state [-]
            p_b2[i],        # <B**2> [T**2]
            p_bm2[i],       # <1/B**2> [/T**2]
            p_eb[i],        # <E.B> [V*T/m]
            p_fhat[i],      # mu_0*F/(dPsi/dr) [rho/m]
            p_fm[i],        # poloidal moments of drift factor for PS [/m**2]
            p_ft[i],        # trapped fraction [-]
            p_grbm2[i],     # <grad(rho)**2/B**2> [rho**2/m**2/T**2]
            p_grphi[i],     # potential gradient Phi' [V/rho]
            p_gr2phi[i],    # second potential gradient Psi'(Phi'/Psi')' [V/rho**2]
            p_ngrth[i],     # <n.grad(Theta)> [/m]
            amu_i[:],       # atomic mass number [-]
            grt_i[i, :],     # temperature gradient [keV/rho]
            temp_i[i, :],    # temperature [keV]
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

        utheta_s[:, 2, :] = p_etap*(jdotb_rm[i] - p_jbbs)*utheta_s[:, 2, :]

        # Ion heatfluxes
        for k, sp in enumerate(core_transport.ion):
            # Ion heat flux
            sp.chi[i, k] = sp.chi[i, k] + chii[k + 1]/geometry_rm.grho2[i]

            sp.Vt[i, k] = sp.VT[i, k] + vq_nt[k + 1] + vq_eb[k + 1]*p_etap*(jdotb_rm[i] - p_jbbs)

            sp.heat_fluxes[i, k] = sp.heat_fluxes[i, k] + SUM(QFL_S[:, k + 1])

            sp.chieff[i, k] = sp.chieff[i, k] + chii[k + 1]/geometry_rm.grho2[i]

            # ion particle fluxes
            sp.d[i, k] = sp.d[i, k] + d_n[k + 1]/geometry_rm.grho2[i]

            sp.vp[i, k] = sp.vp[i, k] + v_nt[k + 1] + v_eb[k + 1]*p_etap*(jdotb_rm[i] - p_jbbs)

            sp.particle_fluxes[i, k] = sp.particle_fluxes[i, k] + SUM(GFL_S[:, k + 1])

            sp.deff[i, k] = sp.deff[i, k] + d_n[k + 1]/geometry_rm.grho2[i]

        # Ionic rotational  momentum transport
        sp.chimom[i, k] = sp.chimom[i, k] + 0.0  # Need to set

        # update poloidal velocities

        profiles_rm.vpol[i, k] = SUM(utheta_s[1, 1:3, k + 1])*geometry_rm.f[i] / r0/geometry_rm.fhat[i]

        # Update toroidal velocities
        if k != 1:  # THEN
            profiles_rm.vtor[i, k] = r0*geometry_rm.fhat[i]/geometry_rm.f[i]*profiles_rm.e_rad[i] +\
                geometry_rm.fhat[i]*profiles_rm.vpol[i, k] - r0 * \
                geometry_rm.fhat[i]/geometry_rm.f[i]*e_r[3]
            # end if
        # END DO

        # Add in electrons last in array
        k = k + 1
        core_transport.chi[i, k] = core_transport.chi[i, k] + chii[1]/geometry_rm.grho2[i]
        core_transport.Vt[i, k] = core_transport.Vt[i, k] + vq_nt[1] + vq_eb[1]*p_etap*(jdotb_rm[i] - p_jbbs)
        core_transport.heat_fluxes[i, k] = core_transport.heat_fluxes[i, k] + SUM(QFL_S[:, 1])

        core_transport.chieff[i, k] = core_transport.chieff[i, k] + chii[1]/geometry_rm.grho2[i]  # need to set 0.0

        core_transport.d[i, k] = core_transport.d[i, k] + d_n[1]/geometry_rm.grho2[i]
        core_transport.Vp[i, k] = core_transport.Vp[i, k] + v_nt[1] + v_eb[1]*p_etap*(jdotb_rm[i] - p_jbbs)
        core_transport.particle_fluxes[i, k] = core_transport.particle_fluxes[i, k] + SUM(GFL_S[:, 1])

        core_transport.deff[i, k] = core_transport.deff[i, k] + d_n[1]/geometry_rm.grho2[i]  # need to set

        # rotational momentum transport
        core_transport.chimom[i, k] = 0.0

        # resistivity and <j dot B>
        core_transport.resistivity[i] = p_etap
        core_transport.jboot[i] = p_jbbs/b0

        # Recalculate E_r for storage
        e_r[1] = geometry_rm.f[i]/r0/geometry_rm.fhat[i]*profiles_rm.vtor[i, 1]
        e_r[2] = -geometry_rm.f[i]/r0*profiles_rm.vpol[i, 1]
        profiles_rm.e_rad[i] = e_r[1] + e_r[2] + e_r[3]
    # END DO

    # Extend to edge values
    core_transport.rho[profiles.nr] = (profiles.rho[profiles.nr] + profiles.rho[profiles.nr - 1])/2
    core_transport.chi[profiles.nr, :] = core_transport.chi[profiles.nr - 1, :]
    core_transport.Vt[profiles.nr, :] = core_transport.Vt[profiles.nr - 1, :]
    core_transport.heat_fluxes[profiles.nr, :] = core_transport.heat_fluxes[profiles.nr - 1, :]
    core_transport.chieff[profiles.nr, :] = core_transport.chieff[profiles.nr - 1, :]

    core_transport.d[profiles.nr, :] = core_transport.d[profiles.nr - 1, :]
    core_transport.Vp[profiles.nr, :] = core_transport.Vp[profiles.nr - 1, :]
    core_transport.particle_fluxes[profiles.nr, :] = core_transport.particle_fluxes[profiles.nr - 1, :]
    core_transport.deff[profiles.nr, :] = core_transport.deff[profiles.nr - 1, :]

    # rotational  momentum core_transport
    core_transport.chimom[profiles.nr, :] = core_transport.chimom[profiles.nr - 1, :]

    core_transport.resistivity[profiles.nr] = core_transport.resistivity[profiles.nr - 1]
    core_transport.jboot[profiles.nr] = core_transport.jboot[profiles.nr - 1]

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
    #            gfl_s, jz_s, utheta_s, STAT=istat)

    # Note dimensioning of d_n, v_eb, v_nt is only valid when no mcs ions are present
    # if istat != 0:  # THEN
    #     #      message='DEA_LOAD_NCLASS:: Error deallocating local densities'
    #     #      call setStateToFailure(Error,message)
    #     RETURN

    # END if

    return core_transport

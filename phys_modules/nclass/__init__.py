import numpy as np
from .nclass_mod import nclass_mod
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium


def transport_nclass(equilibrium: Equilibrium, core_profiles: CoreProfiles) -> CoreTransport:
    rho_tor_norm = core_profiles.grid.rho_tor_norm
    core_transport = CoreTransport()

    # ----------------------------------------------------------------------
    # Model options
    #  kboot           : internal option for metrics
    #                     = 1 -> Hirshman, Sigmar model: fm(k)=Lck^*,
    #                                      xngrth=<(n.gr(B))^2>/<B^2>.
    #                     = 2 -> Kessel model:           fm[1]=Rq/eps**3/2.
    #                                      xngrth not used.
    #                     =   -> Shaing, et al model: fm(k)=Fk,
    #                                      xngrth=<n.gr(theta)>
    # ----------------------------------------------------------------------
    # kboot   = core_transport%codeparam%parameters[1]
    kboot = 0
    #    by default :     kboot = 0nMaxSpec
    # ----------------------------------------------------
    #  kpotato       :  option to include potato orbits (-)
    #                   = 0 -> off
    #                   =   -> else on
    # -----------------------------------------------------
    # kpotato =  core_transport%codeparam%parameters(2)
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

    # p_eb_pr                        = p_eb_pr * bt0_pr
    p_eb_pr = p_eb_pr * b0
    c_potb = rkappa0*bt0/2/q0/q0
    c_potl = q0*xr0
    gr2phi = psidrho * gr2phi
    #      not in the CPO
    #      force1          =    1st moment of external forces for the main ion (datak.source.totale.q?)
    #      force2          =    2nd moment of external forces (?)
    #      force3          =    3rd moment of external forces for the main ion (datak.source.totale.wb?)
    force1 = 0.
    force2 = 0.
    force3 = 0.
    #
    #  kmmaj : mass of the injected NBI particle
    #  ksmaj : charge of the injected NBI particle
    #
    kmmaj = 2
    ksmaj = 2

    # --------------------------------------------------------------------|
    #       xfs     : external poloidal current has a flux surface (A)|
    # --------------------------------------------------------------------|
    xfs_pr = 2.0*z_pi*xr0_pr*bt0_pr/z_mu0
    # -----------------------
    # Loop over radial nodes
    # -----------------------

    #
    # external force set to zero
    #
    p_grphi = 0.0
    p_gr2phi = 0.0
    fex_iz[1, kmmaj, ksmaj] = force1[ipr]
    fex_iz[2, kmmaj, ksmaj] = force2[ipr]
    fex_iz[3, kmmaj, ksmaj] = force3[ipr]
    dencut = 1e10
    xr0 = xr0_pr[ipr]
    rminx = rminx_pr[ipr]
    xqs = xqs_pr[ipr]

    p_ft = p_ft_pr[ipr]
    bt0 = bt0_pr[ipr]
    xdelp = xvpr_pr[ipr]
    xvpr = xdelp
    ab = 1.0
    p_b2 = p_b2_pr[ipr]
    p_bm2 = p_bm2_pr[ipr]
    q0 = xqs_pr[1]
    xeps = rminx/xr0
    xgph = gph_pr[ipr]/4/z_pi/z_pi
    xfs = xfs_pr[ipr]
    den_iz[:, :] = dencut
    den_iz[1:, pcharge[:]] = den[ipr, 1:]

    den_iz[1, 1] = 0.0
    dent = 0.0

    den_iz[1, 1] = den_iz[1, 1]+float(ia)*den_iz[ima, ia]
    denz2[ima] = denz2[ima]+den_iz[ima, ia] * float(ia)**2
    dent = dent+den_iz[ima, ia]*tempi[ipr, ima]

    denz2[1] = den_iz[1, 1]
    dent = dent+den_iz[1, 1]*tempi[ipr, 1]
    xbeta = dent*z_j7kv/(bt0**2/2.0/z_mu0)

    xi[:, :] = den_iz[:, :]*float(ia)**2 / denz2[ima]
    if (den_iz[:, :] == 0):
        grp_iz[:, :] = 0
    else:
        grp_iz[:, :] = grti[ipr, ima]*den_iz[ima, ia] + tempi[ipr, ima]*grden[ipr, ima]

    if (xgph == 0):
        xgph = 0.01

    p_fhat = xqs/xgph
    p_grbm2 = p_grbm2_pr[ipr]
    p_eb = p_eb_pr[ipr]
    c_potb = rkappa0*bt0/2/q0/q0
    c_potl = q0*xr0
    if (kboot == 1):
        # -----------------
        #         Hirshman
        # -----------------
        p_ngrth = (xeps/(xqs*xr0))**2/2.0
        p_fm[1] = xqs*xr0
        p_fm(2) = 0.0
        p_fm(3) = 0.0
    elif (kboot == 2):
        # ---------------
        #         Kessel
        # ---------------
        p_ngrth = 0.0
        p_fm[1] = xqs*xr0/xeps**1.5
    else:
        # ---------------
        #         Shaing
        # ---------------
        p_ngrth = 1.0/(xqs*xr0)
        p_fm[:] = 0

        if (xeps < 0.0):
            eps2 = xeps**2
            b = np.sqrt(1.0-eps2)
            a = (1.0-b)/xeps
            c = (b**3.0)*(xqs*xr0)**2
            p_fm[:] = xm*a**(2.0*xm)*(1.0+xm*b)/c

    # -------------------------------------------
    # Find significant charge states and mapping
    # -------------------------------------------
    m_s = 0
    if (den_iz[:, :] < c_den):
        m_s = m_s+1
        # ---------------------------------------------------------------
        #  Set isotope number and charge state for this species
        # ---------------------------------------------------------------
        jm_s[m_s] = ima
        if (amu_i[ima] < 0.5):
            jz_s[m_s] = -iza
        else:
            jz_s[m_s] = iza

    # ---------------------------
    # Calculate thermal velocity
    # ---------------------------

    vti[:] = np.sqrt(2.0*z_j7kv*tempi[ipr, ima] / amu_i(ima)/z_protonmass)
    temp_i(ima) = tempi[ipr, ima]
    grt_i(ima) = grti[ipr, ima]

    (iflag,
        P_ETAP, P_JBBS, P_JBEX, P_JBOH,
        M_S, JM_S, JZ_S,
        BSJBP_S, BSJBT_S,
        GFL_S, DN_S, VNNT_S, VNEB_S, VNEX_S, DP_SS, DT_SS,
        UPAR_S, UTHETA_S,
        QFL_S, CHI_S, VQNT_S, VQEB_S, VQEX_S,
        CHIP_SS, CHIT_SS,
        CALM_I, CALN_II, CAPM_II, CAPN_II, YMU_S,
        SQZ_S, XI_S, TAU_SS
     ) = nclass_mod.nclass(m_i, m_z, p_b2, p_bm2, p_eb,
                           p_fhat, p_fm, p_ft, p_grbm2, p_grphi,
                           p_gr2phi, p_ngrth, amu_i, grt_i, temp_i,
                           den_iz, fex_iz, grp_iz, ipr)

    # output formatting
    #
    # ----------------------
    # Output variables
    #   <Jbs.B> (A*T/m**2)
    # ----------------------
    #        tabtr[1,ipr)      = P_JBBS/zj7kv
    tabtr[1, :] = p_jbbs
    core_transport.jboot[ipr] = p_jbbs / b0
    # write(*,*) P_JBBS / b0
    # ------------------------------------------------------------------
    # <Jbs.B> driven by unit p'/p of species s (A*T*rho/m**3)
    # s  : 1 -> electrons, 2 -> espece ionique 1, 3 -> espece ionique 2
    # ------------------------------------------------------------------
    tabtr[2, :] = bsjbp_s[1]
    tabtr[3, :] = bsjbp_s[2]
    tabtr[4, :] = bsjbp_s[3]
    # ------------------------------------------------------------------
    # <Jbs.B> driven by unit T'/T of s (A*T*rho/m**3)
    # s  : 1 -> electrons, 2 -> espece ionique 1, 3 -> espece ionique 2
    # ------------------------------------------------------------------
    tabtr[5, :] = bsjbt_s[1]
    tabtr[6, :] = bsjbt_s[2]
    tabtr[7, :] = bsjbt_s[3]
    # -----------------------------------------------
    # <Jex.B> current response to fex_iz (A*T/m**2)
    # -----------------------------------------------
    tabtr[8, :] = p_jbex
    # -----------------------------------------
    #  Parallel electrical resistivity (Ohm*m)
    # -----------------------------------------
    tabtr[9, :] = p_etap
    core_transport.sigma[ipr] = 1.0 / p_etap
    # write(*,*) 'ipr=',ipr,' sigma=',1.0 / P_ETAP
    # ------------------------------------------------------------
    #  Particle and heat fluxes
    #       For each species a and charge state i
    #       gfl            : particle flux (rho/m**3/s)
    #       qfl            : conduction heat flux (J*rho/m**3/s)
    #       qfl+5.2*T*gfl  : total radial heat flux
    #       (1,a,i)        : p' and T' driven banana-plateau flux
    #       (2,a,i)        : Pfirsch-Schluter flux
    #       (3,a,i)        : classical flux
    #       (4,a,i)        : <E.B> driven flux
    #       (5,a,i)        : external force driven flux
    # ------------------------------------------------------------
    tabtr[10, :] = 0.0
    tabtr[11, :] = 0.0
    tabtr[12, :] = 0.0
    tabtr[13, :] = 0.0
    # -------------------------------
    #       Sum over flux components
    # -------------------------------
    # ------------------------------------------
    #         Electron conduction heat flow (w)
    # ------------------------------------------
    tabtr[10, :] = tabtr[10, :]+qfl_s(j, 1)
    tabtr[12, :] = tabtr[12, :]+gfl_s(j, 1)

    # -----------------------------------------
    #             Ion conduction heat flow (w)
    # -----------------------------------------
    tabtr[11, :] = tabtr[11, :]+qfl_s(j, ima)
    tabtr[13, :] = tabtr[13, :]+gfl_s(j, ima)

    # ITM ------------------------->
    core_transport.te_neo.flux[ipr] = tabtr[10, :] * rhomax / grho2_pr[ipr]
    #   core_transport%ti_neo%flux(ipr,1)  =   tabtr[11,:] * rhomax / grho2_pr[ipr]
    core_transport.ne_neo.flux[ipr] = tabtr[12, :] * rhomax / grho2_pr[ipr]
    #   core_transport%ni_neo%flux(ipr,1)  =   tabtr[13,:] * rhomax / grho2_pr[ipr]

    core_transport.ti_neo.flux(ipr, ima-1) = 0.0
    core_transport.ni_neo.flux(ipr, ima-1) = 0.0

    core_transport.ti_neo.flux(ipr, ima-1) = core_transport.ti_neo.flux(ipr, ima-1) + qfl_s(j, ima) * rhomax / grho2_pr[ipr]
    core_transport.ni_neo.flux(ipr, ima-1) = core_transport.ni_neo.flux(ipr, ima-1) + gfl_s(j, ima) * rhomax / grho2_pr[ipr]

    # ITM ------------------------->
    # --------------------------------------------------------------------
    #
    # tabtr[14)  : dif fusion coefficient (diag comp) of s (rho**2/s)
    # tabtr[15)  : somme des dif ferentes vitesses electroniques
    #              vns      -> convection velocity (off diag comps-p', T') of s (rho/s)
    #              vebs     -> <E.B> particle convection velocity of s (rho/s)
    #              qfl(5,1) -> external force driven flux
    # tabtr[16)  : heat electronic cond coefficient of s2 on p'/p of s1 (rho**2/s)
    #              + heat electronic cond coefficient of s2 on T'/T of s1 (rho**2/s)
    # tabtr[17)  : heat electronic convection velocity (rho/s)
    # tabtr[18)  : sum [heat ionic cond coefficient of s2 on p'/p of s1 (rho**2/s)
    #              + heat ionic cond coefficient of s2 on T'/T of s1 (rho**2/s)]
    # tabtr[19)  : sum heat ionic convection velocity (rho/s)
    # --------------------------------------------------------------------
    tabtr[14, :] = dn_s[1]
    # ITM ------------------------->
    core_transport.ne_neo.dif f_eff[ipr] = tabtr[14, :] * rhomax ** 2 / grho2_pr[ipr]
    # ITM ------------------------->
    tabtr[15, :] = vnnt_s[1]+vneb_s[1]+gfl_s[5, 1]/den[ipr, 1] + vnex_s[1]
    # ITM ------------------------->
    core_transport.ne_neo.vconv_eff[ipr] = - tabtr[15, :] * rhomax / grho2_pr[ipr]
    # ITM ------------------------->
    tabtr[16, :] = chit_ss[1, 1] + chip_ss[1, 1]
    # ITM ------------------------->
    core_transport.te_neo.dif f_eff[ipr] = tabtr[16, :] * rhomax**2 / grho2_pr[ipr] * den(ipr, 1)
    # [DPC:2010-08-21] I think the above multiplication by "den" is wrong
    core_transport.te_neo.dif f_eff[ipr] = tabtr[16, :] * rhomax**2 / grho2_pr[ipr]
    # ITM ------------------------->
    tabtr[17, :] = tabtr[10, :] / (den_iz[1, 1]*temp_i[1]*z_j7kv) + tabtr[16, :]*grt_i[1]/temp_i[1]
    # ITM ------------------------->
    core_transport.te_neo.vconv_eff[ipr] = - tabtr[17, :] * rhomax / grho2_pr[ipr]
    # ITM ------------------------->
    tabtr[18, :] = 0.0
    dentot = 0.0
    chitp = chit_ss[:, :]+chip_ss[:, :]
    tabtr[18, :] = tabtr[18, :] + chitp * den(ipr, ima)
    vconv = tabtr[11, :]/(den(ipr, ima)*temp_i(ima)*z_j7kv)+chitp*grt_i(2)/temp_i(2)
    # ITM ------------------------->
    core_transport.ti_neo.dif f_eff(ipr, ima-1) = chitp*rhomax**2.0/grho2_pr[ipr]
    core_transport.ti_neo.vconv_eff(ipr, ima-1) = - vconv * rhomax / grho2_pr[ipr]
    # ITM ------------------------->
    dentot = dentot+den(ipr, ima)

    tabtr[18, :] = tabtr[18, :]/dentot
    tabtr[19, :] = tabtr[11, :] / (dentot*temp_i(2)*z_j7kv) + tabtr[18, :]*grt_i(2)/temp_i(2)
    # donnees ioniques
    tabtr[20, :] = dentot
    tabtr[21, :] = temp_i[2]
    tabtr[22, :] = grt_i[2]
    # donnees electoniques
    tabtr[23, :] = temp_i[1]
    tabtr[24, :] = den_iz[1, 1]
    tabtr[25, :] = grt_i[1]
    # -----------------------------------------------
    # <J_OH.B> Ohmic current [A*T/m**2]
    # -----------------------------------------------
    tabtr[26, :] = p_jboh
    # -------------------------------------------------------|
    #  UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]|
    #                m=1, p', T', Phi'                      |
    #                m=2, <E.B>                             |
    #                m=3, fex_iz                            |
    # -------------------------------------------------------|
    indsave = 26
    tabtr[indsave:, :] = upar_s[1, 1, :] + upar_s[1, 2, :] + upar_s[1, 3, :]

    indsave = indsave + m_i
    # ---------------------------------------------------------|
    #  UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]|
    #                                     m=1, p', T'                                                                     |
    #                                     m=2, <E.B>                                                                      |
    #                                     m=3, fex_iz                                                                     |
    # ---------------------------------------------------------|

    tabtr[indsave+m, :] = utheta_s[1, 1, :] + utheta_s[1, 2, :] + utheta_s[1, 3, :]
    if (ipr == 1):
        #          write(*,*) 'm=',m,UTHETA_S(1,1,m),UTHETA_S(1,2,m)
        #          write(*,*) UTHETA_S(1,3,m)

    indsave = indsave + m_i
    #
    # coefficient de dif fusion par espece et etat de charge de la matiere
    #
    tabtr[indsave+m, :] = dn_s[:]
    core_transport.ni_neo.diff_eff(ipr, m-1) = dn_s[:] * rhomax ** 2 / grho2_pr[ipr]

    #
    # Vitesse totale de convection par espece et etat de charge
    #
    indsave = indsave + m_i - 1

    tabtr[indsave:, :] = vnnt_s[:]+vneb_s[:] + gfl_s[5, :]/den[:, :]+vnex_s[:]
    core_transport.ni_neo.vconv_eff[:, :] = - tabtr[indsave:, :] * rhomax / grho2_pr[ipr]

    #  -- lissage au centre --
    etatemp[2: 4] = 1.0/core_transport.conductivity_parallel[2:4]
    etatemp[1] = 0.0

    core_transport.conductivity_parallel = 1.0/etatemp[1]
    core_transport.j_bootstrap[0] = 0.0

    return core_transport

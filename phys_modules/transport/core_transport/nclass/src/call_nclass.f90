
SUBROUTINE call_nclass(nrho, nion, R0, B0, rho_bdry, rho_tor_norm, amu_i, &
                       Ts, Ns, dTs, dNs, &
                       psi, r_axis, vprime, F_dia, q, &
                       gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8, gm9, &
                       elongation, eparallel, trapped_fraction, &
                       jboot, sigma, te_flux, ti_flux, ne_flux, ni_flux, &
                       te_diff, ti_diff, ne_diff, ni_diff, te_vconv, ti_vconv, ne_vconv, ni_vconv)

    USE SPEC_KIND_MOD
    USE NCLASS_MOD
    IMPLICIT NONE
    !----------------------------------------------------------------------------------------------------
    ! This function is modfied from  ets.neo.neo_calc(equilibrium,coreprof,neoclassic,nrho,neq,nion)
    !----------------------------------------------------------------------------------------------------

    ! INPUT
    integer, intent(in) :: nrho, nion

    REAL(KIND=rspec), intent(in) :: R0, B0, rho_bdry
    REAL(KIND=rspec), intent(in), DIMENSION(nion + 1) ::  amu_i

    REAL(KIND=rspec), intent(in), DIMENSION(nrho, nion + 1) ::  Ts, dTs  ! Temerature in [ev]
    REAL(KIND=rspec), intent(in), DIMENSION(nrho, nion + 1) ::  Ns, dNs  ! Density in [m^-3]
    REAL(KIND=rspec), intent(in), DIMENSION(nrho):: rho_tor_norm, psi, vprime, r_axis, &
                                                    gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8, gm9, elongation, eparallel, F_dia, &
                                                    trapped_fraction, q

    ! OUTPUT

    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       jboot
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       sigma
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       te_flux
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ti_flux
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       ne_flux
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ni_flux
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       te_diff
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ti_diff
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       ne_diff
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ni_diff
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       te_vconv
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ti_vconv
    REAL(KIND=rspec), intent(out), DIMENSION(nrho) ::       ne_vconv
    REAL(KIND=rspec), intent(out), DIMENSION(nrho, nion) :: ni_vconv
    ! constants
    REAL(KIND=rspec), PARAMETER :: &
        z_coulomb = 1.6022e-19_rspec, & !Coulomb charge [coul]
        z_epsilon0 = 8.8542e-12_rspec, & !permittivity of free space [F/m]
        z_j7kv = 1.6022e-16_rspec, & !energy conversion factor [J/keV]
        z_pi = 3.141592654_rspec, & !pi [-]
        z_protonmass = 1.6726e-27_rspec, & !proton mass [kg]
        z_electronmass = 9.1095e-31_rspec, & !electron mass [kg]
        z_mu0 = 1.2566e-06_rspec              !Permeability of free space  : (Henry/meter)

    ! internal
    ! REAL(KIND=rspec)                                           :: tnp1

    !-----------------------------------------------------------------------
    ! arguments  required to call nclass

    integer, PARAMETER ::   mxmz = 100, mxms = 100

    !Declaration of input variables

    REAL(KIND=rspec) :: p_eb, p_fhat, p_fm(3), &
                        p_grphi, p_gr2phi, p_ngrth, &
                        grt_i(nion + 1), temp_i(nion + 1), vti(nion + 1), &
                        den_iz(nion + 1, mxmz), grp_iz(nion + 1, mxmz), fex_iz(3, nion + 1, mxmz)

    !Declaration of output variables
    INTEGER ::       iflag

    !Declaration of optional input variables
    LOGICAL ::     L_BANANA, L_PFIRSCH, L_POTATO

    INTEGER ::         kboot, kpotato, k_order, kpfirsch, kbanana

    REAL(KIND=rspec) ::      C_DEN, C_POTB, C_POTL

    !Declaration of optional output variables
    REAL(KIND=rspec) ::   P_ETAP, P_JBBS, P_JBEX, P_JBOH

    INTEGER ::   M_S, JM_S(mxms), JZ_S(mxms)

    REAL(KIND=rspec) ::       BSJBP_S(mxms), BSJBT_S(mxms)

    REAL(KIND=rspec) ::    DP_SS(mxms, mxms), DT_SS(mxms, mxms), &
                        GFL_S(5, mxms), DN_S(mxms), VNEB_S(mxms), VNEX_S(mxms), VNNT_S(mxms)

    REAL(KIND=rspec) ::   UPAR_S(3, 3, mxms), UTHETA_S(3, 3, mxms)

    REAL(KIND=rspec) ::     CHIP_SS(mxms, mxms), CHIT_SS(mxms, mxms), &
                        QFL_S(mxms, mxms), CHI_S(mxms), VQEB_S(mxms), VQEX_S(mxms), VQNT_S(mxms)

    REAL(KIND=rspec) ::    CAPM_II(3, 3, nion + 1, nion + 1), CAPN_II(3, 3, nion + 1, nion + 1), &
                        CALM_I(3, 3, nion + 1), CALN_II(3, 3, nion + 1, nion + 1)

    REAL(KIND=rspec) ::  YMU_S(3, 3, mxms)

    REAL(KIND=rspec) ::   SQZ_S(mxms), XI_S(mxms), TAU_SS(mxms, mxms)
    !-----------------------------------------------------------------------

    REAL(KIND=rspec) pcharge(nion + 1), pmasse(nion + 1)

    ! REAL(KIND=rspec) ni(200), x_r_axis, xeps

    REAL(KIND=rspec)  :: bt0, dencut, dent, xbeta,  xvpr, ab, xgph, eps2, b, a, c, xm, rminx
    REAL(KIND=rspec)  :: dentot, chitp, vconv, etatemp(4), xi(nion + 1, mxmz)

    integer ::m_i, m_z, nspec, ipr
    integer :: icharge, kmmaj, ksmaj
    REAL(KIND=rspec) q0, c_pot1, x_q, x_r_axis, xeps

    REAL(KIND=rspec) tabtr(1000, nrho)

    REAL(KIND=rspec) grad_rho2, grad_rho, &
        xfs_pr(nrho), bt0_pr(nrho), &
        psidrho(nrho), rminx_pr(nrho), &
        gph_pr(nrho), &
        p_eb_pr(nrho), &
        ergrho(nrho), inter(nrho), gr2phi(nrho), &
        force1(nrho), force2(nrho), force3(nrho), &
        xfs, rap(nrho), denz2(nion + 1)

    integer ::  m
    integer :: itp1, i, ia, ima, ki, kspec, iza, j, indsave

    itp1 = 1

    !----------------------------------------------------------------------
    ! Model options
    !  kboot           : internal option for metrics
    !                     = 1 -> Hirshman, Sigmar model: fm(k)=Lck^*,
    !                                      xngrth=<(n.gr(B))^2>/<B^2>.
    !                     = 2 -> Kessel model:           fm(1)=Rq/eps**3/2.
    !                                      xngrth not used.
    !                     =   -> Shaing, et al model: fm(k)=Fk,
    !                                      xngrth=<n.gr(theta)>
    !----------------------------------------------------------------------
    kboot = 0
    !    by default :     kboot = 0nMaxSpec
    !----------------------------------------------------
    !  kpotato       :  option to include potato orbits (-)
    !                   = 0 -> off
    !                   =   -> else on
    !-----------------------------------------------------
    kpotato = 1
    !     by default :     kpotato = 1
    !-------------------------------------------------------
    !  k_order        : order of v moments to be solved (-)
    !                   = 2 -> u and q
    !                   = 3 -> u, q, and u2
    !                   =   -> else error
    !-------------------------------------------------------
    k_order = 3
    !      by default :     k_order = 3
    !------------------------------------------------
    !  kbanana       : option to include banana viscosity (-)
    !                 = 0 -> off
    !                 =   -> else on
    !------------------------------------------------
    kbanana = 1
    !      by default :     kbanana = 1
    !------------------------------------------------------------------
    !  kpfirsch      : option to include Pfirsch-Schluter viscosity (-)
    !                 = 0 -> off
    !                 =   -> else on
    !------------------------------------------------------------------
    kpfirsch = 1

    !------------------------------------------------------------------

    nspec = nion + 1 ! add electron species
    m_i = nspec
    m_z = 0
    ! pmasse(1) = 1   ! electron
    ! pmasse(2:nspec) = coreprof(itp1)%composition%amn  ! species mass
    ! pcharge(1) = -1    ! electron
    ! pcharge(2:nspec) = coreprof(itp1)%composition%zn
    ! DO i = 1, nspec
    !     icharge = nint(pcharge(i))
    !     IF (icharge .gt. m_z) m_z = icharge
    ! END DO
    ! amu_i(1) = z_electronmass/z_protonmass
    ! DO i = 2, nspec
    !     amu_i(i) = pmasse(i)
    ! END DO
    !------------------------------------------------------------------------
    !  c_den          : density cutoff below which species is ignored (/m**3)
    !------------------------------------------------------------------------
    c_den = 1.0e10
    !-----------------------------------------------------------------------------
    !  p_grphi        : radial electric field Phi' (V/rho)
    !  p_gr2phi       : radial electric field gradient Psi'(Phi'/Psi')' (V/rho**2)
    !-----------------------------------------------------------------------------

    q0 = q(1)

    bt0_pr(1:nrho) = F_dia*gm1/gm9
    gph_pr(1:nrho) = gm1*rho_bdry*vprime(1:nrho)
    ! gph_pr(1:nrho) = gm1(1:nrho)*rho_bdry*vprime

    p_eb_pr = eparallel*bt0_pr
    c_potb = elongation(1)*B0/2/q0/q0
    c_potl = q0*x_r_axis

    ! inter(2:nrho) = -ergrho(2:nrho)/psidrho(2:nrho)
    ! gr2phi = psidrho*gr2phi
    !      not in the CPO
    !      force1          =    1st moment of external forces for the main ion (datak.source.total.q ?)
    !      force2          =    2nd moment of external forces (?)
    !      force3          =    3rd external force moment for the main ion (datak.source.totale.wb ?)
    force1 = 0.
    force2 = 0.
    force3 = 0.
    !
    !  kmmaj : mass of the NBI particle injected
    !  ksmaj : charge of the injected NBI particle
    !
    kmmaj = 2
    ksmaj = 2

    !--------------------------------------------------------------------|
    ! xfs     : external poloidal current has a flux surface (A)|
    !--------------------------------------------------------------------|
    xfs_pr = 2.0*z_pi*r_axis*bt0_pr/z_mu0
    !-----------------------
    ! Loop over radial nodes
    !-----------------------
    DO ipr = 1, nrho
        !
        ! external force set to zero
        !
        grad_rho2 = gm3(ipr)
        grad_rho = gm7(ipr)
        p_grphi = 0.0
        p_gr2phi = 0.0
        fex_iz(1, kmmaj, ksmaj) = force1(ipr)
        fex_iz(2, kmmaj, ksmaj) = force2(ipr)
        fex_iz(3, kmmaj, ksmaj) = force3(ipr)
        dencut = 1e10
        x_r_axis = r_axis(ipr)
        rminx = rminx_pr(ipr)
        x_q = q(ipr)
        if (ipr .eq. 4) then

            !  x_q = 0.6772616311415705

        end if
        bt0 = bt0_pr(ipr)
        xvpr = vprime(ipr)
        ab = 1.0

        q0 = q(1)
        xeps = rminx/x_r_axis
        xgph = gph_pr(ipr)/4/z_pi/z_pi
        xfs = xfs_pr(ipr)
        DO ima = 1, m_i
            DO ia = 1, m_z
                den_iz(ima, ia) = dencut
            END DO
        END DO
        DO ima = 2, m_i
            icharge = nint(pcharge(ima))
            den_iz(ima, icharge) = Ns(ipr, ima)
        END DO
        den_iz(1, 1) = 0.0
        dent = 0.0
        DO ima = 2, m_i
            denz2(ima) = 0.0
            DO ia = 1, m_z
                den_iz(1, 1) = den_iz(1, 1) + float(ia)*den_iz(ima, ia)
                denz2(ima) = denz2(ima) + den_iz(ima, ia)&
                            &                               *float(ia)**2
                dent = dent + den_iz(ima, ia)*Ts(ipr, ima)
            END DO
        END DO
        denz2(1) = den_iz(1, 1)
        dent = dent + den_iz(1, 1)*Ts(ipr, 1)
        xbeta = dent*z_j7kv/(bt0**2/2.0/z_mu0)
        DO ima = 1, m_i
            DO ia = 1, m_z
                xi(ima, ia) = den_iz(ima, ia)*float(ia)**2/denz2(ima)
                IF (den_iz(ima, ia) .eq. 0) THEN
                    grp_iz(ima, ia) = 0
                ELSE
                    grp_iz(ima, ia) = dTs(ipr, ima)*den_iz(ima, ia) + Ts(ipr, ima)*dNs(ipr, ima)
                END IF
            END DO
        END DO
        if (xgph .eq. 0) then
            xgph = 0.01
        end if
        p_fhat = x_q/xgph
        p_eb = p_eb_pr(ipr)
        c_potb = elongation(1)*bt0/2/q0/q0
        c_potl = q0*x_r_axis
        IF (kboot .eq. 1) THEN
            !-----------------
            ! Hirshman
            !-----------------
            p_ngrth = (xeps/(x_q*x_r_axis))**2/2.0
            p_fm(1) = x_q*x_r_axis
            p_fm(2) = 0.0
            p_fm(3) = 0.0
        ELSEIF (kboot .eq. 2) then
            !---------------
            !         Kessel
            !---------------
            p_ngrth = 0.0
            p_fm(1) = x_q*x_r_axis/xeps**1.5
        ELSE
            !---------------
            !         Shaing
            !---------------
            p_ngrth = 1.0/(x_q*x_r_axis)
            DO m = 1, 3
                p_fm(m) = 0
            END DO
            IF (xeps .gt. 0.0) THEN
                eps2 = xeps**2
                b = sqrt(1.0 - eps2)
                a = (1.0 - b)/xeps
                c = (b**3.0)*(x_q*x_r_axis)**2
                DO m = 1, 3
                    xm = float(m)
                    p_fm(m) = xm*a**(2.0*xm)*(1.0 + xm*b)/c
                END DO
            END IF
        END IF
        !-------------------------------------------
        ! Find significant charge states and mapping
        !-------------------------------------------
        m_s = 0
        DO ima = 1, m_i
            DO iza = 1, m_z
                IF (den_iz(ima, iza) .gt. c_den) THEN
                    m_s = m_s + 1
                    !---------------------------------------------------------------
                    ! Set isotope number and charge state for this species
                    !---------------------------------------------------------------
                    jm_s(m_s) = ima
                    IF (amu_i(ima) .lt. 0.5) THEN
                        jz_s(m_s) = -iza
                    ELSE
                        jz_s(m_s) = iza
                    END IF
                END IF
            END DO
        END DO
        !---------------------------
        ! Calculate thermal velocity
        !---------------------------
        DO ima = 1, m_i
            vti(ima) = sqrt(2.0*z_j7kv*Ts(ipr, ima)/amu_i(ima)/z_protonmass)
            temp_i(ima) = Ts(ipr, ima)
            grt_i(ima) = dTs(ipr, ima)
        END DO

        if (kbanana .ne. 0) then
            l_banana = .true.
        else
            l_banana = .false.
        end if
        if (kpotato .ne. 0) then
            l_potato = .true.
        else
            l_potato = .false.
        end if
        if (kpfirsch .ne. 0) then
            l_pfirsch = .true.
        else
            l_pfirsch = .false.
        end if

        CALL nclass( &!                 -------------------------------------------------------------------------
            !                           INPUT
            m_i, &!                         m_i                  -number of isotopes (> 1) [-]
            m_z, &!                         m_z                  -highest charge state [-]
            gm5(ipr), &!                    p_b2                 -<B**2> [T**2]
            gm4(ipr), &!                    p_bm2                -<1/B**2> [/T**2]
            p_eb, &!                        p_eb                 -<E.B> [V*T/m]
            p_fhat, &!                      p_fhat               -mu_0*F/(dPsi/dr) [rho/m]
            p_fm, &!                        p_fm(m)              -poloidal moments of drift factor for PS [/m**2]
            trapped_fraction(ipr), &!       p_ft                 -trapped fraction [-]
            gm6(ipr), &!                    p_grbm2              -<grad(rho)**2/B**2> [rho**2/m**2/T**2]
            p_grphi, &!                     p_grphi              -potential gradient Phi' [V/rho]
            p_gr2phi, &!                    p_gr2phi             -second potential gradient Psi'(Phi'/Psi')' [V/rho**2]
            p_ngrth, &!                     p_ngrth              -<n.grad(Theta)> [/m]
            amu_i, &!                       amu_i(i)             -atomic mass number [-]
            grt_i, &!                       grt_i(i)             -temperature gradient [keV/rho]
            temp_i, &!                      temp_i(i)            -temperature [keV]
            den_iz, &!                      den_iz(i,z)          -density [/m**3]
            fex_iz, &!                      fex_iz(3,i,z)        -moments of external parallel force [T*n/m**3]
            grp_iz, &!                      grp_iz(i,z)          -pressure gradient [keV/m**3/rho]
            ipr, &!
            !                           -------------------------------------------------------------------------
            !                           OUTPUT
            iflag, &!                       iflag               -error and warning flag [-]
            !                                                       =-1 warning
            !                                                       =0 no warnings or errors
            !                                                       =1 error
            !                           -------------------------------------------------------------------------
            !                           Optional input:
            l_banana, &!                    L_BANANA            -option to include banana viscosity [logical]
            l_pfirsch, &!                   L_PFIRSCH           -option to include Pfirsch-Schluter viscosity [logical]
            l_potato, &!                    L_POTATO            -option to include potato orbits [logical]
            k_order, &!                     K_ORDER             -order of v moments to be solved [-]
            !                                                     =2 u and q (default)
            !                                                     =3 u, q, and u2
            !                                                     =else error
            c_den, &!                       C_DEN-density cutoff below which species is ignored (default 1.e10) [/m**3]
            c_potb, &!                      C_POTB-kappa(0)*Bt(0)/[2*q(0)**2] [T]
            c_potl, &!                      C_POTL-q(0)*R(0) [m]
            !                           -------------------------------------------------------------------------
            !                           OUTPUT:
            !                           * Terms summed over species
            p_etap, &!                      P_ETAP-parallel electrical resistivity [Ohm*m]
            p_jbbs, &!                      P_JBBS-<J_bs.B> [A*T/m**2]
            p_jbex, &!                      P_JBEX-<J_ex.B> current response to fex_iz [A*T/m**2]
            p_jboh, &!                      P_JBOH-<J_OH.B> Ohmic current [A*T/m**2]
            !                           * Species mapping
            m_s, &!                         M_S                 -number of species [ms>1]
            jm_s, &!                        JM_S(s)             -isotope number of s [-]
            jz_s, &!                        JZ_S(s)             -charge state of s [-]
            !                           * Bootstrap current and electrical resistivity
            bsjbp_s, &!                     BSJBP_S(s)          -<J_bs.B> driven by unit p'/p of s [A*T*rho/m**2]
            bsjbt_s, &!                     BSJBT_S(s)          -<J_bs.B> driven by unit T'/T of s [A*T*rho/m**2]
            !                           * Continuity equation
            gfl_s, &!                       GFL_S(m,s)-radial particle flux comps of s [rho/m**3/s]
            !                                          m=1, banana-plateau, p' and T'
            !                                          m=2, Pfirsch-Schluter
            !                                          m=3, classical
            !                                          m=4, banana-plateau, <E.B>
            !                                          m=5, banana-plateau, external parallel force fex_iz
            dn_s, &!                        DN_S(s)             -diffusion coefficients (diag comp) [rho**2/s]
            vnnt_s, &!                      VNNT_S(s)           -convection velocity (off diag p',T' comps) [rho/s]
            vneb_s, &!                      VNEB_S(s)           -<E.B> particle convection velocity [rho/s]
            vnex_s, &!                      VNEX_S(s)           -external force particle convection velocity [rho/s]
            dp_ss, &!                       DP_SS(s1,s2)-diffusion coefficient of s2 on p'/p of s1 [rho**2/s]
            dt_ss, &!                       DT_SS(s1,s2)-diffusion coefficient of s2 on T'/T of s1 [rho**2/s]
            !                            * Momentum equation
            upar_s, &!                      UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]
            !                                             m=1, p', T', Phi'
            !                                             m=2, <E.B>
            !                                             m=3, fex_iz
            utheta_s, &!                    UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]
            !                                               m=1, p', T'
            !                                               m=2, <E.B>
            !                                               m=3, fex_iz
            !                            * Energy equation
            qfl_s, &!                       QFL_S(m,s)-radial heat conduction flux comps of s [W*rho/m**3]
            !                                           m=1, banana-plateau, p' and T'
            !                                           m=2, Pfirsch-Schluter
            !                                           m=3, classical
            !                                           m=4, banana-plateau, <E.B>
            !                                           m=5, banana-plateau, external parallel force fex_iz
            chi_s, &!                       CHI_S(s)            -conduction coefficients (diag comp) [rho**2/s]
            vqnt_s, &!                      VQNT_S(s)           -conduction velocity (off diag p',T' comps) [rho/s]
            vqeb_s, &!                      VQEB_S(s)           -<E.B> heat convection velocity [rho/s]
            vqex_s, &!                      VQEX_S(s)           -external force heat convection velocity [rho/s]
            chip_ss, &!                     CHIP_SS(s1,s2)-heat cond coefficient of s2 on p'/p of s1 [rho**2/s]
            chit_ss, &!                     CHIT_SS(s1,s2)-heat cond coefficient of s2 on T'/T of s1 [rho**2/s]
            !                            * Friction coefficients
            calm_i, &!                      CAPM_II(K_ORDER,K_ORDER,m_i,m_i)-test particle (tp) friction matrix [-]
            caln_ii, &!                     CAPN_II(K_ORDER,K_ORDER,m_i,m_i)-field particle (fp) friction matrix [-]
            capm_ii, &!                     CALM_I(K_ORDER,K_ORDER,m_i)-tp eff friction matrix [kg/m**3/s]
            capn_ii, &!                     CALN_II(K_ORDER,K_ORDER,m_i,m_i)-fp eff friction matrix [kg/m**3/s]
            !                            * Viscosity coefficients
            ymu_s, &!                       YMU_S(s)-normalized viscosity for s [kg/m**3/s]
            !                            * Miscellaneous
            sqz_s, &!                       SQZ_S(s)            -orbit squeezing factor for s [-]
            xi_s, &!                        XI_S(s)             -charge weighted density factor of s [-]
            tau_ss &!                       TAU_SS(s1,s2)       -90 degree scattering time [s]
            )

        ! OUTPUT
        !
        !----------------------
        !   <Jbs.B> (A*T/m**2)
        !----------------------
        tabtr(1, ipr) = p_jbbs
        jboot(ipr) = p_jbbs/B0
        !------------------------------------------------------------------
        ! <Jbs.B> driven by unit p'/p of species s (A*T*rho/m**3)
        ! s  : 1 -> electrons, 2 -> espece ionique 1, 3 -> espece ionique 2
        !------------------------------------------------------------------
        tabtr(2, ipr) = bsjbp_s(1)
        tabtr(3, ipr) = bsjbp_s(2)
        tabtr(4, ipr) = bsjbp_s(3)
        !------------------------------------------------------------------
        ! <Jbs.B> driven by unit T'/T of s (A*T*rho/m**3)
        ! s  : 1 -> electrons, 2 -> espece ionique 1, 3 -> espece ionique 2
        !------------------------------------------------------------------
        tabtr(5, ipr) = bsjbt_s(1)
        tabtr(6, ipr) = bsjbt_s(2)
        tabtr(7, ipr) = bsjbt_s(3)
        !-----------------------------------------------
        ! <Jex.B> current response to fex_iz (A*T/m**2)
        !-----------------------------------------------
        tabtr(8, ipr) = p_jbex
        !-----------------------------------------
        !  Parallel electrical resistivity (Ohm*m)
        !-----------------------------------------
        tabtr(9, ipr) = p_etap
        sigma(ipr) = 1.0/p_etap
        !------------------------------------------------------------
        !  Particle and heat fluxes
        !       For each species a and charge state i
        !       gfl            : particle flux (rho/m**3/s)
        !       qfl            : conduction heat flux (J*rho/m**3/s)
        !       qfl+5.2*T*gfl  : total radial heat flux
        !       (1,a,i)        : p' and T' driven banana-plateau flux
        !       (2,a,i)        : Pfirsch-Schluter flux
        !       (3,a,i)        : classical flux
        !       (4,a,i)        : <E.B> driven flux
        !       (5,a,i)        : external force driven flux
        !------------------------------------------------------------
        tabtr(10, ipr) = 0.0d0
        tabtr(11, ipr) = 0.0d0
        tabtr(12, ipr) = 0.0d0
        tabtr(13, ipr) = 0.0d0
        !-------------------------------
        ! Sum over flux components
        !-------------------------------
        DO j = 1, 5
            !------------------------------------------
            ! Electron conduction heat flow (w)
            !------------------------------------------
            tabtr(10, ipr) = tabtr(10, ipr) + qfl_s(j, 1)
            tabtr(12, ipr) = tabtr(12, ipr) + gfl_s(j, 1)
            DO ima = 2, m_i
                !-----------------------------------------
                ! Ion conduction heat flow (w)
                !-----------------------------------------
                tabtr(11, ipr) = tabtr(11, ipr) + qfl_s(j, ima)
                tabtr(13, ipr) = tabtr(13, ipr) + gfl_s(j, ima)
            END DO
        END DO
        te_flux(ipr) = tabtr(10, ipr)*rho_bdry/grad_rho2
        !   ti_neo_flux(ipr,1)  =   tabtr(11,ipr) * rho_bdry / grad_rho2
        ne_flux(ipr) = tabtr(12, ipr)*rho_bdry/grad_rho2
        !   ni_flux(ipr,1)  =   tabtr(13,ipr) * rho_bdry / grad_rho2
        do ima = 2, m_i
            ti_flux(ipr, ima - 1) = 0.0d0
            ni_flux(ipr, ima - 1) = 0.0d0
            do j = 1, 5
                ti_flux(ipr, ima - 1) = &
                  &  ti_flux(ipr, ima - 1) + qfl_s(j, ima)*rho_bdry/grad_rho2
                ni_flux(ipr, ima - 1) = &
                  &  ni_flux(ipr, ima - 1) + gfl_s(j, ima)*rho_bdry/grad_rho2
            end do
        end do
        !--------------------------------------------------------------------
        !
        ! tabtr(14)  : diffusion coefficient (diag comp) of s (rho**2/s)
        ! tabtr(15)  : somme des differentes vitesses electroniques
        !              vns      -> convection velocity (off diag comps-p', T') of s (rho/s)
        !              vebs     -> <E.B> particle convection velocity of s (rho/s)
        !              qfl(5,1) -> external force driven flux
        ! tabtr(16)  : heat electronic cond coefficient of s2 on p'/p of s1 (rho**2/s)
        !              + heat electronic cond coefficient of s2 on T'/T of s1 (rho**2/s)
        ! tabtr(17)  : heat electronic convection velocity (rho/s)
        ! tabtr(18)  : sum [heat ionic cond coefficient of s2 on p'/p of s1 (rho**2/s)
        !              + heat ionic cond coefficient of s2 on T'/T of s1 (rho**2/s)]
        ! tabtr(19)  : sum heat ionic convection velocity (rho/s)
        !--------------------------------------------------------------------
        tabtr(14, ipr) = dn_s(1)
        ne_diff(ipr) = tabtr(14, ipr)*rho_bdry**2/grad_rho2
        tabtr(15, ipr) = vnnt_s(1) + vneb_s(1) + gfl_s(5, 1)/Ns(ipr, 1) + vnex_s(1)
        ne_vconv(ipr) = -tabtr(15, ipr)*rho_bdry/grad_rho2
        tabtr(16, ipr) = chit_ss(1, 1) + chip_ss(1, 1)
        ! te_diff(ipr) = tabtr(16, ipr)*rho_bdry**2/grad_rho2*Ns(ipr, 1)
        te_diff(ipr) = tabtr(16, ipr)*rho_bdry**2/grad_rho2
        tabtr(17, ipr) = tabtr(10, ipr)/(den_iz(1, 1)*temp_i(1)*z_j7kv) + tabtr(16, ipr)*grt_i(1)/temp_i(1)
        te_vconv(ipr) = -tabtr(17, ipr)*rho_bdry/grad_rho2
        tabtr(18, ipr) = 0.0
        dentot = 0.0
        DO ima = 2, m_i
            chitp = chit_ss(ima, ima) + chip_ss(ima, ima)
            tabtr(18, ipr) = tabtr(18, ipr) + chitp*Ns(ipr, ima)
            vconv = tabtr(11, ipr)/(Ns(ipr, ima)*temp_i(ima)*z_j7kv) + chitp*grt_i(2)/temp_i(2)
            ti_diff(ipr, ima - 1) = chitp*rho_bdry**2.0/grad_rho2
            ti_vconv(ipr, ima - 1) = -vconv*rho_bdry/grad_rho2
            dentot = dentot + Ns(ipr, ima)
        END DO
        tabtr(18, ipr) = tabtr(18, ipr)/dentot
        tabtr(19, ipr) = tabtr(11, ipr)/(dentot*temp_i(2)*z_j7kv) + tabtr(18, ipr)*grt_i(2)/temp_i(2)
        ! ION DATA
        tabtr(20, ipr) = dentot
        tabtr(21, ipr) = temp_i(2)
        tabtr(22, ipr) = grt_i(2)
        ! Electron DATA
        tabtr(23, ipr) = temp_i(1)
        tabtr(24, ipr) = den_iz(1, 1)
        tabtr(25, ipr) = grt_i(1)
        !-----------------------------------------------
        ! <J_OH.B> Ohmic current [A*T/m**2]
        !-----------------------------------------------
        tabtr(26, ipr) = p_jboh
        !-------------------------------------------------------|
        !  UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]|
        !                m=1, p', T', Phi'                      |
        !                m=2, <E.B>                             |
        !                m=3, fex_iz                            |
        !-------------------------------------------------------|
        indsave = 26
        DO m = 1, m_i
            tabtr(indsave + m, ipr) = upar_s(1, 1, m) + upar_s(1, 2, m) + upar_s(1, 3, m)
        END DO
        indsave = indsave + m_i
        !---------------------------------------------------------|
        !  UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]|
        !                                     m=1, p', T'                                                                     |
        !                                     m=2, <E.B>                                                                      |
        !                                     m=3, fex_iz                                                                     |
        !---------------------------------------------------------|

        DO m = 1, m_i
            tabtr(indsave + m, ipr) = utheta_s(1, 1, m) + utheta_s(1, 2, m) + utheta_s(1, 3, m)
        END DO
        indsave = indsave + m_i
        !
        ! diffusion coefficient by species and charge state of the material
        !
        DO m = 2, m_i
            tabtr(indsave + m, ipr) = dn_s(m)
            ni_diff(ipr, m - 1) = dn_s(m)*rho_bdry**2/grad_rho2
        END DO
        !
        ! Total convection velocity per species and load state
        !
        indsave = indsave + m_i - 1
        DO m = 2, m_i
            tabtr(indsave + m, ipr) = vnnt_s(m) + vneb_s(m) + gfl_s(5, m)/Ns(ipr, m) + vnex_s(m)
            ni_vconv(ipr, m - 1) = -tabtr(indsave + m, ipr)*rho_bdry/grad_rho2
        END DO

    END DO !    DO ipr = 1, nrho

END SUBROUTINE call_nclass

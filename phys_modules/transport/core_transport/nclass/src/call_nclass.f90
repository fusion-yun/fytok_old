
SUBROUTINE call_nclass(nrho, nion, R0, B0, rho_bdry, rho_tor_norm, amu_i, &
                       Ts, Ns, dTs, dNs, &
                       psi, r_in, r_out, vprime, F_dia, &
                       gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8, gm9, &
                       elongation, eparallel, &
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

    REAL(KIND=rspec), intent(in), DIMENSION(nrho, nion + 1) ::  Ts, dTs  ! Temerature in [Kev]
    REAL(KIND=rspec), intent(in), DIMENSION(nrho, nion + 1) ::  Ns, dNs  ! Temerature in [Kev]
    REAL(KIND=rspec), intent(in), DIMENSION(nrho):: rho_tor_norm, r_in, r_out, psi, vprime,&
    & gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8, gm9, elongation, eparallel, F_dia

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

    REAL(KIND=rspec) :: p_b2, p_bm2, p_eb, p_fhat, p_fm(3), p_ft, p_grbm2, p_grphi, p_gr2phi, p_ngrth, &
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

    ! REAL(KIND=rspec) ni(200), xr0, xeps

    REAL(KIND=rspec)  :: bt0, dencut, dent, xbeta, xdelp, xvpr, ab, xgph, eps2, b, a, c, xm, rminx
    REAL(KIND=rspec)  :: dentot, chitp, vconv, etatemp(4), xi(nion + 1, mxmz)

    integer ::m_i, m_z, nspec, ipr
    integer :: icharge, kmmaj, ksmaj
    REAL(KIND=rspec) rhomax, q0, rkappa0, c_pot1, xqs, xr0, xeps

    REAL(KIND=rspec) tabtr(1000, nrho)

    REAL(KIND=rspec) grad_rho2(nrho), xfs_pr(nrho), bt0_pr(nrho), xr0_pr(nrho), &
        psidrho(nrho), rminx_pr(nrho), p_ft_pr(nrho), xqs_pr(nrho), &
        xvpr_pr(nrho), gph_pr(nrho), grho2_pr(nrho), dpsidrho(nrho), &
        p_b2_pr(nrho), p_bm2_pr(nrho), grho_pr(nrho), p_eb_pr(nrho), &
        ergrho(nrho), inter(nrho), gr2phi(nrho), &
        force1(nrho), force2(nrho), force3(nrho), p_grbm2_pr(nrho), &
        xfs, rap(nrho), denz2(nion + 1)

    integer ::  m
    integer :: itp1, i, ia, ima, ki, kspec, iza, j, indsave

    itp1 = 1
    !____________
    ! data access
    !____________
    nspec = nion + 1 ! add electron species
    ! pmasse(1) = 1   ! electron
    ! pmasse(2:nspec) = coreprof(itp1)%composition%amn  ! species mass
    ! pcharge(1) = -1    ! electron
    ! pcharge(2:nspec) = coreprof(itp1)%composition%zn
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
    ! kpfirsch = 1
    ! m_i = nspec
    ! m_z = 0
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
    ! !-----------------------------------------------------------------------------
    ! !  p_grphi        : radial electric field Phi' (V/rho)
    ! !  p_gr2phi       : radial electric field gradient Psi'(Phi'/Psi')' (V/rho**2)
    ! !-----------------------------------------------------------------------------

    ! x(1:nrho) = rho/rho_bdry
    ! !----------------------
    ! ! temprature in keV
    ! !----------------------
    ! ! Ts(1:nrho, 1) = coreprof(itp1)%te%value(1:nrho)/1000.0
    ! ! do kspec = 2, nspec
    ! !     Ts(1:nrho, kspec) = coreprof(itp1)%ti%value(:, kspec - 1)/1000.0
    ! ! end do
    ! ! Ns(1:nrho, 1) = coreprof(itp1)%ne%value
    ! ! ni = 0
    ! ! do kspec = 2, nspec
    ! !     Ns(1:nrho, kspec) = coreprof(itp1)%ni%value(:, kspec - 1)
    ! !     ni(:) = ni(:) + Ns(:, kspec)
    ! ! end do
    ! ! do kspec = 1, nspec
    ! !     !print*,"calcul de dTi/dx, Ti= ",tempi(101,kspec)
    ! !     !print*,"kspec=",kspec
    ! !     call cos_rpdederive(dTs(:, kspec), nrho, x, Ts(:, kspec), 0, 2, 2, 1)
    ! ! end do
    ! ! !      den    : coreprof(itp1).ne.value
    ! ! do kspec = 1, nspec
    ! !     !print*,"calcul de dni/dx, ni= ",den(1:4,kspec)
    ! !     !print*,"kspec=",kspec
    ! !     call cos_rpdederive(dNs(:, kspec), nrho, x, Ns(:, kspec), 0, 2, 2, 1)
    ! ! end do
    ! !     equilibrium data
    ! !!!psi(1:nrho)                    = coreprof(itp1)%psi%value
    ! ! dpc
    ! !!!psi(1:nrho)                    = equilibrium(1)%profiles_1d%psi
    ! !psi(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%psi, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%psi, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               psi, rho, nrho)
    ! ! cpd
    ! ! call cos_rpdederive(psidrho, nrho, rho, psi, 0, 2, 2, 1)
    ! !!!r_in(1:nrho)                           = equilibrium(1)%profiles_1d%r_inboard
    ! !r_in(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%r_inboard, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%r_inboard, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               r_in, rho, nrho)
    ! !!!r_out(1:nrho)                          = equilibrium(1)%profiles_1d%r_outboard
    ! !r_out(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%r_outboard, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%r_outboard, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               r_out, rho, nrho)
    ! xr0_pr = (r_out + r_in)/2
    ! xr0_pr = r_out               !!! ??? DPC

    ! !rminx_pr(1:nrho)                       = equilibrium%profiles_1d%rho_rttorfl
    ! rminx_pr(1:nrho) = rho(1:nrho)
    ! !!!p_ft_pr(1:nrho)                        = equilibrium(1)%profiles_1d%ftrap
    ! !p_ft_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%ftrap, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%ftrap, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               p_ft_pr, rho, nrho)
    ! !xqs_pr(1:nrho)                         = coreprof%profiles1d%q%value
    ! !!!xqs_pr(1:nrho)                         = equilibrium(1)%profiles_1d%q
    ! !xqs_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%q, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%q, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               xqs_pr, rho, nrho)
    ! q0 = xqs_pr(1)
    ! !!!DPC-EQ-4.08b-problem
    ! !!!xvpr_pr(1:nrho)                        = equilibrium(1)%profiles_1d%vprime
    ! !dpsidrho(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%dpsidrho_tor, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%dpsidrho_tor, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor),  &
    ! !               dpsidrho, rho, nrho)
    ! !xvpr_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%vprime, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%vprime, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               xvpr_pr, rho, nrho)
    ! xvpr_pr = xvpr_pr*dpsidrho*rho(nrho)

    ! !!!gm1(1:nrho)                            = equilibrium(1)%profiles_1d%gm1
    ! !gm1(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm1, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm1, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               gm1, rho, nrho)
    ! !!!gm9(1:nrho)                            = equilibrium(1)%profiles_1d%gm9
    ! !gm9(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm9, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm9, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !   gm9, rho, nrho)
    ! !!!bt0_pr(1:nrho)                         = equilibrium(1)%profiles_1d%F_dia * gm1 / gm9
    ! !bt0_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%F_dia, rho(1:nrho)) * gm1 / gm9
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%F_dia * gm1 / gm9, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor),  &
    ! !               bt0_pr, rho, nrho)
    ! !!!gph_pr(1:nrho)                         = equilibrium(1)%profiles_1d%gm1 * rho_bdry * xvpr_pr(1:nrho)
    ! gph_pr(1:nrho) = gm1(1:nrho)*rho_bdry*xvpr_pr
    ! !!!grad_rho2 (1:nrho)                      = equilibrium(1)%profiles_1d%gm3
    ! !grad_rho2(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm3, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm3, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               grad_rho2, rho, nrho)
    ! !!!p_b2_pr(1:nrho)                        = equilibrium(1)%profiles_1d%gm5
    ! !p_b2_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm5, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm5, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               p_b2_pr, rho, nrho)
    ! !!!p_bm2_pr(1:nrho)                       = equilibrium(1)%profiles_1d%gm4
    ! !p_bm2_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm4, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm4, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               p_bm2_pr, rho, nrho)
    ! !!!p_grbm2_pr(1:nrho)                     = equilibrium(1)%profiles_1d%gm6
    ! !p_grbm2_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm6, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm6, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               p_grbm2_pr, rho, nrho)
    ! !!!grho_pr(1:nrho)                        = equilibrium(1)%profiles_1d%gm7
    ! !grho_pr(1:nrho) = interpolate(equilibrium(1)%profiles_1d%rho_tor, equilibrium(1)%profiles_1d%gm7, rho(1:nrho))
    ! ! CALL l3interp(equilibrium(1)%profiles_1d%gm7, equilibrium(1)%profiles_1d%rho_tor, size(equilibrium(1)%profiles_1d%rho_tor), &
    ! !               grho_pr, rho, nrho)
    ! rkappa0 = elongation

    ! p_eb_pr(1:nrho) = coreprof(itp1)%profiles1d%eparallel%value
    ! !p_eb_pr                        = p_eb_pr * bt0_pr
    ! p_eb_pr = p_eb_pr*B0
    ! c_potb = rkappa0*bt0/2/q0/q0
    ! c_potl = q0*xr0

    ! !attention a prendre de l'equilibre
    ! !ergrho(1:nrho)                         = er / grho_pr
    ! ergrho(1:nrho) = 0

    ! inter(2:nrho) = -ergrho(2:nrho)/psidrho(2:nrho)
    ! call cos_zconversion(inter, nrho)
    ! !print*,"calcul de dinter/dx, inter= ",inter(1:4)
    ! call cos_rpdederive(gr2phi, nrho, rho, inter, 0, 2, 2, 1)
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
    xfs_pr = 2.0*z_pi*xr0_pr*bt0_pr/z_mu0
    !-----------------------
    ! Loop over radial nodes
    !-----------------------
    DO ipr = 1, nrho
        !
        ! external force set to zero
        !
        p_grphi = 0.0
        p_gr2phi = 0.0
        fex_iz(1, kmmaj, ksmaj) = force1(ipr)
        fex_iz(2, kmmaj, ksmaj) = force2(ipr)
        fex_iz(3, kmmaj, ksmaj) = force3(ipr)
        dencut = 1e10
        xr0 = xr0_pr(ipr)
        rminx = rminx_pr(ipr)
        xqs = xqs_pr(ipr)
        if (ipr .eq. 4) then

            !  xqs = 0.6772616311415705

        end if
        p_ft = p_ft_pr(ipr)
        bt0 = bt0_pr(ipr)
        xdelp = xvpr_pr(ipr)
        xvpr = xdelp
        ab = 1.0
        p_b2 = p_b2_pr(ipr)
        p_bm2 = p_bm2_pr(ipr)
        q0 = xqs_pr(1)
        xeps = rminx/xr0
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
        p_fhat = xqs/xgph
        p_grbm2 = p_grbm2_pr(ipr)
        p_eb = p_eb_pr(ipr)
        c_potb = rkappa0*bt0/2/q0/q0
        c_potl = q0*xr0
        IF (kboot .eq. 1) THEN
            !-----------------
            !         Hirshman
            !-----------------
            p_ngrth = (xeps/(xqs*xr0))**2/2.0
            p_fm(1) = xqs*xr0
            p_fm(2) = 0.0
            p_fm(3) = 0.0
        ELSEIF (kboot .eq. 2) then
            !---------------
            !         Kessel
            !---------------
            p_ngrth = 0.0
            p_fm(1) = xqs*xr0/xeps**1.5
        ELSE
            !---------------
            !         Shaing
            !---------------
            p_ngrth = 1.0/(xqs*xr0)
            DO m = 1, 3
                p_fm(m) = 0
            END DO
            IF (xeps .gt. 0.0) THEN
                eps2 = xeps**2
                b = sqrt(1.0 - eps2)
                a = (1.0 - b)/xeps
                c = (b**3.0)*(xqs*xr0)**2
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

        CALL nclass(m_i, m_z, p_b2, p_bm2, p_eb,     &
             &              p_fhat, p_fm, p_ft, p_grbm2, p_grphi,  &
             &              p_gr2phi, p_ngrth, amu_i, grt_i, temp_i,   &
             &              den_iz, fex_iz, grp_iz, ipr, iflag,    &
             &              l_banana, l_pfirsch, l_potato, k_order, c_den,    &
             &              c_potb, c_potl, p_etap, p_jbbs, p_jbex,   &
             &              p_jboh, m_s, jm_s, jz_s, bsjbp_s,  &
             &              bsjbt_s, gfl_s, dn_s, vnnt_s, vneb_s,   &
             &              vnex_s, dp_ss, dt_ss, upar_s, utheta_s, &
             &              qfl_s, chi_s, vqnt_s, vqeb_s, vqex_s,   &
             &              chip_ss, chit_ss, calm_i, caln_ii, capm_ii,  &
             &              capn_ii, ymu_s, sqz_s, xi_s, tau_ss)

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
        te_flux(ipr) = tabtr(10, ipr)*rho_bdry/grad_rho2(ipr)
        !   ti_neo_flux(ipr,1)  =   tabtr(11,ipr) * rho_bdry / grad_rho2(ipr)
        ne_flux(ipr) = tabtr(12, ipr)*rho_bdry/grad_rho2(ipr)
        !   ni_flux(ipr,1)  =   tabtr(13,ipr) * rho_bdry / grad_rho2(ipr)
        do ima = 2, m_i
            ti_flux(ipr, ima - 1) = 0.0d0
            ni_flux(ipr, ima - 1) = 0.0d0
            do j = 1, 5
                ti_flux(ipr, ima - 1) = &
                  &  ti_flux(ipr, ima - 1) + qfl_s(j, ima)*rho_bdry/grad_rho2(ipr)
                ni_flux(ipr, ima - 1) = &
                  &  ni_flux(ipr, ima - 1) + gfl_s(j, ima)*rho_bdry/grad_rho2(ipr)
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
        ne_diff(ipr) = tabtr(14, ipr)*rho_bdry**2/grad_rho2(ipr)
        tabtr(15, ipr) = vnnt_s(1) + vneb_s(1) + gfl_s(5, 1)/Ns(ipr, 1) + vnex_s(1)
        ne_vconv(ipr) = -tabtr(15, ipr)*rho_bdry/grad_rho2(ipr)
        tabtr(16, ipr) = chit_ss(1, 1) + chip_ss(1, 1)
        ! te_diff(ipr) = tabtr(16, ipr)*rho_bdry**2/grad_rho2(ipr)*Ns(ipr, 1)
        te_diff(ipr) = tabtr(16, ipr)*rho_bdry**2/grad_rho2(ipr)
        tabtr(17, ipr) = tabtr(10, ipr)/(den_iz(1, 1)*temp_i(1)*z_j7kv) + tabtr(16, ipr)*grt_i(1)/temp_i(1)
        te_vconv(ipr) = -tabtr(17, ipr)*rho_bdry/grad_rho2(ipr)
        tabtr(18, ipr) = 0.0
        dentot = 0.0
        DO ima = 2, m_i
            chitp = chit_ss(ima, ima) + chip_ss(ima, ima)
            tabtr(18, ipr) = tabtr(18, ipr) + chitp*Ns(ipr, ima)
            vconv = tabtr(11, ipr)/(Ns(ipr, ima)*temp_i(ima)*z_j7kv) + chitp*grt_i(2)/temp_i(2)
            ti_diff(ipr, ima - 1) = chitp*rho_bdry**2.0/grad_rho2(ipr)
            ti_vconv(ipr, ima - 1) = -vconv*rho_bdry/grad_rho2(ipr)
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
            ni_diff(ipr, m - 1) = dn_s(m)*rho_bdry**2/grad_rho2(ipr)
        END DO
        !
        ! Total convection velocity per species and load state
        !
        indsave = indsave + m_i - 1
        DO m = 2, m_i
            tabtr(indsave + m, ipr) = vnnt_s(m) + vneb_s(m) + gfl_s(5, m)/Ns(ipr, m) + vnex_s(m)
            ni_vconv(ipr, m - 1) = -tabtr(indsave + m, ipr)*rho_bdry/grad_rho2(ipr)
        END DO

    END DO !    DO ipr = 1, nrho

END SUBROUTINE call_nclass

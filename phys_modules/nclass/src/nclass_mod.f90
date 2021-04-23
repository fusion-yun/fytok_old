MODULE SPEC_KIND_MOD
    !-------------------------------------------------------------------------------
    !For REAL variables:
    !  Use p=6, r=35   for single precision on 32 bit machine
    !  Use p=12,r=100  for double precision on 32 bit machine
    !                  for single precision on 64 bit machine
    !Parameters for SELECTED_REAL_KIND:
    !  p                   -number of digits
    !  r                   -range of exponent from -r to r
    !-------------------------------------------------------------------------------
    INTEGER, PARAMETER :: &
        rspec = SELECTED_REAL_KIND(p=12, r=100), &
        ispec = SELECTED_INT_KIND(r=8)

END MODULE SPEC_KIND_MOD

MODULE NCLASS_MOD
    !-------------------------------------------------------------------------------
    !NCLASS-Calculates NeoCLASSical transport properties
    !
    !NCLASS_MOD is an F90 module to calculate neoclassical transport properties in
    !  for an axisymmetric toroidal plasma
    !
    !References:
    !
    !  W.A.Houlberg 1/2002
    !
    !Contains PUBLIC routines:
    !
    !    NCLASS            -neoclassical properties on a single surface
    !
    !Contains PRIVATE routines:
    !
    !    NCLASS_FLOW       -flow velocities within a surface
    !                      -called from NCLASS
    !    NCLASS_INIT       -initialize arrays
    !                      -called from NCLASS
    !    NCLASS_K          -velocity-dependent viscositite
    !                      -called from NCLASS_MU
    !    NCLASS_MN         -friction coefficients
    !                      -called from NCLASS
    !    NCLASS_MU         -fluid viscosities
    !                      -called from NCLASS
    !    NCLASS_NU         -collision frequencies
    !                      -called from NCLASS_K
    !    NCLASS_TAU        -collision times
    !                      -called from NCLASS
    !    NCLASS_BACKSUB    -LU matrix back substitution
    !                      -called from NCLASS_FLOW
    !    NCLASS_DECOMP     -LU matrix decomposition
    !                      -called from NCLASS_FLOW
    !    NCLASS_ERF        -error function
    !                      -called from NCLASS_NU
    !
    !Comments: cvs
    !
    !-------------------------------------------------------------------------------
    USE SPEC_KIND_MOD
    IMPLICIT NONE

    !-------------------------------------------------------------------------------
    ! Private procedures
    !-------------------------------------------------------------------------------
    PRIVATE ::                                                               &
   & NCLASS_FLOW,                                                            &
   & NCLASS_INIT,                                                            &
   & NCLASS_K,                                                               &
   & NCLASS_MN,                                                              &
   & NCLASS_MU,                                                              &
   & NCLASS_NU,                                                              &
   & NCLASS_TAU,                                                             &
   & NCLASS_BACKSUB,                                                         &
   & NCLASS_DECOMP,                                                          &
   & NCLASS_ERF,                                                             &
   & NORMALISATION

    !-------------------------------------------------------------------------------
    ! Private data
    !-------------------------------------------------------------------------------
    !Logical switches
    LOGICAL ::                                                               &
   & l_banana_nc, l_pfirsch_nc, l_potato_nc

    !Options
    INTEGER, PRIVATE ::                                                      &
   & k_order_nc

    !Constants
    REAL, PRIVATE, SAVE ::                                                   &
   & cden_nc, cpotb_nc, cpotl_nc, errsum

    !Terms summed over species
    REAL, PRIVATE, SAVE ::                                                   &
   & petap_nc, pjbbs_nc, pjbex_nc, pjboh_nc

    !Dimensions
    INTEGER, PRIVATE, SAVE ::                                                &
   & mf_nc, mi_nc, ms_nc, mz_nc

    !Electron isotope identification
    INTEGER, PRIVATE, SAVE ::                                                &
   & imel_nc

    !Isotope arrays
    REAL(KIND=rspec), PRIVATE, SAVE, ALLOCATABLE ::                          &
   & vti_nc(:), amntii_nc(:, :),                                               &
   & calmi_nc(:, :, :), calnii_nc(:, :, :, :),                                     &
   & capmii_nc(:, :, :, :), capnii_nc(:, :, :, :)

    !Isotope and charge arrays
    REAL(KIND=rspec), PRIVATE, SAVE, ALLOCATABLE ::                          &
   & grppiz_nc(:, :)

    !Species arrays
    INTEGER, PRIVATE, SAVE, ALLOCATABLE ::                                   &
   & jms_nc(:), jzs_nc(:)

    REAL(KIND=rspec), PRIVATE, SAVE, ALLOCATABLE ::                          &
   & sqzs_nc(:), xis_nc(:), ymus_nc(:, :, :),                                    &
   & bsjbps_nc(:), bsjbts_nc(:),                                              &
   & upars_nc(:, :, :), uthetas_nc(:, :, :),                                      &
   & gfls_nc(:, :), dpss_nc(:, :), dtss_nc(:, :),                                 &
   & qfls_nc(:, :), chipss_nc(:, :), chitss_nc(:, :),                             &
   & tauss_nc(:, :)

    !Physical constants, mathematical constants, conversion factors
    REAL(KIND=rspec), PRIVATE, PARAMETER :: &
        z_coulomb = 1.6022e-19_rspec, & !Coulomb charge [coul]
        z_epsilon0 = 8.8542e-12_rspec, & !permittivity of free space [F/m]
        z_j7kv = 1.6022e-16_rspec, & !energy conversion factor [J/keV]
        z_pi = 3.141592654_rspec, & !pi [-]
        z_protonmass = 1.6726e-27_rspec         !proton mass [kg]
    REAL(KIND=rspec), PRIVATE, PARAMETER :: &
        one = 1.0_rspec, & !REAL 1
        zero = 0.0_rspec         !REAL 0

    !-------------------------------------------------------------------------------
    ! Public procedures
    !-------------------------------------------------------------------------------
CONTAINS

    SUBROUTINE NCLASS(m_i, m_z,                                               &
         &                  p_b2, p_bm2, p_eb, p_fhat, p_fm, p_ft, p_grbm2,              &
         &                  p_grphi, p_gr2phi, p_ngrth, amu_i, grt_i, temp_i,           &
         &                  den_iz, fex_iz, grp_iz, ipr,                              &
         &                  iflag,                                         &
         &                  L_BANANA, L_PFIRSCH, L_POTATO, K_ORDER,                   &
         &                  C_DEN, C_POTB, C_POTL,                                   &
         &                  P_ETAP, P_JBBS, P_JBEX, P_JBOH,                           &
         &                  M_S, JM_S, JZ_S,                                         &
         &                  BSJBP_S, BSJBT_S,                                       &
         &                  GFL_S, DN_S, VNNT_S, VNEB_S, VNEX_S, DP_SS, DT_SS,           &
         &                  UPAR_S, UTHETA_S,                                       &
         &                  QFL_S, CHI_S, VQNT_S, VQEB_S, VQEX_S,                      &
         &                  CHIP_SS, CHIT_SS,                                       &
         &                  CALM_I, CALN_II, CAPM_II, CAPN_II, YMU_S,                  &
         &                  SQZ_S, XI_S, TAU_SS)
        !-------------------------------------------------------------------------------
        !NCLASS calculates the neoclassical transport properties of a multiple
        !  species axisymmetric plasma using k_order parallel and radial force
        !  balance equations for each species
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  m_i                 -number of isotopes (> 1) [-]
        !  m_z                 -highest charge state [-]
        !  p_b2                -<B**2> [T**2]
        !  p_bm2               -<1/B**2> [/T**2]
        !  p_eb                -<E.B> [V*T/m]
        !  p_fhat              -mu_0*F/(dPsi/dr) [rho/m]
        !  p_fm(m)             -poloidal moments of drift factor for PS [/m**2]
        !  p_ft                -trapped fraction [-]
        !  p_grbm2             -<grad(rho)**2/B**2> [rho**2/m**2/T**2]
        !  p_grphi             -potential gradient Phi' [V/rho]
        !  p_gr2phi            -second potential gradient Psi'(Phi'/Psi')' [V/rho**2]
        !  p_ngrth             -<n.grad(Theta)> [/m]
        !  amu_i(i)            -atomic mass number [-]
        !  grt_i(i)            -temperature gradient [keV/rho]
        !  temp_i(i)           -temperature [keV]
        !  den_iz(i,z)         -density [/m**3]
        !  fex_iz(3,i,z)       -moments of external parallel force [T*n/m**3]
        !  grp_iz(i,z)         -pressure gradient [keV/m**3/rho]
        !Output:
        !  iflag               -error and warning flag [-]
        !                      =-1 warning
        !                      =0 no warnings or errors
        !                      =1 error
        !Optional input:
        !  L_BANANA            -option to include banana viscosity [logical]
        !  L_PFIRSCH           -option to include Pfirsch-Schluter viscosity [logical]
        !  L_POTATO            -option to include potato orbits [logical]
        !  K_ORDER             -order of v moments to be solved [-]
        !                      =2 u and q (default)
        !                      =3 u, q, and u2
        !                      =else error
        !  C_DEN-density cutoff below which species is ignored (default 1.e10) [/m**3]
        !  C_POTB-kappa(0)*Bt(0)/[2*q(0)**2] [T]
        !  C_POTL-q(0)*R(0) [m]
        !Optional output:
        !
        !  Terms summed over species
        !
        !  P_ETAP-parallel electrical resistivity [Ohm*m]
        !  P_JBBS-<J_bs.B> [A*T/m**2]
        !  P_JBEX-<J_ex.B> current response to fex_iz [A*T/m**2]
        !  P_JBOH-<J_OH.B> Ohmic current [A*T/m**2]
        !
        !  Species mapping
        !
        !  M_S                 -number of species [ms>1]
        !  JM_S(s)             -isotope number of s [-]
        !  JZ_S(s)             -charge state of s [-]
        !
        !  Bootstrap current and electrical resistivity
        !
        !  BSJBP_S(s)          -<J_bs.B> driven by unit p'/p of s [A*T*rho/m**2]
        !  BSJBT_S(s)          -<J_bs.B> driven by unit T'/T of s [A*T*rho/m**2]
        !
        !  Continuity equation
        !
        !  GFL_S(m,s)-radial particle flux comps of s [rho/m**3/s]
        !             m=1, banana-plateau, p' and T'
        !             m=2, Pfirsch-Schluter
        !             m=3, classical
        !             m=4, banana-plateau, <E.B>
        !             m=5, banana-plateau, external parallel force fex_iz
        !  DN_S(s)             -diffusion coefficients (diag comp) [rho**2/s]
        !  VNNT_S(s)           -convection velocity (off diag p',T' comps) [rho/s]
        !  VNEB_S(s)           -<E.B> particle convection velocity [rho/s]
        !  VNEX_S(s)           -external force particle convection velocity [rho/s]
        !  DP_SS(s1,s2)-diffusion coefficient of s2 on p'/p of s1 [rho**2/s]
        !  DT_SS(s1,s2)-diffusion coefficient of s2 on T'/T of s1 [rho**2/s]
        !
        !Momentum equation
        !
        !  UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]
        !                m=1, p', T', Phi'
        !                m=2, <E.B>
        !                m=3, fex_iz
        !  UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]
        !                  m=1, p', T'
        !                  m=2, <E.B>
        !                  m=3, fex_iz
        !
        !Energy equation
        !
        !  QFL_S(m,s)-radial heat conduction flux comps of s [W*rho/m**3]
        !             m=1, banana-plateau, p' and T'
        !             m=2, Pfirsch-Schluter
        !             m=3, classical
        !             m=4, banana-plateau, <E.B>
        !             m=5, banana-plateau, external parallel force fex_iz
        !  CHI_S(s)            -conduction coefficients (diag comp) [rho**2/s]
        !  VQNT_S(s)           -conduction velocity (off diag p',T' comps) [rho/s]
        !  VQEB_S(s)           -<E.B> heat convection velocity [rho/s]
        !  VQEX_S(s)           -external force heat convection velocity [rho/s]
        !  CHIP_SS(s1,s2)-heat cond coefficient of s2 on p'/p of s1 [rho**2/s]
        !  CHIT_SS(s1,s2)-heat cond coefficient of s2 on T'/T of s1 [rho**2/s]
        !
        !Friction coefficients
        !
        !  CAPM_II(K_ORDER,K_ORDER,m_i,m_i)-test particle (tp) friction matrix [-]
        !  CAPN_II(K_ORDER,K_ORDER,m_i,m_i)-field particle (fp) friction matrix [-]
        !  CALM_I(K_ORDER,K_ORDER,m_i)-tp eff friction matrix [kg/m**3/s]
        !  CALN_II(K_ORDER,K_ORDER,m_i,m_i)-fp eff friction matrix [kg/m**3/s]
        !
        !Viscosity coefficients
        !
        !  YMU_S(s)-normalized viscosity for s [kg/m**3/s]
        !
        !Miscellaneous
        !
        !  SQZ_S(s)            -orbit squeezing factor for s [-]
        !  XI_S(s)             -charge weighted density factor of s [-]
        !  TAU_SS(s1,s2)       -90 degree scattering time [s]
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        INTEGER, INTENT(IN) ::                                                   &
       & m_i, m_z, ipr

        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & p_b2, p_bm2, p_eb, p_fhat, p_fm(:), p_ft, p_grbm2, p_grphi, p_gr2phi,           &
       & p_ngrth,                                                                &
       & amu_i(:), grt_i(:), temp_i(:),                                            &
       & den_iz(:, :), grp_iz(:, :), fex_iz(:, :, :)

        !Declaration of output variables
        INTEGER, INTENT(OUT) ::                                                  &
       & iflag

        !Declaration of optional input variables
        LOGICAL, INTENT(IN), OPTIONAL ::                                         &
       & L_BANANA, L_PFIRSCH, L_POTATO

        INTEGER, INTENT(IN), OPTIONAL ::                                         &
       & K_ORDER

        REAL(KIND=rspec), INTENT(IN), OPTIONAL ::                                &
       & C_DEN, C_POTB, C_POTL

        !Declaration of optional output variables
        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & P_ETAP, P_JBBS, P_JBEX, P_JBOH

        INTEGER, INTENT(OUT), OPTIONAL ::                                        &
       & M_S, JM_S(m_i*m_z), JZ_S(m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & BSJBP_S(m_i*m_z), BSJBT_S(m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & DP_SS(m_i*m_z, m_i*m_z), DT_SS(m_i*m_z, m_i*m_z),                                                  &
       & GFL_S(5, m_i*m_z), DN_S(m_i*m_z), VNEB_S(m_i*m_z), VNEX_S(m_i*m_z), VNNT_S(m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & UPAR_S(3, 3, m_i*m_z), UTHETA_S(3, 3, m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & CHIP_SS(m_i*m_z, m_i*m_z), CHIT_SS(m_i*m_z, m_i*m_z),                                              &
       & QFL_S(5, m_i*m_z), CHI_S(m_i*m_z), VQEB_S(m_i*m_z), VQEX_S(m_i*m_z), VQNT_S(m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & CAPM_II(3, 3, m_i, m_i), CAPN_II(3, 3, m_i, m_i), CALM_I(3, 3, m_i), CALN_II(3, 3, m_i, m_i)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & YMU_S(3, 3, m_i*m_z)

        REAL(KIND=rspec), INTENT(OUT), OPTIONAL ::                               &
       & SQZ_S(m_i*m_z), XI_S(m_i*m_z), TAU_SS(m_i*m_z, m_i*m_z)

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, iza, k, l

        REAL(KIND=rspec) ::                                                      &
       & dent, rval

        REAL(KIND=rspec), ALLOCATABLE ::                                         &
       & denz2(:)

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        !Warning and error flag
        iflag = 0
        if (ipr .le. 10) then
            !  write(*,*) ' -- dans fortran 90 --',m_i
        end if
        !     write(*,*) 'iflag=',iflag,' ipr=',ipr
        !     write(*,*) 'M_S',M_S
        !     write(*,*) 'JM_S',JM_S
        !      CALL mexErrMsgTxt('stop')

        !Number of velocity moments
        IF (PRESENT(K_ORDER)) THEN

            IF (K_ORDER < 2 .OR. K_ORDER > 3) THEN

                iflag = 1
                write (*, *) ' K_ORDER =', K_ORDER
                write (*, *) 'NCLASS(1)/ERROR: K_ORDER must be 2 or 3'
                GOTO 9999

            ELSE

                k_order_nc = K_ORDER

            END IF

        ELSE

            k_order_nc = 2

        END IF

        !Banana contribution to viscosity
        !On by default, unless user turns it off or specifies p_ft=0
        !      write(*,*) 'p_ft=',p_ft
        IF (ABS(p_ft) > 0.0_rspec) THEN

            l_banana_nc = .TRUE.

        ELSE

            l_banana_nc = .FALSE.

        END IF

        IF (PRESENT(L_BANANA) .AND. (.NOT. L_BANANA)) l_banana_nc = .FALSE.

        !Pfirsch-Schluter contribution to viscosity
        !On by default, unless user turns it off or specifies p_fm=0

        IF (ABS(SUM(p_fm(:))) > 0.0_rspec) THEN

            l_pfirsch_nc = .TRUE.
            mf_nc = SIZE(p_fm)

        ELSE

            l_pfirsch_nc = .FALSE.
            mf_nc = 0

        END IF

        IF (PRESENT(L_PFIRSCH) .AND. (.NOT. L_PFIRSCH))                           &
       &  l_pfirsch_nc = .FALSE.

        !Potato orbit contribution to viscosity
        !Off by default, unless user turns it on and provides constants
        IF (PRESENT(L_POTATO) .AND.                                               &
       &   PRESENT(C_POTB) .AND.                                                 &
       &   PRESENT(C_POTL)) THEN

            IF (L_POTATO) THEN

                l_potato_nc = .TRUE.
                cpotb_nc = C_POTB
                cpotl_nc = C_POTL

            ELSE

                l_potato_nc = .FALSE.
                cpotb_nc = 0.0_rspec
                cpotl_nc = 0.0_rspec

            END IF

        ELSE

            l_potato_nc = .FALSE.
            cpotb_nc = 0.0_rspec
            cpotl_nc = 0.0_rspec

        END IF

        !No potato viscosity if there are no banana or Pfirsch-Schluter terms
        IF ((.NOT. l_banana_nc) .AND. (.NOT. l_pfirsch_nc)) THEN

            l_potato_nc = .FALSE.
            cpotb_nc = 0.0_rspec
            cpotl_nc = 0.0_rspec

        END IF

        !Cutoff density
        IF (PRESENT(C_DEN)) THEN

            !Use input value
            cden_nc = C_DEN

        ELSE

            !Use default value
            cden_nc = 1.0e10_rspec

        END IF

        !Trapped fraction between 0 and 1 inclusive
        IF ((p_ft < 0.0_rspec) .OR. (p_ft > 1.0_rspec)) THEN

            iflag = 1
            write (*, *) 'NCLASS(2)/ERROR: must have 0 <= p_ft <= 1'
            GOTO 9999

        END IF

        !Allocate private data
        !       write(*,*) 'lancement de nclass_init',cden_nc
        CALL NCLASS_INIT(m_i, m_z, amu_i, den_iz, iflag)

        !Check messages
        IF (iflag /= 0) THEN

            !       write(*,*) 'NCLASS(3)/'
            IF (iflag > 0) GOTO 9999

        END IF

        !-------------------------------------------------------------------------------
        !Set various plasma properties
        !-------------------------------------------------------------------------------
        !Friction coefficients
        !      write(*,*) 'lancement de mn',temp_i
        CALL NCLASS_MN(amu_i, temp_i)

        !Thermal velocities
        vti_nc(1:mi_nc) = SQRT(2.0_rspec*z_j7kv*temp_i(1:mi_nc)                    &
       &                     /amu_i(1:mi_nc)/z_protonmass)
        !       write(*,*) 'mi_nc=',mi_nc
        !       write(*,*) 'Te=',temp_i(1:mi_nc)
        !Collision times
        !     write(*,*) 'lancement de collision'
        CALL NCLASS_TAU(amu_i, temp_i, den_iz)

        !Reduced friction coefficients
        calmi_nc(:, :, :) = 0.0_rspec

        DO i = 1, mi_nc !Over isotopes

            DO k = 1, k_order_nc !Over first moment

                DO l = 1, k_order_nc !Over second moment

                    !Test particle component
                    calmi_nc(k, l, i) =                                                   &
             &        SUM(amntii_nc(i, 1:mi_nc)*capmii_nc(k, l, i, 1:mi_nc))
                    !            write(*,*) 'i=',i,' k=',k,' l=',l
                    !            write(*,*) 'amntii_nc(i,1:mi_nc)=',amntii_nc(i,1)
                    !Field particle component
                    calnii_nc(k, l, i, 1:mi_nc) =                                          &
             &        amntii_nc(i, 1:mi_nc)*capnii_nc(k, l, i, 1:mi_nc)
                    if (k .eq. 1 .and. l .lt. 3 .and. i .eq. 1) then
                        !               write(*,*) 'calmi_nc',calmi_nc(k,l,i),l
                    end if

                END DO !Over second moment

            END DO !Over first moment

        END DO !Over isotopes

        !Species charge state density factor, total pressure, squeezing
        dent = 0.0_rspec
        ALLOCATE (denz2(1:(mi_nc + ms_nc)))
        !      write(*,*) 'ms_nc=',ms_nc,' mi_nc=',mi_nc
        DO i = 1, mi_nc !Over isotopes

            denz2(i) = 0.0_rspec

            DO iza = 1, mz_nc !Over charge states

                IF (den_iz(i, iza) > cden_nc) THEN

                    denz2(i) = denz2(i) + den_iz(i, iza)*iza**2
                    dent = dent + den_iz(i, iza)*temp_i(i)

                END IF

            END DO !Over charge states

        END DO !Over isotopes

        DO i = 1, ms_nc !Over species

            iza = ABS(jzs_nc(i))
            xis_nc(i) = den_iz(jms_nc(i), iza)*REAL(jzs_nc(i), rspec)**2               &
         &            /denz2(jms_nc(i))
            sqzs_nc(i) = 1.0_rspec + p_fhat**2/p_b2*amu_i(jms_nc(i))                   &
         &                       *z_protonmass*p_gr2phi                            &
         &                       /(z_coulomb*REAL(jzs_nc(i), rspec))
            IF (sqzs_nc(i) > 10.0_rspec) sqzs_nc(i) = 10.0_rspec
            IF (sqzs_nc(i) < 0.5_rspec) sqzs_nc(i) = 0.5_rspec

        END DO !Over species

        DEALLOCATE (denz2)

        !Normalized viscosities
        !      write(*,*) 'lancement de Mu'
        CALL NCLASS_MU(p_fm, p_ft, p_ngrth, amu_i, temp_i, den_iz, ipr)

        !Add potential gradient to pressure gradient
        DO i = 1, ms_nc !Over species

            iza = ABS(jzs_nc(i))
            grppiz_nc(jms_nc(i), iza) = grp_iz(jms_nc(i), iza)                         &
         &                           + p_grphi*den_iz(jms_nc(i), iza)                &
         &                           *REAL(jzs_nc(i), rspec)*z_coulomb/z_j7kv
            !        grppiz_nc(jms_nc(i),iza)=grp_iz(jms_nc(i),iza)                         &
            !     &                           +p_grphi*den_iz(jms_nc(i),iza)                &
            !     &                           *REAL(jzs_nc(i),rspec)*z_coulomb
            !        write(*,*) 'grppiz_nc(jms_nc(i),iza)=',grppiz_nc(jms_nc(i),iza)
            !        write(*,*) 'grp_iz(jms_nc(i),iza)=',grp_iz(jms_nc(i),iza)
            !        write(*,*) 'den_iz(jms_nc(i),iza)=',den_iz(jms_nc(i),iza)
            !        write(*,*) 'p_grphi=',p_grphi,z_coulomb,z_j7kv
            rval = p_grphi*den_iz(jms_nc(i), iza)                &
                  &                           *REAL(jzs_nc(i), rspec)*z_coulomb/z_j7kv
            !        write(*,*) 'cont=',rval

        END DO !Over species
        !      write(*,*) 'fin boucle especes'

        !-------------------------------------------------------------------------------
        !Solve for parallel flows within a surface
        !-------------------------------------------------------------------------------
        iflag = 0
        !      write(*,*) 'lancement de nclass_flow ipr=',ipr
        !      write(*,*) 'Te=',temp_i
        !      write(*,*) 'grt=',grt_i
        !      write(*,*) 'ne(1,1)=',den_iz(1,1)
        !      write(*,*) 'pfhat=',p_fhat,'b2=',p_b2
        !       write(*,*) 'flow, gradient Te',grt_i
        !       write(*,*) 'flow, fex',fex_iz(1,1,1)
        !       write(*,*) 'flow, den',den_iz(1,1)

        CALL NCLASS_FLOW(p_b2, p_bm2, p_eb, p_fhat, p_grbm2, grt_i, temp_i,            &
        &                 den_iz, fex_iz, iflag, ipr)
        !      CALL mexErrMsgTxt('stop')

        !Check messages
        IF (iflag /= 0) THEN

            write (*, *) 'NCLASS(4)/'
            IF (iflag > 0) GOTO 9999

        END IF

        !-------------------------------------------------------------------------------
        !Calculate poloidal velocity from parallel velocity
        !-------------------------------------------------------------------------------
        DO i = 1, ms_nc !Over species

            iza = IABS(jzs_nc(i))

            DO k = 1, k_order_nc !Over moments

                DO l = 1, 3 !Over forces

                    uthetas_nc(k, l, i) = upars_nc(k, l, i)/p_b2
                    !            if (l .eq. 2 .and. ipr .eq. 1 .and. k .eq. 1) then
                    !              write(*,*) 'uthetas_nc(1,2,i)=',uthetas_nc(1,2,i)
                    !              write(*,*) 'upars_nc(1,2,i)=',upars_nc(1,2,i)
                    !              write(*,*) 'i=',i,' p_b2=',p_b2
                    !            endif

                    IF (l == 1) THEN

                        IF (k == 1) THEN

                            !Add p' and Phi'
                            uthetas_nc(k, l, i) = uthetas_nc(k, l, i)                            &
                 &                           + p_fhat*grppiz_nc(jms_nc(i), iza)*z_j7kv       &
                 &                            /(z_coulomb*REAL(jzs_nc(i), rspec)            &
                 &                            *den_iz(jms_nc(i), iza))/p_b2

                        ELSEIF (k == 2) THEN

                            !Add T'
                            uthetas_nc(k, l, i) = uthetas_nc(k, l, i)                            &
                 &                            + p_fhat*z_j7kv*grt_i(jms_nc(i))              &
                 &                           /(REAL(jzs_nc(i), rspec)*z_coulomb*p_b2)

                        END IF

                    END IF
                    if (ipr .eq. 1 .and. l .eq. 2) then
                        !                write(*,*) 'i=',i
                        !                write(*,*) 'uthetas_nc(1,2,i)=',uthetas_nc(1,2,i)
                        !              if (i .eq. ms_nc) stop
                    end if

                END DO !Over forces

            END DO !Over moments

        END DO !Over species

        !-------------------------------------------------------------------------------
        !Output
        !-------------------------------------------------------------------------------
        !Species mapping
        IF (PRESENT(M_S)) M_S = ms_nc
        IF (PRESENT(JM_S)) JM_S(1:ms_nc) = jms_nc(1:ms_nc)
        IF (PRESENT(JZ_S)) JZ_S(1:ms_nc) = jzs_nc(1:ms_nc)

        !Electrical resistivity and currents
        IF (PRESENT(P_ETAP)) P_ETAP = petap_nc
        !        write(*,*) 'transfert boot :',pjbbs_nc
        IF (PRESENT(P_JBBS)) P_JBBS = pjbbs_nc
        IF (PRESENT(P_JBEX)) P_JBEX = pjbex_nc
        IF (PRESENT(P_JBOH)) P_JBOH = pjboh_nc
        IF (PRESENT(BSJBP_S)) BSJBP_S(1:ms_nc) = bsjbps_nc(1:ms_nc)
        IF (PRESENT(BSJBT_S)) BSJBT_S(1:ms_nc) = bsjbts_nc(1:ms_nc)

        !Continuity equation
        IF (PRESENT(DP_SS)) DP_SS(1:ms_nc, 1:ms_nc) = dpss_nc(1:ms_nc, 1:ms_nc)
        IF (PRESENT(DT_SS)) DT_SS(1:ms_nc, 1:ms_nc) = dtss_nc(1:ms_nc, 1:ms_nc)
        IF (PRESENT(GFL_S)) GFL_S(1:5, 1:ms_nc) = gfls_nc(1:5, 1:ms_nc)

        IF (PRESENT(DN_S)) THEN

            DO i = 1, ms_nc !Over species

                DN_S(i) = dpss_nc(i, i)

            END DO !Over species

        END IF

        IF (PRESENT(VNNT_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VNNT_S(i) = (SUM(gfls_nc(1:3, i))                                       &
           &               + dn_s(i)*(grp_iz(jms_nc(i), iza)                           &
           &               - den_iz(jms_nc(i), iza)*grt_i(jms_nc(i)))                  &
           &              /temp_i(jms_nc(i)))/den_iz(jms_nc(i), iza)

            END DO !Over species

        END IF

        IF (PRESENT(VNEB_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VNEB_S(i) = gfls_nc(4, i)/den_iz(jms_nc(i), iza)

            END DO !Over species

        END IF

        IF (PRESENT(VNEX_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VNEX_S(i) = gfls_nc(5, i)/den_iz(jms_nc(i), iza)

            END DO !Over species

        END IF

        !Momentum equation

        IF (PRESENT(UPAR_S)) UPAR_S(1:k_order_nc, 1:3, 1:ms_nc)                     &
       &                    = upars_nc(1:k_order_nc, 1:3, 1:ms_nc)
        IF (PRESENT(UTHETA_S)) UTHETA_S(1:k_order_nc, 1:3, 1:ms_nc)                 &
       &                      = uthetas_nc(1:k_order_nc, 1:3, 1:ms_nc)

        !Energy equation
        IF (PRESENT(QFL_S)) QFL_S(1:5, 1:ms_nc) = qfls_nc(1:5, 1:ms_nc)
        !      write(6,*) 'qfls_nc(1,1)=',qfls_nc(1,1)
        IF (PRESENT(CHI_S)) THEN

            DO i = 1, ms_nc !Over species

                CHI_S(i) = chipss_nc(i, i) + chitss_nc(i, i)

            END DO !Over species

        END IF

        IF (PRESENT(VQNT_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VQNT_S(i) = (SUM(qfls_nc(1:3, i))/z_j7kv/den_iz(jms_nc(i), iza)          &
           &               + chi_s(i)*grt_i(jms_nc(i)))/temp_i(jms_nc(i))

            END DO !Over species

        END IF

        IF (PRESENT(VQEB_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VQEB_S(i) = qfls_nc(4, i)/den_iz(jms_nc(i), iza)/temp_i(jms_nc(i))       &
           &              /z_j7kv

            END DO !Over species

        END IF

        IF (PRESENT(VQEX_S)) THEN

            DO i = 1, ms_nc !Over species

                iza = ABS(jzs_nc(i))
                VQEX_S(i) = qfls_nc(5, i)/den_iz(jms_nc(i), iza)/temp_i(jms_nc(i))       &
           &              /z_j7kv

            END DO !Over species

        END IF

        IF (PRESENT(CHIP_SS)) CHIP_SS(1:ms_nc, 1:ms_nc)                            &
       &                     = chipss_nc(1:ms_nc, 1:ms_nc)
        IF (PRESENT(CHIT_SS)) CHIT_SS(1:ms_nc, 1:ms_nc)                            &
       &                     = chitss_nc(1:ms_nc, 1:ms_nc)

        !Friction coefficients
        IF (PRESENT(CALM_I))                                                      &
       &           CALM_I(1:k_order_nc, 1:k_order_nc, 1:mi_nc)                     &
       &           = calmi_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc)
        IF (PRESENT(CALN_II))                                                     &
       &           CALN_II(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)            &
       &           = calnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)
        IF (PRESENT(CAPM_II))                                                     &
       &           CAPM_II(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)            &
       &           = capmii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)
        IF (PRESENT(CAPN_II))                                                     &
       &           CAPN_II(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)            &
       &           = capnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc)

        !Viscosity coefficients
        IF (PRESENT(YMU_S)) YMU_S(1:k_order_nc, 1:k_order_nc, 1:ms_nc)              &
       &                   = ymus_nc(1:k_order_nc, 1:k_order_nc, 1:ms_nc)

        !Miscellaneous
        IF (PRESENT(SQZ_S)) SQZ_S(1:ms_nc) = sqzs_nc(1:ms_nc)
        IF (PRESENT(XI_S)) XI_S(1:ms_nc) = xis_nc(1:ms_nc)
        IF (PRESENT(TAU_SS)) TAU_SS(1:ms_nc, 1:ms_nc)                              &
       &                    = tauss_nc(1:ms_nc, 1:ms_nc)

9999    CONTINUE

    END SUBROUTINE NCLASS

    SUBROUTINE NCLASS_FLOW(p_b2, p_bm2, p_eb, p_fhat, p_grbm2, grt_i,             &
   &                       temp_i, den_iz, fex_iz, iflag, ipr)
        !-------------------------------------------------------------------------------
        !NCLASS_FLOW calculates the neoclassical flows u0, u1=q/p (and and optionally
        !  u2) plus some of the other transport properties
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  p_b2                -<B**2> [T**2]
        !  p_bm2               -<1/B**2> [/T**2]
        !  p_eb                -<E.B> [V*T/m]
        !  p_fhat              -mu_0*F/(dPsi/dr) [rho/m]
        !  p_grbm2             -<grad(rho)**2/B**2> [rho**2/m**2/T**2]
        !  grt_i(i)            -temperature gradient [keV/rho]
        !  temp_i(i)           -temperature [keV]
        !  den_iz(i,z)         -density [/m**3]
        !  fex_iz(3,i,z)       -moments of external parallel force [T*n/m**3]
        !Output:
        !  iflag               -error and warning flag [-]
        !                      =-1 warning
        !                      =0 no warnings or errors
        !                      =1 error
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & p_b2, p_bm2, p_eb, p_fhat, p_grbm2,                                         &
       & grt_i(:), temp_i(:), den_iz(:, :), fex_iz(:, :, :)

        INTEGER, INTENT(in) ::                                                  &
       & ipr

        !Declaration of output variables
        INTEGER, INTENT(OUT) ::                                                  &
       & iflag

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, im, iz, iza, j, jm, k, l, l1, m, m1

        INTEGER ::                                                               &
       & indxk(1:k_order_nc), indxki(1:k_order_nc*mi_nc)

        REAL(KIND=rspec) ::                                                      &
       & cbp, cbpa, cbpaq, cc, ccl, ccla, cclaq, cclb, cclbq, cps, cpsa, cpsaq,            &
       & cpsb, cpsbq, d, denzc

        REAL(KIND=rspec) ::                                                      &
       & akk(1:k_order_nc, 1:k_order_nc),                                         &
       & xk(1:k_order_nc),                                                       &
       & akiki(1:k_order_nc*mi_nc, 1:k_order_nc*mi_nc),                           &
       & akikis(1:k_order_nc*mi_nc, 1:k_order_nc*mi_nc),                          &
       & xki3(1:k_order_nc*mi_nc, 3),                                             &
       & crk6i(1:k_order_nc, 6, mi_nc),                                            &
       & rk6s(1:k_order_nc, 6, ms_nc),                                             &
       & srcth(1:k_order_nc, ms_nc), srcthp(ms_nc), srctht(ms_nc),                  &
       & crhatp(1:k_order_nc, ms_nc, mi_nc),                                       &
       & crhatt(1:k_order_nc, ms_nc, mi_nc),                                       &
       & rhatp(1:k_order_nc, ms_nc, ms_nc),                                        &
       & rhatt(1:k_order_nc, ms_nc, ms_nc),                                        &
       & uaip(1:k_order_nc, ms_nc, ms_nc),                                         &
       & uait(1:k_order_nc, ms_nc, ms_nc),                                         &
       & xabp(k_order_nc*mi_nc, ms_nc),                                           &
       & xabt(k_order_nc*mi_nc, ms_nc)
        !
        ! LAPACK decomposition
        !
        REAL(KIND=rspec) ::                                                      &
       & alp(1:k_order_nc, 1:k_order_nc),                                         &
       & alps(1:k_order_nc, 1:k_order_nc),                                        &
       & vecm(1:k_order_nc),                                                     &
       & alp2(1:k_order_nc*mi_nc, 1:k_order_nc*mi_nc),                            &
       & alp2s(1:k_order_nc*mi_nc, 1:k_order_nc*mi_nc),                           &
       & vecm2(1:k_order_nc*mi_nc),                                              &
       & ama(1:k_order_nc),                                                      &
       & ama2(1:k_order_nc*mi_nc),                                               &
       & sumdgesv(1:k_order_nc*mi_nc)
        REAL(KIND=rspec) ::                                                      &
       & ipvt(1:k_order_nc),                                                     &
       & rcond, z(1:k_order_nc)

        INTEGER ::                                                               &
       & indl2(1:k_order_nc*mi_nc),                                              &
       & indl(1:k_order_nc), n, job, lda, ichoix

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        !Local arrays
        !      CALL mexErrMsgTxt('stop')

        akk(:, :) = 0.0_rspec
        xk(:) = 0.0_rspec
        akiki(:, :) = 0.0_rspec
        xki3(:, :) = 0.0_rspec
        crk6i(:, :, :) = 0.0_rspec
        rk6s(:, :, :) = 0.0_rspec
        srcth(:, :) = 0.0_rspec
        srcthp(:) = 0.0_rspec
        srctht(:) = 0.0_rspec
        crhatp(:, :, :) = 0.0_rspec
        crhatt(:, :, :) = 0.0_rspec
        rhatp(:, :, :) = 0.0_rspec
        rhatt(:, :, :) = 0.0_rspec
        uaip(:, :, :) = 0.0_rspec
        uait(:, :, :) = 0.0_rspec
        xabp(:, :) = 0.0_rspec
        xabt(:, :) = 0.0_rspec

        !Private data
        petap_nc = 0.0_rspec
        pjbbs_nc = 0.0_rspec
        pjbex_nc = 0.0_rspec
        pjboh_nc = 0.0_rspec
        gfls_nc(:, :) = 0.0_rspec
        qfls_nc(:, :) = 0.0_rspec

        !Local constants
        cc = (p_fhat/z_coulomb)*z_j7kv
        cbp = p_fhat/p_b2/z_coulomb
        cps = (p_fhat/z_coulomb)*(1.0/p_b2 - p_bm2)
        ccl = (p_grbm2/z_coulomb)/p_fhat
        !  write(*,*) 'cc=',cc,p_fhat,z_coulomb,z_j7kv

        !-------------------------------------------------------------------------------
        !Calculate species information for reduced charge state formalism
        !-------------------------------------------------------------------------------
        DO i = 1, ms_nc !Over species

            !Isotopic and charge state indices
            im = jms_nc(i)
            iz = jzs_nc(i)
            iza = IABS(iz)

            !Response matrix
            akk(:, :) = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)

            !Get lu decomposition of response matrix
            iflag = 0

            !        CALL NCLASS_DECOMP(akk,k_order_nc,indxk,d,iflag,ipr)
            !        write(*,*) 'fin nclass_decomp i=',i,im,iz,iza

            !Check messages
            !        IF(iflag == 1) THEN

            !          write(*,*) 'NCLASS_FLOW(1)/'
            !          GOTO 9999

            !        ENDIF

            !Source and response from lambda terms involving isotopic flows
            ichoix = 0
            if (ichoix .eq. 0) then
                DO k = 1, k_order_nc !Over velocity moments
                    !          write(*,*) '---------------------',k,'----------------'
                    rk6s(k, k, i) = xis_nc(i)
                    alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
                    alps = alp
                    ama = rk6s(:, k, i)
                    !          write(*,*) 'B=',ama(1),ama(2),ama(3)
                    call normalisation(alp, ama, k_order_nc, vecm)
                    !          write(*,*) 'BN=',ama(1),ama(2),ama(3)
                    !            if (vecm(1) .le. 1e-6) then
                    !              write(*,*) 'vecm=',vecm(1),vecm(2),vecm(3)
                    !            endif
                    call dgesv(k_order_nc, 1, alp, k_order_nc,                             &
             &               indl, ama, k_order_nc, iflag)
                    if (iflag .ne. 0) then

                        write (*, *) 'iflag=', iflag

                    end if
                    !          write(*,*) 'BS=',ama(1),ama(2),ama(3)
                    !       write(*,*) 'A1=',alps(1,1)*ama(1)+alps(1,2)*ama(2)                     &
                    !     &                                  +alps(1,3)*ama(3)
                    !       write(*,*) 'A2=',alps(2,1)*ama(1)+alps(2,2)*ama(2)                     &
                    !     &                                  +alps(2,3)*ama(3)
                    !       write(*,*) 'A3=',alps(3,1)*ama(1)+alps(3,2)*ama(2)                     &
                    !     &                                  +alps(3,3)*ama(3)
                    !          CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,rk6s(:,k,i),ipr)
                    crk6i(:, k, im) = crk6i(:, k, im) + xis_nc(i)*ama
                    rk6s(:, k, i) = ama
                    !          write(*,*) 'rk6s(:,k,i)=',rk6s(:,k,i)
                END DO !Over velocity moments
            end if
            !        stop
            !Source and response from poloidal source (p' and T') terms
            srcth(1, i) = (cc/iz)*grppiz_nc(im, iza)/den_iz(im, iza)
            srcth(2, i) = (cc/iz)*grt_i(im)

            DO k = 1, k_order_nc !Over velocity moments

                rk6s(k, 4, i) = SUM(srcth(:, i)*ymus_nc(k, :, i))

            END DO !Over velocity moments

            alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
            ama = rk6s(:, 4, i)
            !        write(*,*) 'ama=',ama
            call normalisation(alp, ama, k_order_nc, vecm)
            call dgesv(k_order_nc, 1, alp, k_order_nc,                             &
         &               indl, ama, k_order_nc, iflag)
            !        CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,rk6s(:,4,i),ipr)

            crk6i(:, 4, im) = crk6i(:, 4, im) + xis_nc(i)*ama
            rk6s(:, 4, i) = ama
            !Source and response from unit p'/p and T'/T terms for decomposition of fluxes
            srcthp(i) = -(cc/iz)*temp_i(im)
            srctht(i) = -(cc/iz)*temp_i(im)
            rhatp(:, i, i) = srcthp(i)*ymus_nc(:, 1, i)
            rhatt(:, i, i) = srctht(i)*ymus_nc(:, 2, i)

            alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
            ama = rhatp(1:k_order_nc, i, i)
            call normalisation(alp, ama, k_order_nc, vecm)
            call dgesv(k_order_nc, 1, alp, k_order_nc,                                &
         &               indl, ama, k_order_nc, iflag)
            rhatp(1:k_order_nc, i, i) = ama
            crhatp(:, i, im) = crhatp(:, i, im) + xis_nc(i)*ama
            !        CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,                              &
            !     &                      rhatp(1:k_order_nc,i,i),ipr)
            alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
            ama = rhatt(1:k_order_nc, i, i)
            call normalisation(alp, ama, k_order_nc, vecm)
            call dgesv(k_order_nc, 1, alp, k_order_nc,                                &
         &               indl, ama, k_order_nc, iflag)
            !       CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,                              &
            !     &                      rhatt(1:k_order_nc,i,i),ipr)
            rhatt(1:k_order_nc, i, i) = ama
            crhatt(:, i, im) = crhatt(:, i, im) + xis_nc(i)*ama

            !Source and response from parallel electric field terms for resistivity
            rk6s(1, 5, i) = -iz*z_coulomb*den_iz(im, iza)
            rk6s(2, 5, i) = 0.0
            alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
            ama = rk6s(1:k_order_nc, 5, i)
            call normalisation(alp, ama, k_order_nc, vecm)
            call dgesv(k_order_nc, 1, alp, k_order_nc,                               &
         &               indl, ama, k_order_nc, iflag)
            !        CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,rk6s(1:k_order_nc,5,i),ipr)
            crk6i(:, 5, im) = crk6i(:, 5, im) + xis_nc(i)*ama
            rk6s(1:k_order_nc, 5, i) = ama
            !Source and response from external force terms
            rk6s(1:k_order_nc, 6, i) = -fex_iz(1:k_order_nc, im, iza)
            alp = xis_nc(i)*calmi_nc(:, :, im) - ymus_nc(:, :, i)
            ama = rk6s(1:k_order_nc, 6, i)
            call normalisation(alp, ama, k_order_nc, vecm)
            call dgesv(k_order_nc, 1, alp, k_order_nc,                               &
               &               indl, ama, k_order_nc, iflag)
            !        CALL NCLASS_BACKSUB(akk,k_order_nc,indxk,rk6s(1:k_order_nc,6,i),ipr)
            rk6s(1:k_order_nc, 6, i) = ama
            crk6i(:, 6, im) = crk6i(:, 6, im) + xis_nc(i)*ama

        END DO !Over species

        !-------------------------------------------------------------------------------
        !Load coefficient matrix and source terms for isotopic flows
        !-------------------------------------------------------------------------------
        DO im = 1, mi_nc !Over isotopes 1

            DO m = 1, k_order_nc !Over velocity moments 1

                m1 = im + (m - 1)*mi_nc
                !Diagonal coefficients
                akiki(m1, m1) = 1.0_rspec
                !p' and T' force terms
                xki3(m1, 1) = crk6i(m, 4, im)
                !          write(*,*) 'xki3(m1,1)=',xki3(m1,1),m1,m,im
                !Unit p'/p and T'/T
                xabp(m1, :) = crhatp(m, :, im)
                xabt(m1, :) = crhatt(m, :, im)
                !<E.B> force terms
                xki3(m1, 2) = crk6i(m, 5, im)
                !External force
                xki3(m1, 3) = crk6i(m, 6, im)
                !  Field particle friction
                DO jm = 1, mi_nc !Over isotopes 2

                    DO l = 1, k_order_nc !Over velocity moments 2

                        l1 = jm + (l - 1)*mi_nc
                        !              if (ipr .eq. 1 .and. l1 .eq. 2) then

                        !                write(*,*) ' avant somme, akiki(m1,2)=',akiki(m1,l1)

                        !              endif
                        akiki(m1, l1) = akiki(m1, l1)                                        &
               &  + SUM(calnii_nc(1:k_order_nc, l, im, jm)*crk6i(m, 1:k_order_nc, im))

                    END DO !Over velocity moments 2

                END DO !Over isotopes 2

            END DO !Over velocity moments 1

        END DO !Over isotopes 1

        !-------------------------------------------------------------------------------
        !Get lu decomposition of coefficient matrix
        !-------------------------------------------------------------------------------
        iflag = 0
        akikis = akiki
        !      CALL NCLASS_DECOMP(akiki,k_order_nc*mi_nc,indxki,d,iflag,ipr)

        !Check messages
        !      IF(iflag == 1) THEN

        !        write(*,*) 'NCLASS_FLOW(2)/'
        !        GOTO 9999

        !      ENDIF

        !-------------------------------------------------------------------------------
        !Evaluate isotopic flows from back substitution for each source
        !-------------------------------------------------------------------------------
        !xki3(1,k) to xki3(mi_nc,k) are the isotopic velocities
        !xki3(mi_nc+1,k) to xki3(2*mi_nc,k) are the isotopic heat flows
        !xki3(2*mi_nc+1,k) to xki3(3*mi_nc,k) are the u2 flows
        !Evaluate species flows
        DO k = 1, 3 !Over forces
            alp2 = akikis
            ama2 = xki3(1:k_order_nc*mi_nc, k)
            call normalisation(alp2, ama2, k_order_nc*mi_nc, vecm2)
            !        if (k .eq. 2) then
            !          write(*,*) 'ama2=',ama2
            !        endif
            !        stop
            call dgesv(k_order_nc*mi_nc, 1, alp2, k_order_nc*mi_nc,                   &
         &                   indl2, ama2, k_order_nc*mi_nc, iflag)
            errsum = 0.0
            do l = 1, k_order_nc*mi_nc
                sumdgesv(l) = 0.0
                do j = 1, k_order_nc*mi_nc
                    sumdgesv(l) = sumdgesv(l) + akikis(l, j)*ama2(j)
                end do
                if (xki3(l, k) .ne. 0.) then
                    errsum = abs(sumdgesv(l) - xki3(l, k))/xki3(l, k) + errsum
                end if
            end do
            !        write(*,*) '----'
            !        write(*,*) 'errsum=',errsum
            xki3(1:k_order_nc*mi_nc, k) = ama2
            !        write(*,*) 'ama2=',ama2

        END DO !Over forces
        !      stop

        DO i = 1, ms_nc !Over species
            alp2 = akikis
            ama2 = xabp(1:k_order_nc*mi_nc, i)
            call normalisation(alp2, ama2, k_order_nc*mi_nc, vecm2)
            call dgesv(k_order_nc*mi_nc, 1, alp2, k_order_nc*mi_nc,                   &
         &                   indl2, ama2, k_order_nc*mi_nc, iflag)
            xabp(1:k_order_nc*mi_nc, i) = ama2
            !        CALL NCLASS_BACKSUB(akiki,k_order_nc*mi_nc,indxki,                     &
            !     &                      xabp(1:k_order_nc*mi_nc,i),ipr)
            alp2 = akikis
            ama2 = xabt(1:k_order_nc*mi_nc, i)
            call normalisation(alp2, ama2, k_order_nc*mi_nc, vecm2)
            call dgesv(k_order_nc*mi_nc, 1, alp2, k_order_nc*mi_nc,                   &
         &                   indl2, ama2, k_order_nc*mi_nc, iflag)
            xabt(1:k_order_nc*mi_nc, i) = ama2
            !        CALL NCLASS_BACKSUB(akiki,k_order_nc*mi_nc,indxki,                     &
            !     &                      xabt(1:k_order_nc*mi_nc,i),ipr)

        END DO !Over species
        !      stop
        DO i = 1, ms_nc !Over species 1

            im = jms_nc(i)

            !Force contributions
            DO m = 1, 3 !Over forces

                DO k = 1, k_order_nc !Over velocity moments

                    IF (m == 2) THEN

                        !Add normalization to <E.B>
                        upars_nc(k, m, i) = rk6s(k, 5, i)

                    ELSE

                        upars_nc(k, m, i) = rk6s(k, m + 3, i)

                    END IF

                END DO !Over velocity moments

                !Response contributions
                DO jm = 1, mi_nc !Over isotopes

                    xk(:) = 0.0_rspec

                    DO l = 1, k_order_nc !Over velocity moments

                        l1 = jm + (l - 1)*mi_nc
                        xk(:) = xk(:) - calnii_nc(1:k_order_nc, l, im, jm)*xki3(l1, m)

                    END DO !Over velocity moments

                    DO l = 1, k_order_nc !Over velocity moments

                        upars_nc(:, m, i) = upars_nc(:, m, i) + xk(l)*rk6s(:, l, i)

                    END DO !Over velocity moments

                END DO !Over isotopes

            END DO !Over forces
            !        stop
            !Unit p'/p and T'/T
            DO j = 1, ms_nc !Over species 2

                uaip(:, j, i) = rhatp(:, j, i)
                uait(:, j, i) = rhatt(:, j, i)

                !Response contributions
                DO jm = 1, mi_nc !Over isotopes

                    xk(:) = 0.0_rspec

                    DO l = 1, k_order_nc !Over velocity moments

                        l1 = jm + (l - 1)*mi_nc
                        xk(:) = xk(:) - calnii_nc(1:k_order_nc, l, im, jm)*xabp(l1, j)

                    END DO !Over velocity moments

                    DO l = 1, k_order_nc !Over velocity moments

                        uaip(:, j, i) = uaip(:, j, i) + xk(l)*rk6s(:, l, i)

                    END DO !Over velocity moments

                    xk(:) = 0.0_rspec

                    DO l = 1, k_order_nc !Over velocity moments
                        l1 = jm + (l - 1)*mi_nc

                        xk(:) = xk(:) - calnii_nc(1:k_order_nc, l, im, jm)*xabt(l1, j)

                    END DO !Over velocity moments

                    DO l = 1, k_order_nc !Over velocity moments

                        uait(:, j, i) = uait(:, j, i) + xk(l)*rk6s(:, l, i)

                    END DO !Over velocity moments

                END DO !Over isotopes

            END DO !Over species 2

        END DO !Over species 1

        !-------------------------------------------------------------------------------
        !Evaluate currents and fluxes from flows
        !-------------------------------------------------------------------------------
        bsjbps_nc(:) = 0.0_rspec
        bsjbts_nc(:) = 0.0_rspec
        dpss_nc(:, :) = 0.0_rspec
        dtss_nc(:, :) = 0.0_rspec
        chipss_nc(:, :) = 0.0_rspec
        chitss_nc(:, :) = 0.0_rspec

        DO i = 1, ms_nc !Over species 1

            im = jms_nc(i)
            iz = jzs_nc(i)
            iza = IABS(iz)
            !Currents
            !<J_bs.B>
            denzc = den_iz(im, iza)*iz*z_coulomb
            pjbbs_nc = pjbbs_nc + denzc*upars_nc(1, 1, i)
            !<J_OH.B>
            pjboh_nc = pjboh_nc + denzc*upars_nc(1, 2, i)
            !<J_ex.B>
            pjbex_nc = pjbex_nc + denzc*upars_nc(1, 3, i)
            !Unit p'/p and T'/T
            bsjbps_nc(:) = bsjbps_nc(:) + denzc*uaip(1, :, i)
            bsjbts_nc(:) = bsjbts_nc(:) + denzc*uait(1, :, i)
            !Fluxes
            !Banana-Plateau
            cbpa = cbp/iz
            cbpaq = cbpa*(z_j7kv*temp_i(im))
            !Unit p'/p and T'/T
            dpss_nc(i, i) = dpss_nc(i, i) - cbpa*ymus_nc(1, 1, i)*srcthp(i)
            dtss_nc(i, i) = dtss_nc(i, i) - cbpa*ymus_nc(1, 2, i)*srctht(i)
            chipss_nc(i, i) = chipss_nc(i, i) - cbpaq*ymus_nc(2, 1, i)*srcthp(i)
            chitss_nc(i, i) = chitss_nc(i, i) - cbpaq*ymus_nc(2, 2, i)*srctht(i)
            !p' and T'
            gfls_nc(1, i) = gfls_nc(1, i) - cbpa                                         &
                           &               *SUM(ymus_nc(1, :, i)*(upars_nc(:, 1, i) + srcth(:, i)))
            qfls_nc(1, i) = qfls_nc(1, i) - cbpaq                                        &
                           &               *SUM(ymus_nc(2, :, i)*(upars_nc(:, 1, i) + srcth(:, i)))
            !<E.B>
            gfls_nc(4, i) = gfls_nc(4, i) - cbpa                                         &
                           &               *SUM(ymus_nc(1, :, i)*upars_nc(:, 2, i))
            qfls_nc(4, i) = qfls_nc(4, i) - cbpaq                                        &
                           &               *SUM(ymus_nc(2, :, i)*upars_nc(:, 2, i))
            !        write(6,*) 'qfls_nc(4,i)=',qfls_nc(4,i),i
            !      write(6,*) 'flow, upars_nc(1,2,i)=',upars_nc(1,2,i)
            !         External force
            gfls_nc(5, i) = gfls_nc(5, i) - cbpa                                         &
                           &               *SUM(ymus_nc(1, :, i)*upars_nc(:, 3, i))
            qfls_nc(5, i) = qfls_nc(5, i) - cbpaq                                        &
                           &               *SUM(ymus_nc(2, :, i)*upars_nc(:, 3, i))

            !        write(6,*) 'qfls_nc(5,i)=',qfls_nc(5,i)
            DO k = 1, k_order_nc !Over velocity moments

                !Unit p'/p and T'/T
                dpss_nc(:, i) = dpss_nc(:, i) - cbpa*ymus_nc(1, k, i)*uaip(k, :, i)
                dtss_nc(:, i) = dtss_nc(:, i) - cbpa*ymus_nc(1, k, i)*uait(k, :, i)
                chipss_nc(:, i) = chipss_nc(:, i) - cbpaq*ymus_nc(2, k, i)*uaip(k, :, i)
                chitss_nc(:, i) = chitss_nc(:, i) - cbpaq*ymus_nc(2, k, i)*uait(k, :, i)
                !          if (ipr .eq. 2) then
                !          write(*,*) 'i=',i,'k=',k,' ymus_nc(2,k,i)=',ymus_nc(2,k,i)
                !          write(*,*) 'chipss_nc(i,i)=',chipss_nc(i,i)
                !          write(*,*) 'ymus_nc(2,k,i)=',ymus_nc(2,k,i)
                !          if (i .eq. ms_nc) stop
                !          endif

            END DO !Over velocity moments

            !Pfirsch-Schluter and classical
            !Test particle
            cpsa = cps*(xis_nc(i)/iz)
            cpsaq = cpsa*(z_j7kv*temp_i(im))
            ccla = ccl*(xis_nc(i)/iz)
            cclaq = ccla*(z_j7kv*temp_i(im))
            !Pfirsch-Schluter
            gfls_nc(2, i) = gfls_nc(2, i) - cpsa                                        &
                           &               *SUM(calmi_nc(1, :, im)*srcth(:, i))
            qfls_nc(2, i) = qfls_nc(2, i) - cpsaq                                       &
                           &               *SUM(calmi_nc(2, :, im)*srcth(:, i))
            !Classical
            gfls_nc(3, i) = gfls_nc(3, i) + ccla                                        &
                           &               *SUM(calmi_nc(1, :, im)*srcth(:, i))
            qfls_nc(3, i) = qfls_nc(3, i) + cclaq                                       &
                           &               *SUM(calmi_nc(2, :, im)*srcth(:, i))
            !Unit p'/p and T'/T
            dpss_nc(i, i) = dpss_nc(i, i) - (cpsa - ccla)*calmi_nc(1, 1, im)*srcthp(i)
            dtss_nc(i, i) = dtss_nc(i, i) - (cpsa - ccla)*calmi_nc(1, 2, im)*srctht(i)
            !        if (i.eq.1) then
            !             write(*,*) 'dpss_nc(1,1)=',dpss_nc(1,1),cpsa-ccla
            !             write(*,*) 'dtss_nc(1,1)=',dtss_nc(1,1),calmi_nc(1,1,im)
            !             write(*,*) 'calmi_nc(1,2,im)=',calmi_nc(1,2,im),im
            !        endif
            chipss_nc(i, i) = chipss_nc(i, i) - (cpsaq - cclaq)*calmi_nc(2, 1, im)           &
                             &                                *srcthp(i)
            chitss_nc(i, i) = chitss_nc(i, i) - (cpsaq - cclaq)*calmi_nc(2, 2, im)           &
                             &                                *srctht(i)

            !          if (ipr .eq. 2) then
            !          write(*,*) 'i=',i,'chipss_nc(i,i)=',chipss_nc(i,i)
            !          write(*,*) 'i=',i,'chitss_nc(i,i)=',chitss_nc(i,i)
            !          if (i .eq. ms_nc) stop
            !          endif
            !Field particle
            DO j = 1, ms_nc !Over species 2

                jm = jms_nc(j)
                cpsb = cpsa*xis_nc(j)
                cpsbq = cpsb*(z_j7kv*temp_i(im))
                cclb = ccla*xis_nc(j)
                cclbq = cclb*(z_j7kv*temp_i(im))
                !Pfirsch-Schluter
                gfls_nc(2, i) = gfls_nc(2, i) - cpsb                                       &
                               &     *SUM(calnii_nc(1, 1:k_order_nc, im, jm)*srcth(:, j))
                qfls_nc(2, i) = qfls_nc(2, i) - cpsbq                                      &
                               &     *SUM(calnii_nc(2, 1:k_order_nc, im, jm)*srcth(:, j))
                !Classical
                gfls_nc(3, i) = gfls_nc(3, i) + cclb                                       &
                               &     *SUM(calnii_nc(1, 1:k_order_nc, im, jm)*srcth(:, j))
                qfls_nc(3, i) = qfls_nc(3, i) + cclbq                                      &
                               &     *SUM(calnii_nc(2, 1:k_order_nc, im, jm)*srcth(:, j))
                !Unit p'/p and T'/T
                dpss_nc(j, i) = dpss_nc(j, i)                                            &
           &                 - (cpsb - cclb)*calnii_nc(1, 1, im, jm)*srcthp(j)
                dtss_nc(j, i) = dtss_nc(j, i)                                            &
           &                 - (cpsb - cclb)*calnii_nc(1, 2, im, jm)*srctht(j)
                chipss_nc(j, i) = chipss_nc(j, i)                                        &
           &                   - (cpsbq - cclbq)*calnii_nc(2, 1, im, jm)*srcthp(j)
                chitss_nc(j, i) = chitss_nc(j, i)                                        &
           &                   - (cpsbq - cclbq)*calnii_nc(2, 2, im, jm)*srctht(j)

            END DO !Over species 2

        END DO !Over species 1

        !Electrical resistivity
        petap_nc = 1.0_rspec/pjboh_nc
        !      if (ipr .eq. 83) then
        !      write(*,*) 'ipr=',ipr,'petap_nc =',petap_nc,1.0_rspec,pjboh_nc
        !      endif
        !Add <E.B> normalization
        pjboh_nc = p_eb*pjboh_nc
        upars_nc(:, 2, :) = p_eb*upars_nc(:, 2, :)
        qfls_nc(4, :) = p_eb*qfls_nc(4, :)
        gfls_nc(4, :) = p_eb*gfls_nc(4, :)

        !-------------------------------------------------------------------------------
        !Convert full coefficient matrices to diffusivities and conductivities
        !-------------------------------------------------------------------------------
        DO i = 1, ms_nc !Over species

            im = jms_nc(i)
            iza = IABS(jzs_nc(i))
            dpss_nc(:, i) = dpss_nc(:, i)/den_iz(im, iza)
            dtss_nc(:, i) = dtss_nc(:, i)/den_iz(im, iza)
            chipss_nc(:, i) = chipss_nc(:, i)/den_iz(im, iza)/temp_i(im)/z_j7kv
            chitss_nc(:, i) = chitss_nc(:, i)/den_iz(im, iza)/temp_i(im)/z_j7kv

        END DO !Over species

9999    CONTINUE

    END SUBROUTINE NCLASS_FLOW

    SUBROUTINE NCLASS_INIT(m_i, m_z, amu_i, den_iz, iflag)
        !-------------------------------------------------------------------------------
        !NCLASS_INIT initializes the species information and allocates arrays
        !References:
        !  W.A.Houlberg 1/2002
        !Input:
        !  m_i                 -number of isotopes (> 1) [-]
        !  m_z                 -highest charge state [-]
        !  amu_i(i)            -atomic mass number [-]
        !  den_iz(i,z)         -density [/m**3]
        !Output:
        !  iflag               -error and warning flag [-]
        !                      =-1 warning
        !                      =0 no warnings or errors
        !                      =1 error
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        INTEGER, INTENT(IN) ::                                                   &
       & m_i, m_z

        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & amu_i(m_i),                                                               &
       & den_iz(m_i, m_z)

        !Declaration of output variables
        INTEGER, INTENT(OUT) ::                                                  &
       & iflag

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, j, k,                                                                  &
       & nset1, nset2

        !-------------------------------------------------------------------------------
        !Get number of significant isotopes and charge states, and max charge
        !-------------------------------------------------------------------------------
        mi_nc = 0
        mz_nc = 0
        ms_nc = 0

        DO i = 1, m_i !Over isotopes

            DO j = 1, m_z !Over charge states
                IF (den_iz(i, j) >= cden_nc) THEN

                    ms_nc = ms_nc + 1
                    IF (i > mi_nc) mi_nc = i
                    IF (j > mz_nc) mz_nc = j

                END IF

                !         write(*,*) 'i=',i,' j=',j,' m_i=',m_i,' m_z=',m_z
                !         write(*,*) 'ms_nc=',ms_nc,' mi_nc=',mi_nc
                !         write(*,*) 'cden_nc =',cden_nc,' den_iz=',den_iz(i,j)

            END DO !Over charge states

        END DO !Over isotopes

        !At least two species
        IF (mi_nc < 2) THEN

            iflag = 1
            write (*, *) 'NCLASS_INIT(1)/ERROR:m_i must be >= 2'
            write (*, *) 'mi_nc =', mi_nc
            GOTO 9999

        END IF

        !Highest charge state at least 1
        IF (mz_nc < 1) THEN

            iflag = 1
            write (*, *) 'NCLASS_INIT(2)/ERROR:m_z must be >= 1'
            GOTO 9999

        END IF

        !-------------------------------------------------------------------------------
        !Allocate or reallocate isotope info
        !-------------------------------------------------------------------------------
        IF (ALLOCATED(calmi_nc)) THEN

            nset1 = SIZE(calmi_nc, 1)
            nset2 = SIZE(calmi_nc, 3)

            !If storage requirements have changed, reallocate
            IF (nset1 /= k_order_nc .OR. nset2 /= mi_nc) THEN

                !Reallocate
                DEALLOCATE (vti_nc,                                                   &
           &               amntii_nc,                                                &
           &               calmi_nc,                                                 &
           &               calnii_nc,                                                &
           &               capmii_nc,                                                &
           &               capnii_nc)
                ALLOCATE (vti_nc(1:mi_nc),                                            &
           &             amntii_nc(1:mi_nc, 1:mi_nc),                                 &
           &             calmi_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc),                &
           &             calnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc),       &
           &             capmii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc),       &
           &             capnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc))

            END IF

        ELSE

            !Allocate
            ALLOCATE (vti_nc(1:mi_nc),                                              &
         &           amntii_nc(1:mi_nc, 1:mi_nc),                                   &
         &           calmi_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc),                  &
         &           calnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc),         &
         &           capmii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc),         &
         &           capnii_nc(1:k_order_nc, 1:k_order_nc, 1:mi_nc, 1:mi_nc))

        END IF

        !-------------------------------------------------------------------------------
        !Allocate or reallocate isotope/charge state info
        !-------------------------------------------------------------------------------
        IF (ALLOCATED(grppiz_nc)) THEN

            nset1 = SIZE(grppiz_nc, 1)
            nset2 = SIZE(grppiz_nc, 2)

            !If storage requirements have changed, reallocate
            IF (nset1 /= mi_nc .OR. nset2 /= mz_nc) THEN

                !Reallocate
                DEALLOCATE (grppiz_nc)
                ALLOCATE (grppiz_nc(1:mi_nc, 1:mz_nc))

            END IF

        ELSE

            !Allocate
            ALLOCATE (grppiz_nc(1:mi_nc, 1:mz_nc))

        END IF

        !-------------------------------------------------------------------------------
        !Allocate or reallocate species info
        !-------------------------------------------------------------------------------
        IF (ALLOCATED(jms_nc)) THEN

            nset1 = SIZE(jms_nc)

            !If storage requirements have changed, reallocate
            IF (nset1 /= ms_nc) THEN

                !Reallocate
                DEALLOCATE (jms_nc,                                                   &
           &               jzs_nc,                                                   &
           &               sqzs_nc,                                                  &
           &               xis_nc,                                                   &
           &               bsjbps_nc,                                                &
           &               bsjbts_nc,                                                &
           &               gfls_nc,                                                  &
           &               dpss_nc,                                                  &
           &               dtss_nc,                                                  &
           &               qfls_nc,                                                  &
           &               chipss_nc,                                                &
           &               chitss_nc,                                                &
           &               tauss_nc)

                ALLOCATE (jms_nc(1:ms_nc),                                            &
           &             jzs_nc(1:ms_nc),                                            &
           &             sqzs_nc(1:ms_nc),                                           &
           &             xis_nc(1:ms_nc),                                            &
           &             bsjbps_nc(1:ms_nc),                                         &
           &             bsjbts_nc(1:ms_nc),                                         &
           &             gfls_nc(5, 1:ms_nc),                                         &
           &             dpss_nc(1:ms_nc, 1:ms_nc),                                   &
           &             dtss_nc(1:ms_nc, 1:ms_nc),                                   &
           &             qfls_nc(5, 1:ms_nc),                                         &
           &             chipss_nc(1:ms_nc, 1:ms_nc),                                 &
           &             chitss_nc(1:ms_nc, 1:ms_nc),                                 &
           &             tauss_nc(1:ms_nc, 1:ms_nc))

            END IF

        ELSE

            !Allocate
            ALLOCATE (jms_nc(1:ms_nc),                                              &
         &           jzs_nc(1:ms_nc),                                              &
         &           sqzs_nc(1:ms_nc),                                             &
         &           xis_nc(1:ms_nc),                                              &
         &           bsjbps_nc(1:ms_nc),                                           &
         &           bsjbts_nc(1:ms_nc),                                           &
         &           gfls_nc(5, 1:ms_nc),                                           &
         &           dpss_nc(1:ms_nc, 1:ms_nc),                                     &
         &           dtss_nc(1:ms_nc, 1:ms_nc),                                     &
         &           qfls_nc(5, 1:ms_nc),                                           &
         &           chipss_nc(1:ms_nc, 1:ms_nc),                                   &
         &           chitss_nc(1:ms_nc, 1:ms_nc),                                   &
         &           tauss_nc(1:ms_nc, 1:ms_nc))

        END IF

        !-------------------------------------------------------------------------------
        !Allocate or reallocate species/moments info
        !-------------------------------------------------------------------------------
        IF (ALLOCATED(ymus_nc)) THEN

            nset1 = SIZE(ymus_nc, 1)
            nset2 = SIZE(ymus_nc, 3)

            !If storage requirements have changed, reallocate
            IF (nset1 /= ms_nc .OR. nset2 /= k_order_nc) THEN

                !Reallocate
                DEALLOCATE (ymus_nc,                                                  &
           &               upars_nc,                                                 &
           &               uthetas_nc)

                ALLOCATE (ymus_nc(1:k_order_nc, 1:k_order_nc, 1:ms_nc),                 &
           &             upars_nc(1:k_order_nc, 1:5, 1:ms_nc),                         &
           &             uthetas_nc(1:k_order_nc, 1:5, 1:ms_nc))

            END IF

        ELSE

            !Allocate
            ALLOCATE (ymus_nc(1:k_order_nc, 1:k_order_nc, 1:ms_nc),                   &
         &           upars_nc(1:k_order_nc, 1:5, 1:ms_nc),                           &
         &           uthetas_nc(1:k_order_nc, 1:5, 1:ms_nc))

        END IF

        !-------------------------------------------------------------------------------
        !Find significant charge states and mapping
        !-------------------------------------------------------------------------------
        k = 0
        imel_nc = 0

        DO i = 1, mi_nc !Over isotopes

            DO j = 1, mz_nc !Over charge states

                IF (den_iz(i, j) >= cden_nc) THEN

                    !           Set isotope number and charge state for this species
                    k = k + 1
                    jms_nc(k) = i

                    IF (amu_i(i) < 0.5) THEN

                        !Electrons
                        jzs_nc(k) = -j
                        imel_nc = i

                    ELSE

                        !Ions
                        jzs_nc(k) = j

                    END IF

                END IF

            END DO !Over charge states

        END DO !Over isotopes

9999    CONTINUE

    END SUBROUTINE NCLASS_INIT

    SUBROUTINE NCLASS_K(p_fm, p_ft, p_ngrth, x, amu_i, temp_i, ykb_s, ykp_s,        &
   &                    ykpo_s, ykpop_s)
        !-------------------------------------------------------------------------------
        !NCLASS_K calculates the velocity-dependent neoclassical viscosity coefficients
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  C.E.Kessel, Nucl Fusion 34 (1994) 1221
        !  K.C.Shaing, M.Yokoyama, M.Wakatani, C.T.Hsu, Phys Plasmas 3 (1996) 965
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  p_fm(:)             -poloidal moments of drift factor for PS [/m**2]
        !  p_ft                -trapped fraction [-]
        !  p_ngrth             -<n.grad(Theta)> [/m]
        !  x                   -normalized velocity v/(2kT/m)**0.5 [-]
        !  amu_i(i)            -atomic mass number [-]
        !  temp_i(i)           -temperature [keV]
        !Output:
        !  ykb_s(s)            -banana viscosity for s [/s]
        !  ykp_s(s)            -Pfirsch-Schluter viscosity for s [/s]
        !  ykpo_s(s)           -potato viscosity for s [/s]
        !  ykpop_s(s)          -potato-plateau viscosity for s [/s]
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & p_fm(:), p_ft, p_ngrth, x,                                                 &
       & amu_i(:), temp_i(:)

        !Declaration of output variables
        REAL(KIND=rspec) ::                                                      &
       & ykb_s(:), ykp_s(:), ykpo_s(:), ykpop_s(:)

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, im, iz

        REAL(KIND=rspec) ::                                                      &
       & c1, c2, c3, c4

        REAL(KIND=rspec) ::                                                      &
       & ynud_s(ms_nc), ynut_s(ms_nc), ynutis(mf_nc, ms_nc)

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        ykb_s(:) = 0.0_rspec
        ykp_s(:) = 0.0_rspec
        ykpo_s(:) = 0.0_rspec
        ykpop_s(:) = 0.0_rspec

        !Get collisional frequencies
        CALL NCLASS_NU(p_ngrth, x, temp_i, ynud_s, ynut_s, ynutis)

        !-------------------------------------------------------------------------------
        !Set velocity dependent viscosities (K's)
        !-------------------------------------------------------------------------------
        c1 = 1.5_rspec*x**2
        c2 = 3.0_rspec*2.19_rspec/(2.0_rspec**1.5)*x**(1.0/3.0)

        IF (l_potato_nc) c3 = 3.0_rspec*z_pi/(64.0_rspec*2.0_rspec**0.33333)        &
       &                   /ABS(cpotl_nc)

        DO i = 1, ms_nc !Over species

            im = jms_nc(i)
            iz = jzs_nc(i)

            !Banana
            IF (l_banana_nc) THEN

                !Provide cutoff to eliminate failure at unity trapped fraction
                !At A=>1 viscosity will go over to Pfirsch-Schluter value
                c4 = 1.0_rspec - p_ft
                IF (c4 < 1.0e-3_rspec) c4 = 1.0e-3_rspec
                ykb_s(i) = p_ft/c4/sqzs_nc(i)**1.5*ynud_s(i)

            END IF

            !Pfirsch-Schuter
            IF (l_pfirsch_nc) THEN

                ykp_s(i) = ykp_s(i) + c1*vti_nc(im)**2                                   &
                          &                      *SUM(p_fm(:)*ynutis(:, i))/ynut_s(i)

            END IF

            !Potato
            IF (l_potato_nc) THEN

                c4 = ABS(amu_i(im)*z_protonmass*vti_nc(im)                             &
           &       /(iz*z_coulomb*cpotb_nc*cpotl_nc))
                ykpo_s(i) = c2*c4**(0.333333_rspec)*ynud_s(i)                          &
           &              /sqzs_nc(i)**(1.666667_rspec)
                ykpop_s(i) = c3*vti_nc(im)*c4**(1.333333_rspec)

            END IF

        END DO !Over species

    END SUBROUTINE NCLASS_K

    SUBROUTINE NCLASS_MN(amu_i, temp_i)
        !-------------------------------------------------------------------------------
        !NCLASS_MN calculates neoclassical friction coefficients
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  S.P.Hirshman, Phys Fluids 20 (1977) 589
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  amu_i(i)            -atomic mass number [-]
        !  temp_i(i)           -temperature [keV]
        !Comments:
        !  The k_order*korder matrix of test particle (M) and field particle (N)
        !    coefficients of the collision operator use the Laguerre polynomials of
        !    order 3/2 as basis functions for each isotopic combination
        !  The indices on the M and N matrices are one greater than the notation in the
        !    H&S review article so as to avoid 0 as an index
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & amu_i(:), temp_i(:)

        !Declaration of local variables
        INTEGER ::                                                               &
       & im, jm

        REAL(KIND=rspec) ::                                                      &
       & xab, xab2, xmab, xtab, yab32, yab52, yab72, yab92

        !-------------------------------------------------------------------------------
        !Calculate friction coefficients
        !-------------------------------------------------------------------------------
        DO im = 1, mi_nc !Over isotopes a

            DO jm = 1, mi_nc !Over isotopes b

                !Mass ratio
                xmab = amu_i(im)/amu_i(jm)
                !Temperature ratio
                xtab = temp_i(im)/temp_i(jm)
                !Thermal velocity ratio, vtb/vta
                xab = SQRT(xmab/xtab)

                xab2 = xab**2
                yab32 = (1.0_rspec + xab2)*SQRT(1.0_rspec + xab2)
                yab52 = (1.0_rspec + xab2)*yab32

                IF (k_order_nc == 3) THEN

                    yab72 = (1.0_rspec + xab2)*yab52
                    yab92 = (1.0_rspec + xab2)*yab72

                END IF

                !Test particle coefficients, M
                !Eqn 4.11 (HS81) for M00
                capmii_nc(1, 1, im, jm) = -(1.0_rspec + xmab)/yab32
                !Eqn 4.12 (HS81) for M01
                capmii_nc(1, 2, im, jm) = 1.5_rspec*(1.0_rspec + xmab)/yab52
                !Eqn 4.8 (HS81) for M10
                capmii_nc(2, 1, im, jm) = capmii_nc(1, 2, im, jm)
                !Eqn 4.13 (HS81) for M11
                capmii_nc(2, 2, im, jm) = -(3.25_rspec + xab2*(4.0_rspec + 7.5_rspec*xab2))/yab52

                IF (k_order_nc == 3) THEN

                    !Eqn 4.15 (HS81) for M02
                    capmii_nc(1, 3, im, jm) = -1.875_rspec*(1.0_rspec + xmab)/yab72
                    !Eqn 4.16 (HS81) for M12
                    capmii_nc(2, 3, im, jm) = (4.3125_rspec + xab2*(6.0_rspec + 15.75_rspec*xab2))/yab72
                    !Eqn 4.8 (HS81) for M20
                    capmii_nc(3, 1, im, jm) = capmii_nc(1, 3, im, jm)
                    !Eqn 4.8 (HS81) for M21
                    capmii_nc(3, 2, im, jm) = capmii_nc(2, 3, im, jm)
                    !Eqn 5.21 (HS81) for M22
                    capmii_nc(3, 3, im, jm) = -(433.0/64.0 + xab2*(17.0_rspec +&
                    &  xab2*(57.375_rspec + xab2*(28.0_rspec + xab2*21.875_rspec))))/yab92

                END IF

                !Field particle coefficients, N
                !Momentum conservation, Eqn 4.11 (HS81) for N00
                capnii_nc(1, 1, im, jm) = -capmii_nc(1, 1, im, jm)
                !Eqn 4.9 and 4.12 (HS81) for N01
                capnii_nc(1, 2, im, jm) = -xab2*capmii_nc(1, 2, im, jm)
                !Momentum conservation, Eqn 4.12 (HS81) for N10
                capnii_nc(2, 1, im, jm) = -capmii_nc(2, 1, im, jm)
                !Eqn 4.14 (HS81) for N11  - corrected rhs
                capnii_nc(2, 2, im, jm) = 6.75_rspec*SQRT(xtab)*xab2/yab52

                IF (k_order_nc == 3) THEN

                    !Eqn 4.15 for N02 (HS81) - corrected rhs by Ta/Tb
                    capnii_nc(1, 3, im, jm) = -xab2**2*capmii_nc(1, 3, im, jm)
                    !Eqn 4.17 for N12 (HS81)
                    capnii_nc(2, 3, im, jm) = -14.0625_rspec*xtab*xab2**2/yab72
                    !Momentum conservation for N20 (HS81)
                    capnii_nc(3, 1, im, jm) = -capmii_nc(3, 1, im, jm)
                    !Eqn 4.9 and 4.17 for N21 (HS81)
                    capnii_nc(3, 2, im, jm) = -14.0625_rspec*xab2**2/yab72
                    !Eqn 5.22 for N22 (HS81)
                    capnii_nc(3, 3, im, jm) = 2625.0/64.0*xtab*xab2**2/yab92

                END IF

            END DO !Over isotopes b

        END DO !Over isotopes a

    END SUBROUTINE NCLASS_MN

    SUBROUTINE NCLASS_MU(p_fm, p_ft, p_ngrth, amu_i, temp_i, den_iz, ipr)
        !-------------------------------------------------------------------------------
        !NCLASS_MU calculates the matrix of neoclassical fluid viscosities
        !References:
        !  K.C.Shaing, M.Yokoyama, M.Wakatani, C.T.Hsu, Phys Plasmas 3 (1996) 965
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  p_fm(:)             -poloidal moments of drift factor for PS [/m**2]
        !  p_ft                -trapped fraction [-]
        !  p_ngrth             -<n.grad(Theta)> [/m]
        !  amu_i(i)            -atomic mass number [-]
        !  temp_i(i)           -temperature [keV]
        !  den_iz(i,z)         -density [/m**3]
        !Comments:
        !  Integrates the velocity-dependent banana and Pfirsch-Schluter contributions
        !    over velocity space
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & p_ft, p_ngrth, p_fm(:),                                                   &
       & amu_i(:), temp_i(:),                                                     &
       & den_iz(:, :)

        !Declaration of local variables
        !Parameters
        INTEGER, PARAMETER ::                                                    &
       & mpnts = 13

        REAL(KIND=rspec), PARAMETER ::                                           &
       & bmax = 3.2

        !Saved
        INTEGER, SAVE ::                                                         &
       & init = 0

        REAL(KIND=rspec), SAVE ::                                                &
       & c1, h,                                                                   &
       & x(mpnts), w(mpnts, 5)

        !Other
        INTEGER ::                                                               &
       & i, im, iza, k, l, m, ipr

        REAL(KIND=rspec) ::                                                      &
       & c2, ewt, expmx2, x2, x4, x6, x8, x10, x12, xk, xx,                                &
       & dum(3),                                                                 &
       & ykb_s(ms_nc), ykp_s(ms_nc), ykpo_s(ms_nc), ykpop_s(ms_nc),                 &
       & ymubs(1:k_order_nc, 1:k_order_nc, ms_nc),                                 &
       & ymubps(1:k_order_nc, 1:k_order_nc, ms_nc),                                &
       & ymupps(1:k_order_nc, 1:k_order_nc, ms_nc),                                &
       & ymupos(1:k_order_nc, 1:k_order_nc, ms_nc)

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        IF (init == 0) THEN

            !Set integration points and weights
            h = bmax/(mpnts - 1)
            x(1) = 0.0_rspec
            w(:, :) = 0.0_rspec

            DO m = 2, mpnts !Over velocity nodes for integration

                x(m) = h*(m - 1)
                x2 = x(m)*x(m)
                expmx2 = EXP(-x2)
                x4 = x2*x2
                w(m, 1) = x4*expmx2
                x6 = x4*x2
                w(m, 2) = x6*expmx2
                x8 = x4*x4
                w(m, 3) = x8*expmx2
                x10 = x4*x6
                w(m, 4) = x10*expmx2
                x12 = x6*x6
                w(m, 5) = x12*expmx2

            END DO !Over velocity nodes for integration

            c1 = 8.0_rspec/3.0_rspec/SQRT(z_pi)*h
            init = 1

        END IF

        !Null out viscosity arrays
        ymus_nc(:, :, :) = 0.0_rspec
        ymubs(:, :, :) = 0.0_rspec
        ymubps(:, :, :) = 0.0_rspec
        ymupos(:, :, :) = 0.0_rspec
        ymupps(:, :, :) = 0.0_rspec

        !-------------------------------------------------------------------------------
        !Integrate over velocity space for fluid viscosities
        !-------------------------------------------------------------------------------
        IF (l_banana_nc .OR. l_pfirsch_nc) THEN

            DO m = 2, mpnts !Over velocity grid

                IF (m == mpnts) THEN

                    !Use half weight for end point
                    ewt = 0.5_rspec

                ELSE

                    !Use full weight for intervening points
                    ewt = 1.0_rspec

                END IF

                xx = x(m)
                !Get velocity-dependent k values
                CALL NCLASS_K(p_fm, p_ft, p_ngrth, xx, amu_i, temp_i, ykb_s, ykp_s,         &
           &                  ykpo_s, ykpop_s)

                DO i = 1, ms_nc !Over species

                    im = jms_nc(i)
                    iza = IABS(jzs_nc(i))
                    c2 = c1*ewt*den_iz(im, iza)*amu_i(im)*z_protonmass
                    dum(1) = c2*w(m, 1)
                    dum(2) = c2*(w(m, 2) - 2.5_rspec*w(m, 1))
                    dum(3) = c2*(w(m, 3) - 5.0_rspec*w(m, 2) + 6.25_rspec*w(m, 1))

                    IF (.NOT. l_banana_nc) THEN

                        !Only Pfirsch-Schluter
                        xk = ykp_s(i)

                    ELSEIF (.NOT. l_pfirsch_nc) THEN

                        !Only banana
                        xk = ykb_s(i)

                    ELSE

                        !Both banana and Pfirsch-Schluter
                        xk = ykb_s(i)*ykp_s(i)/(ykb_s(i) + ykp_s(i))

                    END IF

                    ymubs(1, 1, i) = ymubs(1, 1, i) + ykb_s(i)*dum(1)
                    ymubs(1, 2, i) = ymubs(1, 2, i) + ykb_s(i)*dum(2)
                    ymubs(2, 2, i) = ymubs(2, 2, i) + ykb_s(i)*dum(3)
                    ymubps(1, 1, i) = ymubps(1, 1, i) + xk*dum(1)
                    ymubps(1, 2, i) = ymubps(1, 2, i) + xk*dum(2)
                    ymubps(2, 2, i) = ymubps(2, 2, i) + xk*dum(3)

                    IF (l_potato_nc) THEN

                        ymupos(1, 1, i) = ymupos(1, 1, i) + ykpo_s(i)*dum(1)
                        ymupos(1, 2, i) = ymupos(1, 2, i) + ykpo_s(i)*dum(2)
                        ymupos(2, 2, i) = ymupos(2, 2, i) + ykpo_s(i)*dum(3)
                        xk = ykpo_s(i)*ykpop_s(i)/(ykpo_s(i) + ykpop_s(i))
                        ymupps(1, 1, i) = ymupps(1, 1, i) + xk*dum(1)
                        ymupps(1, 2, i) = ymupps(1, 2, i) + xk*dum(2)
                        ymupps(2, 2, i) = ymupps(2, 2, i) + xk*dum(3)

                    END IF

                    IF (k_order_nc == 3) THEN

                        dum(1) = c2*(0.5_rspec*w(m, 3) - 3.5_rspec*w(m, 2)                     &
               &               + 4.375_rspec*w(m, 1))
                        dum(2) = c2*(0.5_rspec*w(m, 4) - 4.75_rspec*w(m, 3)                    &
               &               + 13.125_rspec*w(m, 2) - 10.9375_rspec*w(m, 1))
                        dum(3) = c2*(0.25_rspec*w(m, 5) - 3.5_rspec*w(m, 4)                    &
               &               + 16.625_rspec*w(m, 3) - 30.625*w(m, 2)                        &
               &               + (1225.0/64.0)*w(m, 1))
                        ymubs(1, 3, i) = ymubs(1, 3, i) + ykb_s(i)*dum(1)
                        ymubs(2, 3, i) = ymubs(2, 3, i) + ykb_s(i)*dum(2)
                        ymubs(3, 3, i) = ymubs(3, 3, i) + ykb_s(i)*dum(3)
                        xk = ykb_s(i)*ykp_s(i)/(ykb_s(i) + ykp_s(i))
                        ymubps(1, 3, i) = ymubps(1, 3, i) + xk*dum(1)
                        ymubps(2, 3, i) = ymubps(2, 3, i) + xk*dum(2)
                        ymubps(3, 3, i) = ymubps(3, 3, i) + xk*dum(3)

                        IF (l_potato_nc) THEN

                            ymupos(1, 3, i) = ymupos(1, 3, i) + ykpo_s(i)*dum(1)
                            ymupos(2, 3, i) = ymupos(2, 3, i) + ykpo_s(i)*dum(2)
                            ymupos(3, 3, i) = ymupos(3, 3, i) + ykpo_s(i)*dum(3)
                            xk = ykpo_s(i)*ykpop_s(i)/(ykpo_s(i) + ykpop_s(i))
                            ymupps(1, 3, i) = ymupps(1, 3, i) + xk*dum(1)
                            ymupps(2, 3, i) = ymupps(2, 3, i) + xk*dum(2)
                            ymupps(3, 3, i) = ymupps(3, 3, i) + xk*dum(3)

                        END IF

                    END IF

                END DO  !Over species

            END DO !Over velocity grid

            !Load net viscosity
            DO i = 1, ms_nc !Over species

                DO l = 1, k_order_nc !Over rows

                    DO k = 1, l !Over columns in upper part of matrix

                        IF (l_potato_nc) THEN

                            !Banana Pfirsch-Schluter plus potato potato-plateau
                            ymus_nc(k, l, i) = (ymupos(k, l, i)**3*ymupps(k, l, i)                 &
                 &                         + ymubs(k, l, i)**3*ymubps(k, l, i))                 &
                 &                         /(ymupos(k, l, i)**3 + ymubs(k, l, i)**3)
                        ELSE

                            !Banana Pfirsch-Schluter
                            ymus_nc(k, l, i) = ymubps(k, l, i)

                        END IF
                        !              if (ipr .eq. 2 .and. k .eq. 2) then
                        !                write(*,*) 'ymubps(2,l,i)=',ymubps(k,l,i),l,i
                        !                write(*,*) 'ymupos(2,l,i)=',ymupos(k,l,i)
                        !                write(*,*) 'ymupps(2,l,i)=',ymupps(k,l,i)
                        !                write(*,*) 'ymubs(2,l,i)=',ymubs(k,l,i),l_potato_nc
                        !                stop
                        !              endif

                    END DO !Over columns in upper part of matrix

                END DO !Over rows

            END DO !Over species

            !Fill viscosity matrix using symmetry
            DO i = 1, ms_nc !Over species

                DO l = 1, k_order_nc - 1 !Over rows

                    DO k = l + 1, k_order_nc !Over columns

                        ymus_nc(k, l, i) = ymus_nc(l, k, i)

                    END DO !Over columns

                END DO !Over rows

            END DO !Over species

        END IF

    END SUBROUTINE NCLASS_MU

    SUBROUTINE NCLASS_NU(p_ngrth, x, temp_i, ynud_s, ynut_s, ynutis)
        !-------------------------------------------------------------------------------
        !NCLASS_NU calculates the velocity dependent pitch angle diffusion and
        !  anisotropy relaxation rates, nu_D, nu_T, and nu_T*I_Rm
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Phys Fluids 19 (1976) 1532
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  K.C.Shaing, M.Yokoyama, M.Wakatani, C.T.Hsu, Phys Plasmas 3 (1996) 965
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  p_ngrth             -<n.grad(Theta)> [/m]
        !  x                   -normalized velocity v/(2kT/m)**0.5 [-]
        !  temp_i(i)           -temperature [keV]
        !Output:
        !  ynud_s(s)           -pitch angle diffusion rate for s [/s]
        !  ynut_s(s)           -anisotropy relaxation rate for s [/s]
        !  ynutis(:,s)         -PS anisotropy relaxation rates for s [-]
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & p_ngrth, x,                                                              &
       & temp_i(:)

        !Declaration of output variables
        REAL(KIND=rspec) ::                                                      &
       & ynud_s(:), ynut_s(:), ynutis(:, :)

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, im, j, jm, m

        REAL(KIND=rspec) ::                                                      &
       & c1, c2, c3, g, phi

        !-------------------------------------------------------------------------------
        !Initializaton
        !-------------------------------------------------------------------------------
        ynud_s(:) = 0.0_rspec
        ynut_s(:) = 0.0_rspec
        ynutis(:, :) = 0.0_rspec

        !-------------------------------------------------------------------------------
        !Collision rates
        !-------------------------------------------------------------------------------
        DO i = 1, ms_nc !Over species a

            im = jms_nc(i)

            !nu_D and nu_T
            DO j = 1, ms_nc !Over species b

                jm = jms_nc(j)
                c1 = vti_nc(jm)/vti_nc(im)
                c2 = x/c1
                phi = NCLASS_ERF(c2)
                g = (phi - c2*(2.0_rspec/SQRT(z_pi))*EXP(-c2**2))                        &
           &      /(2.0_rspec*c2**2)
                ynud_s(i) = ynud_s(i) + (3.0_rspec*SQRT(z_pi)/4.0_rspec)                 &
                           &                       *(phi - g)/x**3/tauss_nc(i, j)
                ynut_s(i) = ynut_s(i) + ((3.0_rspec*SQRT(z_pi)/4.0_rspec)                &
           &                       *((phi - 3.0_rspec*g)/x**3                          &
           &                       + 4.0_rspec*(temp_i(im)/temp_i(jm)                 &
           &                       + 1.0_rspec/c1**2)*g/x))/tauss_nc(i, j)

            END DO !Over species b

            !nu_T*I_m
            DO m = 1, mf_nc !Over poloidal moments

                IF (ABS(p_ngrth) > 0.0_rspec) THEN

                    c1 = x*vti_nc(im)*m*p_ngrth
                    c2 = (ynut_s(i)/c1)**2

                    IF (c2 > 9.0_rspec) THEN

                        !Use asymptotic limit for efficiency
                        ynutis(m, i) = 0.4_rspec

                    ELSE

                        !Use full calculation
                        c3 = ynut_s(i)/c1*ATAN(c1/ynut_s(i))
                        ynutis(m, i) = 0.5_rspec*c3 + c2*(3.0_rspec*(c3 - 0.5_rspec)            &
               &                                     + c2*4.5_rspec*(c3 - 1.0_rspec))

                    END IF

                ELSE

                    ynutis(m, i) = 0.4_rspec

                END IF

            END DO !Over poloidal moments

        END DO !Over species a

    END SUBROUTINE NCLASS_NU

    SUBROUTINE NCLASS_TAU(amu_i, temp_i, den_iz)
        !-------------------------------------------------------------------------------
        !NCLASS_TAU calculates the collision times for 90 degree scattering and the
        !  effective collision rates
        !References:
        !  S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
        !  W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)
        !    3230
        !  W.A.Houlberg 1/2002
        !Input:
        !  amu_i(i)            -atomic mass number of i [-]
        !  temp_i(i)           -temperature of i [keV]
        !  den_iz(i,z)         -density of i,z [/m**3]
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & amu_i(:), temp_i(:),                                                     &
       & den_iz(:, :)

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, im, iz, iza, j, jm, jz, jza

        REAL(KIND=rspec) ::                                                      &
       & c1, c2, cln

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        !Arrays
        amntii_nc(:, :) = 0.0_rspec
        tauss_nc(:, :) = 0.0_rspec

        !Use electron Coulomb logarithm for all species
        cln = 37.8_rspec - LOG(SQRT(den_iz(imel_nc, 1))/temp_i(imel_nc))

        !-------------------------------------------------------------------------------
        !Collision times and mass density weighted collision rates
        !-------------------------------------------------------------------------------
        c1 = 4.0_rspec/3.0_rspec/SQRT(z_pi)*4.0_rspec*z_pi*cln*(z_coulomb          &
       &   /(4.0_rspec*z_pi*z_epsilon0))**2*(z_coulomb/z_protonmass)**2
        !      write(*,*) 'cln=',cln,imel_nc,'den=',den_iz(imel_nc,1)
        DO i = 1, ms_nc

            im = jms_nc(i)
            iz = jzs_nc(i)
            iza = IABS(iz)
            c2 = (vti_nc(im)**3)*amu_i(im)**2/c1
            if (im .eq. 12) then
                !           write(*,*) 'im=',im,jm,iza,iz
                !           write(*,*) 'c1=',c1,amu_i(im)
            end if

            DO j = 1, ms_nc

                jm = jms_nc(j)
                jz = jzs_nc(j)
                jza = ABS(jz)
                tauss_nc(i, j) = c2/iz**2/(den_iz(jm, jza)*jz**2)
                amntii_nc(im, jm) = amntii_nc(im, jm) + amu_i(im)*z_protonmass             &
                                   &                     *den_iz(im, iza)/tauss_nc(i, j)

            END DO

        END DO

    END SUBROUTINE NCLASS_TAU

    SUBROUTINE NCLASS_BACKSUB(a, n, indx, b, ipr)
        !-------------------------------------------------------------------------------
        !NCLASS_BACKSUB solves the matrix equation a x = b, where a is in LU form as
        !  generated by a prior call to NCLASS_DECOMP
        !References:
        !  Flannery, Teukolsky, Vetterling, Numerical Recipes
        !  W.A.Houlberg 1/2002
        !Input:
        !  a(n,n)              -coefficient matrix in lu decomposed form [-]
        !  n                   -number of equations to be solved [-]
        !  indx(n)             -row permutations due to partial pivoting [-]
        !Input/output:
        !  b                   -(input) right hand side of equation [-]
        !                      -(output) solution vector [-]
        !Comments:
        !  The complete procedure is, given the equation a*x = b,
        !    CALL NCLASS_DECOMP( )
        !    CALL NCLASS_BACKSUB( )
        !    and b is now the solution vector
        !  To solve with a different right hand side, just reload b as  desired
        !    and use the same LU decomposition
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        INTEGER, INTENT(IN) ::                                                   &
       & n,                                                                      &
       & indx(:)

        REAL(KIND=rspec), INTENT(IN) ::                                          &
       & a(:, :)

        !Declaration of input/output variables
        REAL(KIND=rspec), INTENT(INOUT) ::                                       &
       & b(:)

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, ii, j, k, ipr

        REAL(KIND=rspec) ::                                                      &
       & sumhoul

        !-------------------------------------------------------------------------------
        !Find the index of the first nonzero element of b
        !-------------------------------------------------------------------------------
        ii = 0
        sumhoul = 0.0_rspec
        !      if (n .eq. 18) then
        !        write(*,*) 'n=',n,indx
        !      endif
        DO i = 1, n

            k = indx(i)
            sumhoul = b(k)
            b(k) = b(i)
            !        if (n .eq. 3 .and. ipr .eq. 1) then
            !        write(*,*) 'dans backsub, k=',k,' b(k)=',b(k),' i=',i
            !        write(*,*) 'ii=',ii,' sumhoul=',sumhoul
            !        endif
            IF (ii /= 0) THEN

                if (i .gt. 1) then
                    DO j = ii, i - 1
                        sumhoul = sumhoul - a(i, j)*b(j)
                    END DO
                end if

                !        ELSEIF (sumhoul /= 0.0_rspec) THEN
            ELSEIF (sumhoul /= 0.0) THEN

                ii = i

            END IF

            b(i) = sumhoul
            !          write(*,*) 'i=',i,'b(i)=',b(i)
            !          write(*,*) 'a(i,i)=',a(i,i)
            !        endif

        END DO
        !      if (iecrit .eq. 1 .and. iindice .eq. 2) then
        !        write(*,*) 'rspec=',rspec,'sumhoul=',sumhoul
        !        write(*,*) 'iecrit=',iecrit
        !        stop
        !      endif

        !-------------------------------------------------------------------------------
        !Back substitution
        !-------------------------------------------------------------------------------
        DO i = n, 1, -1

            sumhoul = b(i)

            IF (i < n) THEN

                DO j = i + 1, n

                    sumhoul = sumhoul - a(i, j)*b(j)

                END DO

            END IF

            b(i) = sumhoul/a(i, i)
            !        if (n .eq. 3 .and. ipr .eq. 1) then
            !          write(*,*) 'i=',i,'b(i)=',b(i)
            !          write(*,*) 'a(i,i)=',a(i,i)
            !        endif

        END DO
        !      write(*,*) 'rspec=',rspec,'sumhoul=',sumhoul
        !      stop
    END SUBROUTINE NCLASS_BACKSUB
    SUBROUTINE NCLASS_DECOMP(a, n, indx, d, iflag, ipr)
        !-------------------------------------------------------------------------------
        !NCLASS_DECOMP performs an LU decomposition of the matrix a and is called
        !  prior to NCLASS_BACKSUB to solve linear equations or to invert a matrix
        !References:
        !  Flannery, Teukolsky, Vetterling, Numerical Recipes
        !  W.A.Houlberg 1/2002
        !Input:
        !  n                   -number of equations to be solved [-]
        !Input/output:
        !  a(n,n)              -coefficient matrix, overwritten on return [-]
        !Output:
        !  indx(n)             -row permutations due to partial pivoting [-]
        !  d                   -flag for number of row exchanges [-]
        !                      =1.0 even number
        !                      =-1.0 odd number
        !  iflag               -error and warning flag [-]
        !                      =-1 warning
        !                      =0 no warnings or errors
        !                      =1 error
        !-------------------------------------------------------------------------------

        !Declaration of input variables
        INTEGER, INTENT(IN) ::                                                   &
       & n

        !Declaration of input/output variables
        REAL(KIND=rspec), INTENT(INOUT) ::                                       &
       & a(:, :)

        !Declaration of output variables
        INTEGER, INTENT(OUT) ::                                                  &
       & iflag,                                                                  &
       & indx(:)

        REAL(KIND=rspec) ::  alp(n, n)
        REAL(KIND=rspec), INTENT(OUT) ::                                         &
       & d

        !Declaration of local variables
        INTEGER ::                                                               &
       & i, imax, j, k, ik, ipr

        REAL(KIND=rspec) ::                                                      &
       & aamax, dum, sumhoul,                                                          &
       & vv(n), varia, sumhoult, precis, precis2

        !-------------------------------------------------------------------------------
        !Initialization
        !-------------------------------------------------------------------------------
        d = 1.0_rspec
        sumhoul = 0.0_rspec
        precis = 1e-19
        precis2 = 1e-19

        !-------------------------------------------------------------------------------
        !Loop over rows to get the implicit scaling information
        !-------------------------------------------------------------------------------
        DO i = 1, n

            aamax = 0.0_rspec
            DO j = 1, n
                IF (ABS(a(i, j)) > aamax) aamax = ABS(a(i, j))

            END DO

            IF (aamax == 0.0_rspec) THEN

                iflag = 1
                write (*, *) 'NCLASS_DECOMP/ERROR:singular matrix(1)', aamax
                print *, 'erreur nclas_decomp'
                GOTO 9999

            END IF

            vv(i) = 1.0_rspec/aamax

        END DO
        !      if (n .eq. 18) stop
        !-------------------------------------------------------------------------------
        !Use Crout's method for decomposition
        !-------------------------------------------------------------------------------
        !Loop over columns

        DO j = 1, n

            if (j .eq. 1) goto 20
            DO i = 1, j - 1
                sumhoul = a(i, j)
                ik = i - 1
                if (ik .ge. 1) then

                    DO k = 1, ik

                        sumhoul = sumhoul - a(i, k)*a(k, j)

                    END DO

                end if

                a(i, j) = sumhoul

            END DO
20          continue

            !Search for largest pivot element using dum as a figure of merit
            aamax = 0.0_rspec
            dum = 0.0_rspec
            DO i = j, n

                sumhoul = a(i, j)
                sumhoult = sumhoul
                varia = 0.0_rspec
                if (j .gt. 1) then
                    DO k = 1, j - 1
                        sumhoul = sumhoul - a(i, k)*a(k, j)
                        varia = varia + a(i, k)*a(k, j)

                    END DO
                    if (abs(sumhoul) .lt. precis) then
                        write (*, *) 'sumhoul=', sumhoul
                        sumhoul = sign(precis, sumhoul)
                        !              sumhoul=0.0_rspec
                    end if
                    if (abs(sumhoul) .lt. precis) then
                        !            write(*,*) 'sumhoul=',sumhoul
                        !            sumhoul=sign(precis,sumhoul)
                        !            sumhoul=precis
                        !            write(*,*) 'sumhoul2=',sumhoul

                    end if
                    !            if (abs(sumhoul) .lt. precis) then
                    !              write(*,*) 'sumhoul=',sumhoul,i
                    !              sumhoul=0.0
                    !              if (sumhoul .ge. 0) then
                    !                sumhoul=precis
                    !              else
                    !                sumhoul=-precis
                    !              endif

                    !            endif
                end if
                a(i, j) = sumhoul
                dum = vv(i)*ABS(sumhoul)
                !          if (n .eq. 18 .and. j .eq. 1) then
                !
                !            write(*,*) 'dum=',dum,' sumhoul=',sumhoul
                !            write(*,*) 'i=',i,'j=',j,' vv(i)=',vv(i)
                !            write(*,*) 'k=',k,' aamax=',aamax,' n=',n
                !            write(*,*) 'a(i,1)=',a(i,1),' imax=',imax
                !
                !          endif
                !          if (n .eq. 18) then
                !              write(*,*) 'dum=',dum,' aamax=',aamax,' i=',i,imax,j
                !          endif
                IF ((dum >= aamax) .or. (abs(dum - aamax) < precis2)) THEN
                    if (abs(dum - aamax) < precis2) then
                        !              write(*,*) 'dum=',(dum-aamax)/precis2,i
                        !              stop
                    end if
                    imax = i
                    aamax = dum
                    if (i .eq. 18) then
                        !              stop

                    end if
                END IF

            END DO

            !        write(*,*) '---- imax=',imax,'  aamax=',aamax

            IF (j /= imax) THEN

                !Interchange rows
                DO k = 1, n

                    dum = a(imax, k)
                    a(imax, k) = a(j, k)
                    a(j, k) = dum

                END DO

                d = -d
                vv(imax) = vv(j)

            END IF

            indx(j) = imax
            !        if (n .eq. 18 .and. ipr .eq. 1) then
            !          write(*,*) 'j=',j,' imax=',imax
            !          write(*,*) 'dum=',dum,' indx(j)=',indx(j)
            !          stop
            !        endif
            IF (a(j, j) == 0.0_rspec) THEN

                iflag = 1
                write (*, *) 'NCLASS_DECOMP/ERROR:singular matrix(2)'
                GOTO 9999

            END IF

            IF (j /= n) THEN

                !Divide by pivot element
                dum = 1.0_rspec/a(j, j)

                DO i = j + 1, n

                    a(i, j) = a(i, j)*dum

                END DO

            END IF

        END DO

        !      write(*,*) 'a(2,2)=',a(2,2)
        !      stop

9999    CONTINUE

    END SUBROUTINE NCLASS_DECOMP

    FUNCTION NCLASS_ERF(x)
        !-------------------------------------------------------------------------------
        !NCLASS_ERF evaluates the error function erf(x)
        !References:
        !  M.Abramowitz, L.A.Stegun, Handbook of Math. Functions, p. 299
        !  W.A.Houlberg 1/2002
        !Input:
        !  x                   -argument of werror function [-]
        !Output:
        !  NCLASS_ERF          -value of error function [-]
        !Comments:
        !  The error function can consume a lot of time deep in the multiple species
        !    loop so a very efficient calculation is called for
        !  This is much more critical than accuracy as suggested by T.Amano
        !-------------------------------------------------------------------------------

        REAL(KIND=rspec), INTENT(IN) :: &
            x                      !argument of error function [-]

        !Declaration of output variables
        REAL(KIND=rspec) :: &
            NCLASS_ERF             !value of error function [-]

        !-------------------------------------------------------------------------------
        !Declaration of local variables
        REAL(KIND=rspec) :: &
            t

        REAL(KIND=rspec), PARAMETER :: &
            c = 0.3275911_rspec, &
            a1 = 0.254829592_rspec, &
            a2 = -0.284496736_rspec, &
            a3 = 1.421413741_rspec, &
            a4 = -1.453152027_rspec, &
            a5 = 1.061405429_rspec

        !-------------------------------------------------------------------------------
        !Apply fit
        !-------------------------------------------------------------------------------
        t = one/(one + c*x)
        NCLASS_ERF = one - (a1 + (a2 + (a3 + (a4 + a5*t)*t)*t)*t)*t*EXP(-x**2)

    END FUNCTION NCLASS_ERF

    SUBROUTINE NORMALISATION(RMAT, B, n, VECM)

        !Declaration of input/output variables
        REAL(KIND=rspec), INTENT(INOUT) :: RMAT(:, :), B(:)
        !Declaration of output variables
        REAL(KIND=rspec), INTENT(OUT) :: VECM(:)
        !Declaration of input variables
        INTEGER, INTENT(in) ::  n

        INTEGER :: k, j

        do k = 1, n
            VECM(k) = -1e10

            do j = 1, n
                if (VECM(k) .le. RMAT(k, j)) VECM(k) = RMAT(k, j)
            end do
            RMAT(k, :) = RMAT(k, :)/VECM(k)
            B(k) = B(k)/VECM(k)
        end do

    END SUBROUTINE NORMALISATION

END MODULE NCLASS_MOD



import collections

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import (CoreTransport,
                                                   CoreTransportProfiles1D)
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class GLF23(CoreTransport.Model):
    r"""
        GLF23
        ===============================
            - 2D GLF equations with massless isothermal passing electrons from

        References:
        =============
            - Waltz et al, Phys. of Plasmas 6(1995)2408
    """

    def __init__(self, d, *args, grid: RadialGrid, **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": "anomalous",
                "index": 6,
                "description": f"{self.__class__.__name__}"
            }}, d or {}), *args, grid=grid, **kwargs)

    def update(self, equilibrium: Equilibrium, core_profiles: CoreProfiles, core_transport: CoreTransport):
        if (core_profiles.nions > 2):  # THEN
            ifail = 22
            return
        # END if

        # if (core_profiles.nions == 1) : GLF23.idengrad = 2
        # if (PRESENT(diagnostics)) : # THEN
        #     diagnostics.nr = core_profiles.nr
        #     diagnostics.nk = 1
        # #END if
        # CALL TCI_ZERO_OUT(TRANSPORT, ifail)

        # Direct inputs

        # ... simple geometric and magnetic information

        rmajor_exp = geometry.R0  # RMAEQ2(1) * z_mpcm
        arho_exp = geometry.arho  # RA * z_mpcm # pis changed
        bt_exp = geometry.BT0  # FEQ2(1)/RMAEQ2(1) * z_Tpgauss

        # ... impurities
        if (core_profiles.nions == 1):  # THEN
            zimp_exp = 6.0
            amassimp_exp = 12.0
        else:
            zimp_exp = core_profiles.z_charges[:, 2]
            amassimp_exp = core_profiles.z_masses[:, 2]  # 2*zimp_exp
        # END if

        amassgas_exp = core_profiles.z_mass(1)  # ATM(1)

        # ... Profiles -  on cellboundaries

        rho = core_profiles.rho(1: core_profiles.nr)
        Te_m = core_profiles.Te(1: core_profiles.nr)
        Ti_m = core_profiles.Ti(1: core_profiles.nr, 1)
        ni_m = core_profiles.ni(1: core_profiles.nr, 1)/1.0e19
        ne_m = core_profiles.ne(1: core_profiles.nr)/1.0e19
        ns_m = core_profiles.ns(1: core_profiles.nr)/1.0e19
        if (core_profiles.nions == 2):  # THEN # Added for additional impurity information PIS 2007-11-20
            nq_m = core_profiles.ni(1: core_profiles.nr, 2)/1.0e19
        # END if
        zeff_exp = core_profiles.zeff(1: core_profiles.nr)

        OMEB = core_profiles.wexb
        alpha_exp = core_profiles.alpha
        if (GLF23.x_alpha == 2.0):  # THEN # set alpha stabilization only in outer half
            GLF23.x_alpha = 1.0  # set back input value to GLF23 for prescribed alpha_mhd
            WHERE(rho < 0.5) alpha_exp = 0.
        # END if
        rmin_exp = geometry.rminor  # AMIEQ1[i] * z_mpcm
        rmaj_exp = geometry.rmajor  # 0.5*(RMAEQ1[i]+RMIEQ1[i]) * z_mpcm
        vphi_m = core_profiles.Vtor(: , 1)
        vpar_m = core_profiles.Vtor(:, 1)  # 0.0
        vper_m = core_profiles.Vpol(:, 1)  # 0.0

        # Profiles on -midpoint grid
        q_exp = geometry.q  # QSF[i]
        elong_exp = geometry.kappa  # ELOEQ1[i]
        gradrho_exp = geometry.grho1  # SEQ1[i]/DVEQ1[i]
        gradrhosq_exp = geometry.grho2  # SEQ1[i]/DVEQ1[i]
        shat_exp = geometry.shear     #

        # Derived quantities
        cs_da = sqrt(Te_m/amassgas_exp*1.0E3*z_electroncharge/z_protonmass)/arho_exp

        angrotp_exp = core_profiles.Vtor(:, 1)/rmaj_exp  # U(I,NETOTP1)/RHOI[i] # Correct normalization 2004/10/06 PIS
        egamma_exp = OMEB/cs_da
        gamma_p_exp = 0.
        if (GLF23.irotstab == 2):  # THEN # set ExB stabilization only in outer half
            GLF23.irotstab = 0  # set back input value to GLF23 for prescribed ExB
            # WHERE (rho < 0.5)
            # egamma_exp = 0.
            # angrotp_exp = 0.
            # END WHERE
        # END if

        z_beta = 2*z_mu0/bt_exp**2*(ne_m*(te_m + (zeff_exp - 1.0)/zimp_exp /
                                          (zimp_exp - 1.0) * Ti_m) + ni_m*ti_m)*1.60219e3

        zpte_in = 0.
        zpti_in = 0.
        zpne_in = 0.
        zpni_in = 0.

        if (core_profiles.nions == 2):  # THEN
            zpnq_in = 0.  # Added for additional impurity information PIS 2007-11-20
        # END if

        GLF23.jmaxm = core_profiles.nr - 1
        # set up diagnostics coupling
        # if (PRESENT(DIAGNOSTICS)) : # THEN
        #     nk = 1
        #     norder = 2
        #     if (.NOT. ALLOCATED(diagnostics.omega)) : # THEN
        #     #PIS      if (.NOT. ASSOCIATED(diagnostics.omega)) : # THEN
        #         CALL TCI_ALLOCATE(core_profiles.nions, core_profiles.nr, nk, norder, DIAGNOSTICS, ifAIL)
        #     else: if (diagnostics.norder /= norder .or. diagnostics.nk /= nk) : # THEN
        #         call TCI_DEALLOCATE(diagnostics, ifail)
        #         CALL TCI_ALLOCATE(core_profiles.nions, core_profiles.nr, nk, norder, DIAGNOSTICS, ifAIL)
        #     #END if
        #     CALL TCI_ZERO_OUT(DIAGNOSTICS, ifail)
        #     DIAGNOSTICS.rho = core_profiles.rho
        # #END if
        (
            #
            # ..default outputs
            #
            diffnem,
            chietem,
            chiitim,
            etaphim,
            etaparm,
            etaperm,
            exchm,
            #

            diff_m,
            diffz_m,
            chie_m,
            chii_m,
            etaphi_m,
            etapar_m,
            etaper_m,
            exch_m,
            egamma_m,
            gamma_p_m,
            anrate_m,
            anrate2_m,
            anfreq_m,
            anfreq2_m,

            egamma_d
        ) = glf2d(GLF23.leigen, GLF23.nroot, GLF23.iglf, GLF23.jshoot, GLF23.jmm, GLF23.jmaxm, GLF23.itport_pt, GLF23.irotstab, te_m, ti_m, ne_m, ni_m, ns_m, GLF23.igrad, GLF23.idengrad, zpte_in, zpti_in, zpne_in, zpni_in, angrotp_exp, egamma_exp, gamma_p_exp, vphi_m, vpar_m, vper_m, zeff_exp, bt_exp, GLF23.bt_flag, rho, arho_exp, gradrho_exp, gradrhosq_exp, rmin_exp, rmaj_exp, rmajor_exp, zimp_exp, amassimp_exp, q_exp, shat_exp, alpha_exp, elong_exp, amassgas_exp, GLF23.x_alpha_exb, GLF23.x_alpha, GLF23.i_delay
                  #    , diffnem, chietem, chiitim, etaphim, etaparm, etaperm
                  #    , exchm, diff_m, diffz_m, chie_m, chii_m, etaphi_m, etapar_m   # diffz_m Added for additional impurity information PIS 2007-11-20
                  #    , etaper_m
                  #    , exch_m, egamma_m, egamma_d, gamma_p_m
                  #    , anrate_m, anrate2_m, anfreq_m, anfreq2_m

                  )

        # Need to calculate scalelengths###

        core_transport.rho[0] = 0.0
        # DO I = 2, core_profiles.nr

        core_transport.rho[1:] = (core_profiles.rho[1:] + core_profiles.rho[:-1])/2
        # END do

        # Set up local variables

        # CALL TCI_ALLOCATE(geometry.nr, geometry_rm, ifail)
        # CALL TCI_ALLOCATE(core_profiles.nions, core_profiles.nr, profiles_rm, ifail)

        # Map geometry to midpoint for needed quantities
        # CALL TCI_MAP2MIDPOINT(GEOMETRY, GEOMETRY_RM)
        # CALL TCI_MAP2MIDPOINT(Profiles, PROFILES_RM)

        # DO I = 2, core_profiles.nr
        if True:
            drho = core_profiles.rho[i] - core_profiles.rho[i - 1]
            zpte_in = (te_m[i] - te_m[i - 1] + sqrt(epsilon(1.0)))/drho
            zpti_in = (ti_m[i] - ti_m[i - 1] + sqrt(epsilon(1.0)))/drho
            zpne_in = (ne_m[i] - ne_m[i - 1] + sqrt(epsilon(1.0)))/drho
            zpni_in = (ni_m[i] - ni_m[i - 1] + sqrt(epsilon(1.0)))/drho
            k = 1

            # Effective heat fluxes
            core_transport.chieff[i, k] = chii_m[i]
            core_transport.heat_fluxes[i, k] = -profiles_rm.ni[i, k]*core_transport.chieff[i,
                                                                                           k] * zpti_in*z_kev2Joule*geometry_rm.grho2[i]*geometry_rm.vprime[i]

            # Effective main ion particle fluxes
            core_transport.deff[i, k] = diff_m[i]
            core_transport.particle_fluxes[i, k] = -core_transport.deff[i, k] * \
                zpni_in*geometry_rm.grho2[i] * geometry_rm.vprime[i]*1.0e19

            # Set heat diffusivity and pinch term
            # Does not allow heat pinches####
            core_transport.chi[i, k] = chii_m[i]
            core_transport.vt[i, k] = 0.0

            # ion particle fluxes
            core_transport.d[i, k] = diff_m[i]
            core_transport.vp[i, k] = 0.0

            # Momenmun core_transport...
            core_transport.chimom[i, k] = etaphi_m[i]

            # Transport splits a la IMP4 recipe
            if (GLF23.l_vd_shift):  # THEN
                F_BDS = -core_transport.deff[i, k]*zpni_in*geometry_rm.grho2[i]
                G_BDS = -core_transport.chieff[i, k]*zpti_in*geometry_rm.grho2[i]

                # Floor the particle transport

                if (F_BDS/G_BDS <= 0.2):  # THEN

                    core_transport.d[i, k] = 0.2*core_transport.chi[i, k]
                    core_transport.vp[i, k] = ((F_BDS + core_transport.d[i, k]*geometry_rm.grho2[i]
                                                * zpni_in)/core_profiles.ni[i, k])/geometry_rm.grho1[i]

                # END if
            # END if

            # Add impurity contribution
            if (core_profiles.nions == 2):  # THEN
                k = k + 1

                # Added for additional impurity information PIS 2007-11-20
                zpnq_in = (nq_m[i] - nq_m(i - 1) + sqrt(epsilon(1.0)))/drho

                core_transport.chieff[i, k] = chii_m[i]  # Assumed equal to main ions
                core_transport.heat_fluxes[i, k] = -profiles_rm.ni[i, k]*core_transport.chieff[i,
                                                                                               k] * zpti_in*z_kev2Joule*geometry_rm.grho2[i]*geometry_rm.vprime[i]

                # Does not allow heat pinches####
                core_transport.chi[i, k] = chii_m[i]  # Assumed equal to main
                core_transport.vt[i, k] = 0.0

                core_transport.chimom[i, k] = 0.0

                core_transport.deff[i, k] = diffz_m[i]
                core_transport.particle_fluxes[i, k] = -core_transport.deff[i, k] * \
                    zpnq_in*geometry_rm.grho2[i]*geometry_rm.vprime[i]*1.0e19

                # Does not allow heat pinches####
                core_transport.chi[i, k] = chii_m[i]
                core_transport.vt[i, k] = 0.0

                core_transport.d[i, k] = diffz_m[i]
                core_transport.vp[i, k] = 0.0

                if (GLF23.l_vd_shift):  # THEN

                    F_BDS = -core_transport.deff[i, k]*zpnq_in*geometry_rm.grho2[i]
                    G_BDS = -core_transport.chieff[i, k]*zpti_in*geometry_rm.grho2[i]

                    # Floor the particle transport

                    if (F_BDS/G_BDS <= 0.2):  # THEN
                        core_transport.d[i, k] = 0.2*core_transport.chi[i, k]
                        core_transport.vp[i, k] = ((F_BDS + core_transport.d[i, k]*geometry_rm.grho2[i]
                                                    * zpnq_in)/core_profiles.ni[i, k])/geometry_rm.grho1[i]

                    # END if
                # END if

            # END if

            # Add in electrons last in array
            k = k + 1

            core_transport.chieff[i, k] = chie_m[i]
            core_transport.heat_fluxes[i, k] = -profiles_rm.ne[i]*core_transport.chieff[i,
                                                                                        k] * zpte_in*z_kev2Joule*geometry_rm.grho2[i]*geometry_rm.vprime[i]

            # Does not allow heat pinches####
            core_transport.chi[i, k] = chie_m[i]
            core_transport.vt[i, k] = 0.0
            core_transport.chimom[i, k] = 0.0

            if (core_profiles.nions == 2):  # THEN
                core_transport.deff[i, k] = (diff_m[i]*zpni_in + zimp_exp[i]*diffz_m[i]
                                             * zpnq_in)/zpne_in  # By ambipolarity
            else:
                core_transport.deff[i, k] = diff_m[i]
            # END if

            core_transport.d[i, k] = core_transport.deff[i, k]
            core_transport.vp[i, k] = 0.0

            core_transport.particle_fluxes[i, k] = -core_transport.deff[i, k] * \
                zpne_in*geometry_rm.grho2[i]*geometry_rm.vprime[i]*1.0e19

            # Floor the particle transport
            if (GLF23.l_vd_shift):  # THEN
                F_BDS = -core_transport.deff[i, k]*zpne_in*geometry_rm.grho2[i]
                G_BDS = -core_transport.chieff[i, k]*zpte_in*geometry_rm.grho2[i]

                if (F_BDS/G_BDS <= 0.2):  # THEN

                    core_transport.d[i, k] = 0.2*core_transport.chi[i, k]
                    core_transport.vp[i, k] = ((F_BDS + core_transport.d[i, k]*geometry_rm.grho2[i]
                                                * zpne_in)/core_profiles.ne[i]) / geometry_rm.grho1[i]

                # END if
            # END if

            if (PRESENT(DIAGNOSTICS)):  # THEN
                diagnostics.n_modes(i, 1) = 0
                diagnostics.n_modes(i, 1) = 0
                if (anrate_m[i] > 0.0):  # THEN
                    diagnostics.n_modes(i, 1) = diagnostics.n_modes(i, 1) + 1
                    diagnostics.n_modemap(i, 1, diagnostics.n_modes(i, 1)) = 1
                    diagnostics.omega(i, 1, diagnostics.n_modes(i, 1)) = cmplx(anfreq_m[i], anrate_m[i], RP)*cs_da[i]
                #       else:
                #          diagnostics.omega(i,1,diagnostics.n_modes(i,1)) = cmplx(0.0, 0.0,RP)
                # END if
                if (anrate2_m[i] > 0.0):  # THEN
                    diagnostics.n_modes(i, 1) = diagnostics.n_modes(i, 1) + 1
                    diagnostics.n_modemap(i, 1, diagnostics.n_modes(i, 1)) = 2
                    diagnostics.omega(i, 1, diagnostics.n_modes(i, 1)) = cmplx(anfreq2_m[i], anrate2_m[i], RP)*cs_da[i]
                #       else:
                #          diagnostics.n_modemap(i,1, diagnostics.n_modes(i,1)) = 0
                #          diagnostics.omega(i,1,diagnostics.n_modes(i,1)) = cmplx(0.0, 0.0, RP)
                # END if
            # END if
        # END DO

        # CALL TCI_DEALLOCATE(PROFILES_RM, ifail)
        # CALL TCI_DEALLOCATE(GEOMETRY_RM, ifAIL)

    # #END SUBROUTINE TCI_GLF23

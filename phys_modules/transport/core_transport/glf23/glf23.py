
import collections
import collections.abc
import enum
from math import sqrt

import scipy
from fytok.transport.CoreProfiles import CoreProfiles, CoreProfiles1D
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.numlib import constants, np
from spdm.numlib.misc import array_like
from spdm.util.logger import logger, SP_NO_DEBUG

from .glf23_mod import glf2d, glf

PI = constants.pi
EPSILON = 1.0e-34


class GLF23(CoreTransport.Model):
    r"""
        GLF23
        ===============================
            - 2D GLF equations with massless isothermal passing electrons from Waltz et al, Phys. of Plasmas 6(1995)2408

        @ref:
            - Advances in the simulation of toroidal gyro‐Landau fluid model turbulence ,Waltz R.E., eta. Physics of Plasmas 2(6) 2408-2416 1995
            - A gyro-Landau-fluid transport model, Waltz R.E., eta.   Physics of Plasmas 7(4) 2482–2496 July 1997
    """

    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": "anomalous", "index": 6,
                            "description": f"anomalous {self.__class__.__name__}"},
             "code": {"name": "glf23",
                      "paremeters": {
                          "has_impurity": True
                      }}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles,  **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        has_impurity = self.get("code.paratmeters.has_impurity", False)

        core_profiles_1d = core_profiles.profiles_1d
        eq_profiles_1d = equilibrium.time_slice.profiles_1d
        rho_tor = np.asarray(self.profiles_1d.grid.rho_tor)
        rho_tor_norm = np.asarray(self.profiles_1d.grid.rho_tor_norm)
        psi_norm = np.asarray(self.profiles_1d.grid.psi_norm)

        n_grid_point = len(rho_tor_norm)
        nmode = 40  # <=20

        #
        a = self.grid.rho_tor[-1]

        R0 = self.grid.vacuum_toroidal_field.r0
        B0 = np.abs(self.grid.vacuum_toroidal_field.b0)

        # t all plasma gradients are taken along the average minor axis (d/dr), where r=(Rin + Rout)/2.

        r_minor = eq_profiles_1d.minor_radius(psi_norm)

        r_major = eq_profiles_1d.geometric_axis.r(psi_norm)

        r = r_minor

        dr_drho_tor_norm = Function(rho_tor_norm, r).derivative(rho_tor_norm)

        drho_tor_norm_dr = 1.0/dr_drho_tor_norm

        drho_tor_norm_dr_r_rho_tor_norm = drho_tor_norm_dr*r/a/(rho_tor_norm+EPSILON)

        elongation = eq_profiles_1d.elongation(psi_norm)

        q = eq_profiles_1d.q(psi_norm)

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)  # [ev]
        dTe = core_profiles_1d.electrons.temperature.derivative(rho_tor_norm)  # [m^-3]
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)
        dNe = core_profiles_1d.electrons.density.derivative(rho_tor_norm)

        rlte = -(a * dTe / Te*drho_tor_norm_dr)
        rlne = -(a * dNe / Ne*drho_tor_norm_dr)

        Ti: Function = np.zeros_like(rho_tor_norm)
        Ni: Function = np.zeros_like(rho_tor_norm)
        Nimp: Function = np.zeros_like(rho_tor_norm)

        n_imp = 0
        for ion in core_profiles_1d.ion:
            if not ion.is_impurity:
                Ti += ion.temperature(rho_tor_norm) * ion.z*ion.density(rho_tor_norm)
                Ni += ion.z*ion.density(rho_tor_norm)
            else:
                n_imp += 1
                Nimp = Nimp + ion.z*ion.density(rho_tor_norm)

        # atomic number working hydrogen gas
        A_H = 1
        # charge number of ion
        Z_H = 1

        Ti /= Ni

        # Ti = array_like(rho_tor, Ti)
        # Ni = array_like(rho_tor, Ni)
        rlti = -a*Function(rho_tor_norm, Ti).derivative(rho_tor_norm)/Ti*drho_tor_norm_dr
        rlni = -a*Function(rho_tor_norm, Ni).derivative(rho_tor_norm)/Ni*drho_tor_norm_dr

        if n_imp > 0:
            rlnimp = -a * Function(rho_tor_norm, Nimp).derivative(rho_tor_norm)/Nimp*drho_tor_norm_dr
        else:
            rlnimp = np.zeros_like(rho_tor_norm)

        # Shafranov shift
        shafranov_shift = eq_profiles_1d.shape_property.geometric_axis.r(psi_norm)-R0

        c_s = 3.09e5*np.sqrt(Te/1000/A_H)

        omega_i = (constants.elementary_charge/constants.m_p)*(Z_H/A_H)*B0

        rho_s = c_s/omega_i

        geo_fac = eq_profiles_1d.gm7(psi_norm)/eq_profiles_1d.gm3(psi_norm)*drho_tor_norm_dr

        # local gyrobohm unit of diffusion $c_{s}\left(\rho_{s}^{2}/a\right)$
        gyrobohm_unit = c_s*(rho_s**2/a)
        # gyrobohm_unit=1.e-4* 9.79e5*(Te )**.5e0/(a*100.0)*(1.02e2*(Te )**.5e0/B0)**2*A_H**.50

        taui = Ti/Te

        magnetic_shear = eq_profiles_1d.magnetic_shear(psi_norm) * drho_tor_norm_dr_r_rho_tor_norm

        zeff = core_profiles_1d.zeff(rho_tor_norm)

        beta_e = Ne * Te/(B0**2)*(2*constants.mu_0 * constants.electron_volt)  # Te [eV]

        nu_ei = 1.5625e-15 * Ne*(Te/1000.0)**(-3/2)

        xnu = nu_ei/(c_s/a)*zeff

        ########################################################################
        # Output

        #  plasma diffusivity for ions
        diff_m = np.zeros_like(rho_tor_norm)

        #  plasma diffusivity for impurities
        diff_im_m = np.zeros_like(rho_tor_norm)

        # electron energy diffusivity
        chi_e_m = np.zeros_like(rho_tor_norm)

        # ion energy diffusivity
        chi_i_m = np.zeros_like(rho_tor_norm)

        chi_e_e_m = np.zeros_like(rho_tor_norm)

        # anomalous e to i energy exchange
        exch_m = np.zeros_like(rho_tor_norm)

        # toroidal momentun diffusivity
        eta_phi_m = np.zeros_like(rho_tor_norm)

        # parallel component of toroidal momentum diffusivity
        eta_par_m = np.zeros_like(rho_tor_norm)

        # perpendicular
        eta_per_m = np.zeros_like(rho_tor_norm)

        # leading mode growth rate in c_s/a
        gamma = np.zeros_like(rho_tor_norm)

        # leading mode freq rate in c_s/a
        freq = np.zeros_like(rho_tor_norm)

        ph_m = np.zeros_like(rho_tor_norm)

        kyf = np.zeros_like(rho_tor_norm)

        # rlti = np.linspace(0.0, 5.0, n_grid_point)

        xkyf_k = np.zeros([n_grid_point, nmode])
        gamma_k = np.zeros([n_grid_point, nmode])
        freq_k = np.zeros([n_grid_point, nmode])
        diff_k = np.zeros([n_grid_point, nmode])
        chi_e_k = np.zeros([n_grid_point, nmode])
        chi_i_k = np.zeros([n_grid_point, nmode])

        ######################################################################
        # INPUT

        iglf = 0

        if True:  # INPUT : configure

            # print debug information
            glf.lprint_gf = 0 if SP_NO_DEBUG else 98

            # eigen_gf = 0 use cgg eigenvalue solver (default)
            #           = 1 use generalized tomsqz eigenvalue solver
            #           = 2 use zgeev eigenvalue solver
            glf.eigen_gf = 1  # self.get("code.parameters.eigenvalue_solver", 0)

            #  nroot number of equations
            # 8 for pure plasma, 12 for full impurity dynamics
            glf.nroot_gf = 8 if not has_impurity else 12

            #  iflagin(1:20) control flags

            #   iflagin(1) 0 use ky=ky0; 1 use landau damping point
            glf.iflagin_gf[1 - 1] = 1

            #   iflagin(2) 0. local w_d and k_par "2d"; 1 fit to trial function "3d"
            glf.iflagin_gf[2 - 1] = 1

            #   iflagin(3) 0,1,and 2 fix up park low high beta and beta lim elong factor
            glf.iflagin_gf[3 - 1] = 1

            #   iflagin(4) 0 trapped electron Waltz EoS 1 weiland EoS
            glf.iflagin_gf[4 - 1] = 0

            #   iflagin(5) rms_theta
            #           0:fixed;
            #           1: inverse to q/2 ;
            #           2: inverse to root q/2
            #           3: inverse to xparam(13)*(q/2-1)+1.
            #           5: for retuned rms-theta
            glf.iflagin_gf[5 - 1] = 5

            #  xparam(1:20) control parameters

            # xparam(1:2): idelta=xi*xparam(1)+xparam(2) nonadiabatic electron response
            glf.xparam_gf[1 - 1] = 0.0
            glf.xparam_gf[2 - 1] = 0
            # xparam(3) multiplier park_gf(high betae)/ park_gf(low betae) -1
            glf.xparam_gf[3 - 1] = .70

            glf.xparam_gf[4 - 1] = 0.0

            #   xparam(6)+1. is enhancement of xnueff
            glf.xparam_gf[6 - 1] = 0.0

            #   xparam(7) coef of resistivity
            glf.xparam_gf[7 - 1] = 1.0

            #   xparam(8) cut off on rotational stabilization
            glf.xparam_gf[8 - 1] = 0.0

            #   xparam(9)+1. is shape (triangularity) enhancement to beta_crit
            glf.xparam_gf[9 - 1] = 1.0

            #   xparam(10) is high k electron mode enhancement
            glf.xparam_gf[10 - 1] = 0.0

            #   xparam(11:12) lamda parameters
            glf.xparam_gf[11 - 1] = 0.0
            glf.xparam_gf[12 - 1] = 0.0

            #   xparam(13) rms_theta q-dependence
            glf.xparam_gf[13 - 1] = 0.20

            #   xparam(14)  adjustment to gamma_p avoiding negative viscosity
            glf.xparam_gf[14 - 1] = 1.0

            #   xparam(15)   (1+xparam(15)*reps trapped electron fraction
            glf.xparam_gf[15 - 1] = -0.10

            #   xparam(16) rms_theta shat dependence
            glf.xparam_gf[16 - 1] = 0.0

            #   xparam(17) ""
            glf.xparam_gf[17 - 1] = 0.10

            #   xparam(18) rms_theta betae dependence
            glf.xparam_gf[18 - 1] = 0.00

            #   xparam(19:20)  extra
            glf.xparam_gf[19 - 1] = 0.0
            glf.xparam_gf[20 - 1] = 0.0

            #   xparam(21) 1 add impurity energy diffusivity to ion energy diffusivity
            glf.xparam_gf[21 - 1] = 0.0

            #   xparam(22) >0 keeps gamma_e from changeing spectrum
            glf.xparam_gf[22 - 1] = 0.0

            #   xparam(23) 1. kills kx**2 in k_m**2
            glf.xparam_gf[23 - 1] = 1.0

            #   xparam(24) exb damping model
            glf.xparam_gf[24 - 1] = 0
            glf.xparam_gf[25 - 1] = 0
            glf.xparam_gf[26 - 1] = 0
            glf.xparam_gf[27 - 1] = 0
            glf.xparam_gf[28 - 1] = 0
            glf.xparam_gf[29 - 1] = 0
            glf.xparam_gf[30 - 1] = 0

            glf.ikymax_gf = nmode
            glf.xkymin_gf = .020
            glf.xkymax_gf = .80

            #     turn on high-k eta-e modes
            glf.xparam_gf[10] = 0

            # turn on self-consistant alpha-stabilization
            #      ialphastab=1
            #      turn on EXB shear stabilization
            #      alpha_e_gf=1. full on ExB shear

            #  alpha_e O(1-3)   doppler rot shear coef
            glf.alpha_e_gf = 0
            #  gamma_e    doppler rot shear rate in units of c_s/a
            glf.gamma_e_gf = 0  # -.000000000001  # egamma_m[idx]

            #  alpha_p 1.5  fit for parallel velocity shear effect at rmaj=3 and q=2
            glf.alpha_p_gf = 0.500
            #  gamma_p    parallel velocity shear rate (-d v_phi/ drho) in units of c_s/a
            glf.gamma_p_gf = 0.0  # gamma_p_m[idx]

            #  alpha_star O(1-3)  gyyrobohm breaking coef for diamg. rot. shear
            glf.alpha_star_gf = 0.0
            #  gamma_star ion diamagnetic rot shear rate in units of c_s/a
            glf.gamma_star_gf = 0.0  # vstarp_m[idx]

            glf.alpha_mode_gf = 0.0

            #  atomic number working hydrogen gas
            glf.amassgas_gf = A_H

            # x_alpha,        ! 1 full (0 no) alpha stabilization  with alpha_exp
            #                 !-1 full (0 no) self consistent alpha_m stab.

            #  ky0=k_theta*rho_s; k_theta= nq/r; normally 0.3
            glf.xky0_gf = .30

            #  rms_theta width of phi**2 mode function for best fit near pi/3
            glf.rms_theta_gf = scipy.constants.pi/3.0

            #  park=1  (0) is a control parameter to turn on (off) parallel motion
            #       0.405 best at zero beta and 2.5x larger at high beta..see iflagin(3)
            glf.park_gf = 0.70

            #  ghat=1  (0) is a control parameter to turn on (off) curvature drift
            glf.ghat_gf = 1

            #  gchat=1 (0) is a control parameter to turn on (off) div EXB motion
            glf.gchat_gf = 1

            #  adamp= radial mode damping exponent  1/4 < adamp < 3/4
            #       0.25 from direct fit of simulations varying radial mode damping
            #   but 0.75 is better fit to rlti dependence
            glf.adamp_gf = .500

            #  kdamp model damping normally 0.
            glf.xkdamp_gf = 0

            glf.cnorm_gf = 1.0

            # zimp_exp,       ! effective Z of impurity
            glf.zpmnimp = 1.0
            # amassimp_exp,   ! effective A of impurity
            glf.amassimp_gf = 1.0  # amassimp_exp[idx]

            # impurity dynamics not turned on by default
            # and simple dilution included (idengrad=2, dil_gf=1-nim/nem)
            # to turn on impurity dynamics need to change number of roots
            # supply zimp_exp, amassimp_exp, and fractional density weights
            # apwt_gf and aiwt_gf

        if (iglf == 1):  # for retuned model
            glf.cnorm_gf = 50               # ITG normalization (via GYRO runs)
            glf.xparam_gf[10-1] = 12.0      # ETG normalization (cnorm*xparam(10))
            glf.iflagin_gf[5-1] = 5         # rms theta fit formula
            glf.xparam_gf[13-1] = 0.15      # rms_theta q-dependence
            glf.xparam_gf[16-1] = 0.15      # rms_theta shat dependence
            glf.xparam_gf[17-1] = 0.25      # rms_theta shat dependence
            glf.xparam_gf[19-1] = 1.0       # rms_theta alpha dependence
            glf.adamp_gf = .700           # radial mode damping exponent
            glf.alpha_p_gf = 0.350        # parallel velocity shear fit
            glf.park_gf = 0.80            # parallel ion motion fit
            # glf.bt_flag = 1               # use real geometry ExB shear

        for idx, x in enumerate(rho_tor_norm):

            if False:  # INPUT for test
                #######################################################################
                # rho dependent
                #######################################################################

                #  rlti=a/L_Ti   a/L_f=   a d ln f / d rho
                glf.rlti_gf = 3  # rlti[idx]

                #  rlte=a/L_Te
                glf.rlte_gf = 3  # rlte[idx]

                #  rlne= a/L_ne
                glf.rlne_gf = 1  # rlne[idx]

                #  rlni= a/L_ni
                glf.rlni_gf = 1  # rlni[idx]

                #  rlnimp= a/L_nim
                glf.rlnimp_gf = 0  # rlnimp[idx]

                #  dil=1.-ni_0/ne_0  dilution
                glf.dil_gf = 0  # 1.0 - Ni[idx]/Ne[idx]

                #  apwt = ni_0/ne_0
                glf.apwt_gf = 1  # Ni[idx]/Ne[idx]

                #  aiwt = nim_0/ne_0
                glf.aiwt_gf = 0  # Nimp[idx]/Ne[idx]

                #  taui=Ti/Te
                glf.taui_gf = 1  # Ti[idx]/Te[idx]

                #  rmin=r/a
                glf.rmin_gf = 0.5  # r_minor[idx]/a

                #  rmaj=Rmaj/a
                glf.rmaj_gf = 3  # r_major[idx]/a

                # q
                glf.q_gf = q[idx]

                #  xnu=nu_ei/(c_s/a)
                glf.xnu_gf = 0.0  # xnu[idx]

                #  betae=neTe/(B**2/(8pi))  0 is electrostatic
                glf.betae_gf = 0  # beta_e[idx]

                #  shat= dlnr/dlnrho used only for parallel dynamics part
                glf.shat_gf = 1  # magnetic_shear[idx]

                #  alpha local shear parameter or MHD pressure grad (s-alpha diagram)
                glf.alpha_gf = 0.0

                #  elong= local elongation or kappa
                glf.elong_gf = 1.6  # elongation[idx]

                glf.zimp_gf = 1.6

                # glf.amassimp_gf=12.0
                # zimp_gf=zimp_exp         ! made radially depende FK
                # amassimp_gf=amassimp_exp ! FL made radially dependent

            else:   # INPUT for reall
                #######################################################################
                # rho dependent
                #######################################################################

                #  rlti=a/L_Ti   a/L_f=   a d ln f / d rho
                glf.rlti_gf = rlti[idx]

                #  rlte=a/L_Te
                glf.rlte_gf = rlte[idx]

                #  rlne= a/L_ne
                glf.rlne_gf = rlne[idx]

                #  rlni= a/L_ni
                glf.rlni_gf = rlni[idx]

                #  rlnimp= a/L_nim
                glf.rlnimp_gf = rlnimp[idx]

                #  dil=1.-ni_0/ne_0  dilution
                glf.dil_gf = 1.0 - Ni[idx]/Ne[idx]

                #  apwt = ni_0/ne_0
                glf.apwt_gf = Ni[idx]/Ne[idx]

                #  aiwt = nim_0/ne_0
                glf.aiwt_gf = Nimp[idx]/Ne[idx]

                #  taui=Ti/Te
                glf.taui_gf = Ti[idx]/Te[idx]

                #  rmin=r/a
                glf.rmin_gf = r_minor[idx]/a

                #  rmaj=Rmaj/a
                glf.rmaj_gf = r_major[idx]/a

                # q
                glf.q_gf = q[idx]

                #  xnu=nu_ei/(c_s/a)
                glf.xnu_gf = xnu[idx]

                #  betae=neTe/(B**2/(8pi))  0 is electrostatic
                glf.betae_gf = beta_e[idx]

                #  shat= dlnr/dlnrho used only for parallel dynamics part
                glf.shat_gf = magnetic_shear[idx]

                #  alpha local shear parameter or MHD pressure grad (s-alpha diagram)
                glf.alpha_gf = 0.0

                #  elong= local elongation or kappa
                glf.elong_gf = elongation[idx]

                glf.zimp_gf = zeff[idx]

                # glf.amassimp_gf=12.0
                # zimp_gf=zimp_exp         ! made radially depende FK
                # amassimp_gf=amassimp_exp ! FL made radially dependent

            # Call GLF2D
            try:
                glf2d(iglf)
            except ValueError as error:
                logger.error("f2py Error:")
                logger.exception(error)
                raise RuntimeError(f"glf2d: failed")

            if True:  # OUTPUT
                # yparam(20) output diagnostics

                # kyf  value of ky used
                kyf[idx] = glf.xky_gf[0]

                # gamma   leading mode growth rate in c_s/a
                gamma[idx] = glf.gamma_gf[0]

                # freq    leading mode freq rate in c_s/a
                freq[idx] = glf.freq_gf[0]

                # ph_m    (e phi /T_e)/(rho_s/a)  saturation value
                ph_m[idx] = glf.phi_norm_gf[0]

                # d_hat    plasma diffusivity for ions
                diff_m[idx] = glf.diff_gf

                # d_im_hat    plasma diffusivity for impurities
                diff_im_m[idx] = glf.diff_im_gf

                # chii_hat ion energy diffusivity
                chi_i_m[idx] = glf.chii_gf

                # chie_hat electron energy diffusivity
                chi_e_m[idx] = glf.chie_gf

                # eta_par_hat parallel component of toroidal momentum diffusivity
                eta_par_m[idx] = glf.eta_par_gf

                # eta_per_hat perpendicular    ""
                eta_per_m[idx] = glf.eta_per_gf

                # eta_phi_hat toroidal momentun diffusivity
                eta_phi_m[idx] = glf.eta_phi_gf

                chi_e_e_m[idx] = glf.chie_e_gf

                # exch_hat anomalous e to i energy exchange
                exch_m[idx] = glf.exch_gf

                xkyf_k[idx, :] = glf.xkyf_k_gf
                gamma_k[idx, :] = glf.gamma_k_gf[0]
                freq_k[idx, :] = glf.freq_k_gf[0]
                diff_k[idx, :] = glf.diff_k_gf
                chi_e_k[idx, :] = glf.chie_k_gf
                chi_i_k[idx, :] = glf.chii_k_gf

        # self.profiles_1d.electrons.particles["d"] = Function(rho_tor_norm, diff_m)

        self.profiles_1d.electrons.energy["d"] = Function(rho_tor_norm,  chi_e_m*geo_fac*gyrobohm_unit)
        self.profiles_1d.electrons.energy["v"] = 0
        self.profiles_1d.electrons.particles["d"] = Function(rho_tor_norm,  diff_m*geo_fac*gyrobohm_unit)
        self.profiles_1d.electrons.particles["v"] = 0

        chi_i = Function(rho_tor_norm, chi_i_m*geo_fac*gyrobohm_unit)
        diff_i = Function(rho_tor_norm,  diff_m*geo_fac*gyrobohm_unit)
        diff_imp = Function(rho_tor_norm,  diff_im_m*geo_fac*gyrobohm_unit)
        momentum_tor = Function(rho_tor_norm,  eta_phi_m*geo_fac*gyrobohm_unit)
        momentum_par = Function(rho_tor_norm,  eta_par_m*geo_fac*gyrobohm_unit)
        momentum_per = Function(rho_tor_norm,  eta_per_m*geo_fac*gyrobohm_unit)
        for ion in core_profiles_1d.ion:
            self.profiles_1d.ion.put(_next_, {
                "label": ion.label,
                "a": ion.a,
                "z": ion.z,
                "is_impurity":  ion.is_impurity})

            d = self.profiles_1d.ion[-1]

            if not ion.is_impurity:
                d.particles["d"] = diff_i
                d.particles["v"] = 0

            else:
                d.particles["d"] = diff_imp
                d.particles["v"] = 0

            # assumed same as majority ions
            d.energy["d"] = chi_i
            d.energy["v"] = 0

            d.momentum.toroidal["d"] = momentum_tor
            d.momentum.toroidal["v"] = 0
            d.momentum.parallel["d"] = momentum_par
            d.momentum.parallel["v"] = 0
            d.momentum["perpendicular.d"] = momentum_per
            d.momentum["perpendicular.v"] = 0

        self.profiles_1d["debug_xkyf_k"] = xkyf_k
        self.profiles_1d["debug_gamma_k"] = gamma_k
        self.profiles_1d["debug_freq_k"] = freq_k
        self.profiles_1d["debug_diff_k"] = diff_k
        self.profiles_1d["debug_chi_e_k"] = chi_e_k
        self.profiles_1d["debug_chi_i_k"] = chi_i_k

        self.profiles_1d["debug_gamma"] = Function(rho_tor_norm, gamma)
        self.profiles_1d["debug_freq"] = Function(rho_tor_norm, freq)
        self.profiles_1d["debug_kyf"] = Function(rho_tor_norm, kyf)
        self.profiles_1d["debug_rho_s"] = Function(rho_tor_norm, rho_s)

        # trans_imp: CoreTransportModel.Profiles1D.Ion = self.profiles_1d.ion.combine(predication={"is_impurity": True})
        # trans_imp.particles["d"] = Function(rho_tor_norm, diff_im_m)
        self.profiles_1d["debug_Ti"] = Function(rho_tor_norm, Ti)
        self.profiles_1d["debug_Te"] = Function(rho_tor_norm, Te)
        self.profiles_1d["debug_r_minor"] = Function(rho_tor_norm, r_minor)
        self.profiles_1d["debug_r_major"] = Function(rho_tor_norm, r_major)

        self.profiles_1d["debug_rlti"] = Function(rho_tor_norm, rlti)
        self.profiles_1d["debug_rlte"] = Function(rho_tor_norm, rlte)
        self.profiles_1d["debug_rlni"] = Function(rho_tor_norm, rlni)
        self.profiles_1d["debug_rlne"] = Function(rho_tor_norm, rlne)
        self.profiles_1d["debug_taui"] = Function(rho_tor_norm, taui)
        self.profiles_1d["debug_beta_e"] = Function(rho_tor_norm, beta_e)

        self.profiles_1d["debug_geo_fac"] = Function(rho_tor_norm, geo_fac)
        self.profiles_1d["debug_drho_tor_norm_dr"] = Function(rho_tor_norm, drho_tor_norm_dr)
        self.profiles_1d["debug_gyrobohm_unit"] = Function(rho_tor_norm,  gyrobohm_unit)
        self.profiles_1d["debug_magnetic_shear"] = Function(rho_tor_norm, magnetic_shear)
        self.profiles_1d["debug_elongation"] = Function(rho_tor_norm, elongation)
        self.profiles_1d["debug_zeff"] = Function(rho_tor_norm, zeff)
        self.profiles_1d["debug_q"] = Function(rho_tor_norm, q)
        self.profiles_1d["debug_beta_e"] = Function(rho_tor_norm, beta_e)
        self.profiles_1d["debug_psi_norm"] = Function(rho_tor_norm, psi_norm)


__SP_EXPORT__ = GLF23

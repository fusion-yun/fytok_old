
import collections
import collections.abc
import enum
from math import sqrt

import scipy
from fytok.transport.CoreProfiles import CoreProfiles, CoreProfiles1D
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium, TWOPI
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.numlib import constants, np
from spdm.numlib.misc import array_like
from spdm.util.logger import logger, SP_NO_DEBUG

from .mod_glf23 import callglf2d

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

        num_grid_point = len(rho_tor_norm)
        nmode = 40  # <=20

        #
        rho_lcfs = self.grid.rho_tor[-1]

        R0 = self.grid.vacuum_toroidal_field.r0
        B0 = np.abs(self.grid.vacuum_toroidal_field.b0)

        # t all plasma gradients are taken along the average minor axis (d/dr), where r=(Rin + Rout)/2.

        r_minor = eq_profiles_1d.minor_radius(psi_norm)

        r_major = eq_profiles_1d.geometric_axis.r(psi_norm)

        elongation = eq_profiles_1d.elongation(psi_norm)

        q = eq_profiles_1d.q(psi_norm)

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)  # [ev]
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)  # [m^-3]

        Ti: Function = np.zeros([num_grid_point])
        Ni: Function = np.zeros([num_grid_point])
        Ns: Function = np.zeros([num_grid_point])

        zeff: Function = np.zeros([num_grid_point])

        N_imp: Function = np.zeros([num_grid_point])
        zeff_imp: Function = np.zeros([num_grid_point])

        A_imp: float = 0.0
        Z_imp: float = 0.0

        n_imp = 0
        for ion in core_profiles_1d.ion:
            if not ion.is_impurity:
                Ti += ion.temperature(rho_tor_norm) * ion.z*ion.density(rho_tor_norm)
                Ni += ion.z*ion.density(rho_tor_norm)
                zeff += ion.z*ion.density(rho_tor_norm)

            else:

                n_imp += 1
                A_imp += ion.a*ion.z
                Z_imp += ion.z
                zeff_imp += ion.z*ion.z*ion.density(rho_tor_norm)
                N_imp += ion.z*ion.density(rho_tor_norm)

        zeff /= Ni

        if n_imp > 0:
            zeff_imp /= N_imp
            A_imp /= Z_imp
        else:
            A_imp = 1
            Z_imp = 1

        # atomic number working hydrogen gas
        A_H = 1
        # charge number of ion
        Z_H = 1

        Ti /= Ni

        # c_s = 3.09e5*np.sqrt(Te/1000/A_H)
        # omega_i = (constants.elementary_charge/constants.m_p)*(Z_H/A_H)*B0
        # rho_s = c_s/omega_i

        grad_rho = eq_profiles_1d.gm7(psi_norm)
        grad_rho2 = eq_profiles_1d.gm3(psi_norm)

        magnetic_shear = eq_profiles_1d.magnetic_shear(psi_norm)

        #  toroidal angular rotation frequency due to the ExB drift, experimental toroidal angular velocity (1/s)
        rotation_frequency_tor_sonic = core_profiles_1d.rotation_frequency_tor_sonic(rho_tor_norm)

        # zeff = core_profiles_1d.zeff(rho_tor_norm)

        # beta_e = Ne * Te/(B0**2)*(2*constants.mu_0 * constants.electron_volt)  # Te [eV]

        # Shafranov shift
        shafranov_shift = eq_profiles_1d.shape_property.geometric_axis.r(psi_norm)-R0

        # alpha from experiment  $−q^2 R (d\beta/dr)$
        alpha_MHD = shafranov_shift  # q**2 * R0 * Function(rho_tor, beta_e).derivative(rho_tor)

        # exp exb shear rate in units of csda_exp
        #  if itport_pt(4)=-1 itport_pt(5)=0
        egamma_exp = np.zeros([num_grid_point])

        # exp par. vel. shear rate in units of csda_ex
        #  if itport_pt(4)=-1 itport_pt(5)=0
        gamma_p_exp = np.zeros([num_grid_point])

        # vphi_m,          0:jmaxm toroidal velocity m/sec
        #                  if itport_pt(4)=1 itport_pt(5)=0 otherwise output
        vphi_m = np.zeros([num_grid_point])

        # vpar_m,          0:jmaxm parallel velocity m/sec
        #                  if itport_pt(4)=1 itport_pt(5)=1 otherwise output
        vpar_m = np.zeros([num_grid_point])

        # vper_m,          0:jmaxm perp. velocity m/sec
        #                  if itport_pt(4)=1 itport_pt(5)=1 otherwise output
        vper_m = np.zeros([num_grid_point])

        ########################################################################
        # Output

        # # ion plasma diffusivity in m**2/sec
        # diffnem = np.zeros([num_grid_point])
        # # electron ENERGY diffuivity in m**2/sec
        # chietem = np.zeros([num_grid_point])
        # # ion      ENERGY diffuivity in m**2/sec
        # chiitim = np.zeros([num_grid_point])
        # # toroidal velocity diffusivity in m**2/sec
        # etaphim = np.zeros([num_grid_point])
        # # parallel velocity diffusivity in m**2/sec
        # etaparm = np.zeros([num_grid_point])
        # # perpendicular velocity diffusivity in m**2/sec
        # etaperm = np.zeros([num_grid_point])
        # # turbulent electron to ion ENERGY exchange in MW/m**3  0:jmaxm values
        # exchm = np.zeros([num_grid_point])

        #  plasma diffusivity for ions
        diff_m = np.zeros([num_grid_point])

        #  plasma diffusivity for impurities
        diffz_m = np.zeros([num_grid_point])

        # electron energy diffusivity
        chie_m = np.ones([num_grid_point])

        # ion energy diffusivity
        chii_m = np.zeros([num_grid_point])

        chie_e_m = np.zeros([num_grid_point])

        # anomalous e to i energy exchange
        exch_m = np.zeros([num_grid_point])

        # toroidal momentun diffusivity
        etaphi_m = np.zeros([num_grid_point])

        # parallel component of toroidal momentum diffusivity
        etapar_m = np.zeros([num_grid_point])

        # perpendicular
        etaper_m = np.zeros([num_grid_point])

        # 0:jmaxm exb shear rate in units of local csda_m
        egamma_m = np.zeros([num_grid_point])
        # 0:jmaxm exb shear rate delayed by i_delay steps
        egamma_d = np.zeros([num_grid_point, 10])
        # 0:jmaxm par. vel. shear rate in units of local csda_m
        gamma_p_m = np.zeros([num_grid_point])
        # 0:jmaxm leading mode rate in unints of local csda_m
        anrate_m = np.zeros([num_grid_point])
        # 0:jmaxm 2nd mode rate in units of local csda_m
        anrate2_m = np.zeros([num_grid_point])
        # 0:jmaxm leading mode frequency
        anfreq_m = np.zeros([num_grid_point])
        # !0:jmaxm 2nd mode frequency
        anfreq2_m = np.zeros([num_grid_point])
        # 0:jmaxm output added for renormalization outside GLF (F.I. 13/05/2004)
        csda_m = np.ones([num_grid_point])

        #################################################################################
        callglf2d(
            # ----------------------------------------------------------------------------
            # INPUTS
            99,             # lprint
            2,              # leigen,          eigenvalue solver
                            #                  0 for cgg (default), 1 for tomsqz, 2 for zgeev
            8,              # nroot,           no. roots,8 for default, 12 for impurity dynamics
            0,              # iglf,            0 for original GLF23, 1 for retuned version
            0,              # jshoot,          jshoot=0 time-dep code;jshoot=1 shooting code
            0,              # jmm,             grid number;jmm=0 does full grid jm=1 to jmaxm-1
            np.array([      # itport_pt,       1:5 transport flags
                1,  # itport pt(1) density transport
                1,  # itport pt(2) electron transport
                1,  # itport pt(3) ion transport
                0,  # itport pt(4) vphi transport(-1 use egamma exp)
                0,  # itport pt(5) vtheta transport(-1 use gamma p exp)
            ]),
            1,              # irotstab,        0 to use egamma_exp; 1 use egamma_m
            Te/1000.0,      # te_m,            0:jmaxm te electron temperature Kev       itport_pt(2)=1 transport
            Ti/1000.0,      # ti_m,            0:jmaxm ti Ion temperature Kev            itport_pt(3)=1 transport
            Ne*1e-19,       # ne_m,            0:jmaxm ne electron density 10**19 1/m**3
            Ni*1e-19,       # ni_m,            0:jmaxm ni Ion density 10**19 1/m**3      itport_pt(1)=1 transport
            Ns*1e-19,       # ns_m,            0:jmaxm ns Fast ion density (10**19m−3)
            0,              # i_grad,          default 0, for D-V method use i_grad=1 to input gradients
            2,              # idengrad,        default 2, for simple dilution
            0,              # zpte_in,         externally provided log gradient te w.r.t rho (i_grad=1)
            0,              # zpti_in,         externally provided log gradient ti w.r.t rho
            0,              # zpne_in,         externally provided log gradient ne w.r.t rho
            0,              # zpni_in,         externally provided log gradient ni w.r.t rho
            rotation_frequency_tor_sonic,    # angrotp_exp,     0:jmaxm exp plasma toroidal angular velocity 1/sec
                            #                  if itport_pt(4)=0 itport_pt(5)=0
            egamma_exp,     # egamma_exp,      0:jmaxm exp exb shear rate in units of csda_exp
                            #                  if itport_pt(4)=-1 itport_pt(5)=0
            gamma_p_exp,    # gamma_p_exp,     0:jmaxm exp par. vel. shear rate in units of csda_exp
                            #                  if itport_pt(4)=-1 itport_pt(5)=0
            vphi_m,         # vphi_m,          0:jmaxm toroidal velocity m/sec
                            #                  if itport_pt(4)=1 itport_pt(5)=0 otherwise output
            vpar_m,         # vpar_m,          0:jmaxm parallel velocity m/sec
                            #                  if itport_pt(4)=1 itport_pt(5)=1 otherwise output
            vper_m,         # vper_m,          0:jmaxm perp. velocity m/sec
                            #                  if itport_pt(4)=1 itport_pt(5)=1 otherwise output
            zeff,           # zeff_exp,        0:jmaxm ne in 10**19 1/m**3
            B0,             # bt_exp,          vaccuum axis toroidal field in tesla
            0,              # bt_flag,         switch for effective toroidal field use in rhosda
            rho_tor_norm,   # rho,             0:jmaxm 0 < rho < 1 normalized toroidal flux (rho=rho/rho(a))
            rho_lcfs,       # arho_exp,        rho(a), toroidal flux at last closed flux surface (LCFS)
                            #                  toroidal flux= B0*rho_phys**2/2 (m)
                            #                  B0=bt_exp, arho_exp=rho_phys_LCFS
            grad_rho2,      # gradrho_exp,     0:jmaxm dimensionless <|grad rho_phys |**2>
            grad_rho,       # gradrhosq_exp,   0:jmaxm dimensionless <|grad rho_phys |>
                            #                  NOTE:can set arho_exp=1.,if gradrho_exp=<|grad rho |>
                            #                                  and gradrhosq_exp = <|grad rho |**2>
            r_minor,        # rmin_exp,        0:jmaxm minor radius in meters
            r_major,        # rmaj_exp,        0:jmaxm major radius in meters
            R0,             # rmajor_exp,      axis major radius
            Z_imp,          # zimp_exp,        effective Z of impurity
            A_imp,          # amassimp_exp,    effective A of impurity
            q,              # q_exp,           0:jmaxm safety factor
            magnetic_shear,  # shat_exp,       0:jmaxm magnetic shear, d (ln q_exp)/ d (ln rho)
            alpha_MHD,      # alpha_exp,       0:jmaxm MHD alpha from experiment  $−q^2 R (d\beta/dr)$
            elongation,     # elong_exp,       0:jmaxm elongation
            A_H,            # amassgas_exp,    atomic number working hydrogen gas
            0,              # alpha_e,         1 full (0 no) no ExB shear stab
            0,              # x_alpha,         1 full (0 no) alpha stabilization  with alpha_exp
                            #                  -1 full (0 no) self consistent alpha_m stab.
            0,              # i_delay,         i_delay time delay for ExB shear should be non-zero only
                            #                  once per step and is less than or equal 10
            # ----------------------------------------------------------------------------
            # OUTPUTS
            0,              # ion plasma diffusivity in m**2/sec
            0,              # electron ENERGY diffuivity in m**2/sec
            0,              # ion      ENERGY diffuivity in m**2/sec
            0,              # toroidal velocity diffusivity in m**2/sec
            0,              # parallel velocity diffusivity in m**2/sec
            0,              # perpendicular velocity diffusivity in m**2/sec
            0,              # turbulent electron to ion ENERGY exchange in MW/m**3 

            # 0:jmaxm values
            diff_m,
            diffz_m,        # impurity  diffusivity in m**2/sec  Added 08/04/2004 by F. Imbeaux
            chie_m,
            chii_m,
            etaphi_m,
            etapar_m,
            etaper_m,
            exch_m,

            egamma_m,       # exb shear rate in units of local csda_m
            egamma_d,       # exb shear rate delayed by i_delay steps
            gamma_p_m,      # par. vel. shear rate in units of local csda_m
            anrate_m,       # leading mode rate in unints of local csda_m
            anrate2_m,      # 2nd mode rate in units of local csda_m
            anfreq_m,       # leading mode frequency
            anfreq2_m,      # 2nd mode frequency
            csda_m          # output added for renormalization outside GLF (F.I. 13/05/2004)
        )

        ########################################################

        # self.profiles_1d.electrons.particles["d"] = Function(rho_tor_norm, diff_m)

        self.profiles_1d.electrons.energy["d"] = Function(rho_tor_norm,  chie_m)
        self.profiles_1d.electrons.energy["v"] = 0
        self.profiles_1d.electrons.particles["d"] = Function(rho_tor_norm,  diff_m)
        self.profiles_1d.electrons.particles["v"] = 0

        chi_i = Function(rho_tor_norm, chii_m)
        diff_i = Function(rho_tor_norm,  diff_m)
        diff_imp = Function(rho_tor_norm,  diffz_m)
        momentum_tor = Function(rho_tor_norm,  etaphi_m)
        momentum_par = Function(rho_tor_norm,  etapar_m)
        momentum_per = Function(rho_tor_norm,  etaper_m)
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

        self.profiles_1d["debug_csda_m"] = Function(rho_tor_norm, csda_m)
        self.profiles_1d["debug_gamma_p_m"] = Function(rho_tor_norm, gamma_p_m)

        # trans_imp: CoreTransportModel.Profiles1D.Ion = self.profiles_1d.ion.combine(predication={"is_impurity": True})
        # trans_imp.particles["d"] = Function(rho_tor_norm, diff_im_m)
        self.profiles_1d["debug_Ti"] = Function(rho_tor_norm, Ti)
        self.profiles_1d["debug_Te"] = Function(rho_tor_norm, Te)
        # self.profiles_1d["debug_r_minor"] = Function(rho_tor_norm, r_minor)
        # self.profiles_1d["debug_r_major"] = Function(rho_tor_norm, r_major)

        # self.profiles_1d["debug_taui"] = Function(rho_tor_norm, taui)
        # self.profiles_1d["debug_beta_e"] = Function(rho_tor_norm, beta_e)

        # self.profiles_1d["debug_drho_tor_norm_dr"] = Function(rho_tor_norm, drho_tor_norm_dr)
        # self.profiles_1d["debug_gyrobohm_unit"] = Function(rho_tor_norm,  gyrobohm_unit)
        # self.profiles_1d["debug_magnetic_shear"] = Function(rho_tor_norm, magnetic_shear)
        # self.profiles_1d["debug_elongation"] = Function(rho_tor_norm, elongation)
        # self.profiles_1d["debug_zeff"] = Function(rho_tor_norm, zeff)
        # self.profiles_1d["debug_q"] = Function(rho_tor_norm, q)
        # self.profiles_1d["debug_beta_e"] = Function(rho_tor_norm, beta_e)
        # self.profiles_1d["debug_psi_norm"] = Function(rho_tor_norm, psi_norm)


__SP_EXPORT__ = GLF23

import sys
import pprint
import numpy as np
sys.path.append("/home/salmon/workspace/fytok/phys_modules/")
import nclass

if __name__ == "__main__":
    iflag = 1
    m_s = 1
    print(nclass.nclass_mod.nclass.__doc__)
    m_i = 1
    m_z = 1
    maxms = 5
    d = nclass.nclass_mod.nclass(
        m_i,                             # m_i                 -number of isotopes (> 1) [-]
        m_z,                             # m_z                 -highest charge state [-]
        0.10,                           # p_b2                -<B**2> [T**2]
        0.10,                           # p_bm2               -<1/B**2> [/T**2]
        0.10,                           # p_eb                -<E.B> [V*T/m]
        0.10,                           # p_fhat              -mu_0*F/(dPsi/dr) [rho/m]
        # p_fm(m)             -poloidal moments of drift factor for PS [/m**2]
        np.zeros([m_i],   dtype=float),
        0.10,                           # p_ft                -trapped fraction [-]
        0.10,                           # p_grbm2             -<grad(rho)**2/B**2> [rho**2/m**2/T**2]
        0.10,                           # p_grphi             -potential gradient Phi' [V/rho]
        0.10,                           # p_gr2phi            -second potential gradient Psi'(Phi'/Psi')' [V/rho**2]
        0.10,                           # p_ngrth             -<n.grad(Theta)> [/m]
        np.zeros([m_i], dtype=float),                        # amu_i(i)            -atomic mass number [-]
        np.zeros([m_i], dtype=float),                        # grt_i(i)            -temperature gradient [keV/rho]
        np.zeros([m_i], dtype=float),                        # temp_i(i)           -temperature [keV]
        np.zeros([m_i, m_z], dtype=float),                   # den_iz(i,z)         -density [/m**3]
        # fex_iz(3,i,z)       -moments of external parallel force [T*n/m**3]
        np.zeros([3, m_i, m_z], order='F', dtype=float),

        np.zeros([m_i, m_z], dtype=float),                   # grp_iz(i,z)         -pressure gradient [keV/m**3/rho]
        1,                                      # ipr
        l_banana=False,               # L_BANANA,
        l_pfirsch=True,               # L_PFIRSCH,
        l_potato=True,               # L_POTATO,
        k_order=2,                  # K_ORDER,
        c_den=0.2,                # C_DEN,
        c_potb=0.2,                # C_POTB,
        c_potl=0.2,                # C_POTL,
    )

    # pprint.pprint(d)

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
     ) = d

    pprint.pprint(TAU_SS.shape)
    # iflag=iflag,
    # p_etap=0.2,                # P_ETAP,
    # 0.2,                # P_JBBS,
    # 0.2,                # P_JBEX,
    # 0.2,                # P_JBOH,
    # m_s=m_s,                  # M_S,
    # jm_s=np.ones(5, dtype=int),         # JM_S,
    # jz_s=np.ones(5),         # JZ_S,
    # np.ones(5),         # BSJBP_S,
    # np.ones(5),         # BSJBT_S,
    # np.ones([5, 5]),    # GFL_S,
    # np.ones(5),         # DN_S,
    # np.ones(5),         # VNNT_S,
    # np.ones(5),         # VNEB_S,
    # np.ones(5),         # VNEX_S,
    # np.ones([3, 3]),    # DP_SS,
    # np.ones([3, 3]),    # DT_SS,
    # np.ones([3, 3, 3]),  # UPAR_S,
    # np.ones([3, 3, 3]),  # UTHETA_S,
    # np.ones([3, 3]),     # QFL_S,
    # np.ones([3, 3]),     # CHI_S,
    # np.ones(5),          # VQNT_S,
    # np.ones(5),          # VQEB_S,
    # np.ones(5),          # VQEX_S,
    # np.ones([3, 3]),     # CHIP_SS,
    # np.ones([3, 3]),     # CHIT_SS,
    # np.ones([3, 3, 3, 3]),  # CALM_I,
    # np.ones([3, 3, 3, 3]),  # CALN_II,
    # np.ones([3, 3, 3, 3]),  # CAPM_II,
    # np.ones([3, 3, 3, 3]),  # CAPN_II,
    # np.ones([3, 3, 3]),       # YMU_S,
    # np.ones([3]),             # SQZ_S,
    # np.ones([3]),             # XI_S,
    # np.ones([3, 3]),  # TAU_SS)
    # )

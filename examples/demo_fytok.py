import pprint
import sys
sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fytok.FyTok import FyTok
    from fytok.Wall import Wall
    from fytok.PFActive import PFActive

    from spdm.data.Entry import open_entry
    from spdm.util.logger import logger

    entry = open_entry("east+mdsplus:///home/salmon/public_data/~t/?default_tree_name=efit_east#shot=55555")

    wall = Wall(
        limiter=entry.wall.description_2d[0].limiter.unit[0].outline,
        vessel=entry.wall.description_2d[0].vessel.annular
    )

    pf_active = PFActive(entry.pf_active)

    # tok = FyTok(entry)
    # logger.debug(tok)
    # logger.debug(tok.entry.wall())
    # lfcs_r = entry.equilibrium.time_slice[10].boundary.outline.r.__value__()[:, 0]
    # lfcs_z = entry.equilibrium.time_slice[10].boundary.outline.z.__value__()[:, 0]
    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # tok.entry.equilibrium.solve(core_profiles=None, psivals=psivals)

    # tok.entry.core_profiles.vacuum_toroidal_field.b0 = 1.0
    # tok.entry.core_profiles.vacuum_toroidal_field.r0 = 1.0
    # tok.entry.core_profiles.profiles_1d.conductivity_parallel = 1.0

    # # tok.solve(0.1, max_iters=1, constraints={"psivals": psivals})

    fig = plt.figure()

    axis = fig.add_subplot(111)
    wall.plot(axis=axis)
    pf_active.plot(axis=axis)

    axis.axis("scaled")
    fig.savefig("a.svg")

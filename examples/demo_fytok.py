
import sys
sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


#################################################
import matplotlib.pyplot as plt
from spdm.util.logger import logger
from spdm.data.Collection import Collection
from fytok.FyTok import FyTok

if __name__ == "__main__":

    db = Collection("east+mdsplus:///home/salmon/public_data/~t/", default_tree_name="efit_east")
    entry = db.open(shot=55555).entry

    tok = FyTok()

    tok.wall.limiter = [entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                        entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]

    tok.wall.vessel = {"inner": [entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),
                                 entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()],

                       "outer":  [entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),
                                  entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]}

    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__value__()
        tok.pf_coils.add(str(coil.name),
                         r=float(rect.r),
                         z=float(rect.z),
                         width=float(rect.width),
                         height=float(rect.height),
                         turns=int(coil.element[0].turns_with_sign)
                         )

    lfcs_r = entry.equilibrium.time_slice[10].boundary.outline.r.__value__()[:, 0]
    lfcs_z = entry.equilibrium.time_slice[10].boundary.outline.z.__value__()[:, 0]

    psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]
    # psivals = [ (R, Z, 0.0) for R, Z in zip(entry.equilibrium.time_slice[10].boundary.outline.r.__value__(),
    #             entry.equilibrium.time_slice[10].boundary.outline.z.__value__()) ]

    tok.equilibrium.solve(psivals=psivals)

    fig = plt.figure()
    
    tok.plot(axis= fig.add_subplot(111))

    fig.savefig("a.svg")

import collections
import pathlib
import pprint
from functools import cached_property

from spdm.numlib import np
from spdm.util.logger import logger
from spdm.data.Node import Dict
from spdm.data.Entry import Entry, _next_
from spdm.data.File import File


def sp_read_geqdsk(file):
    """
    :param file: input file / file path
    :return: profile object
    """

    description = file.read(48)
    idum = int(file.read(4))
    nw = int(file.read(4))
    nh = int(file.read(4))
    file.readline()

    rdim = float(file.read(16))
    zdim = float(file.read(16))
    rcentr = float(file.read(16))
    rleft = float(file.read(16))
    zmid = float(file.read(16))
    file.readline()
    rmaxis = float(file.read(16))
    zmaxis = float(file.read(16))
    simag = float(file.read(16))
    sibry = float(file.read(16))
    bcentr = float(file.read(16))
    file.readline()
    current = float(file.read(16))
    simag = float(file.read(16))
    xdum = float(file.read(16))
    rmaxis = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    zmaxis = float(file.read(16))
    xdum = float(file.read(16))
    sibry = float(file.read(16))
    xdum = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    def _read_data(count, width=16):
        data = []
        for n in range(count):
            data.append(float(file.read(width)))
            if n >= count - 1 or ((n + 1) % 5 == 0):
                file.readline()
        data = np.asarray(data)
        return data

    #
    fpol = _read_data(nw)
    pres = _read_data(nw)
    ffprim = _read_data(nw)
    pprim = _read_data(nw)

    psirz = _read_data(nw * nh).reshape([nh, nw])

    qpsi = _read_data(nw)

    try:
        nbbs = int(file.read(5))
        limitr = int(file.read(5))
        file.readline()

        bbsrz = _read_data(nbbs * 2).reshape([nbbs, 2])
        limrz = _read_data(limitr * 2).reshape([limitr, 2])
    except:
        nbbs = 0
        limitr = 0
        bbsrz = None
        limrz = None

    data = {
        "description": description,
        # "idum": idum,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        # "simag": simag,
        # "rmaxis": rmaxis,
        # "zmaxis": zmaxis,
        # "sibry": sibry,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprim": pprim,
        "psirz": psirz,
        "qpsi": qpsi,
        "bbsrz": bbsrz,
        "limrz": limrz,

    }
    return data


def sp_write_geqdsk(p, file):
    """
    :param profile: object

    :param file: file path / file
    :return:
    """

    nw = p["nw"]
    nh = p["nh"]

    file.write("%48s%4i%4i%4i\n" % (p["description"], 3, p["nw"], p["nh"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rdim"], p["zdim"], p["rcentr"], p["rleft"], p["zmid"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rmaxis"], p["zmaxis"], p["simag"], p["sibry"], p["bcentr"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["current"], p["simag"], 0, p["rmaxis"], 0))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["zmaxis"], 0, p["sibry"], 0, 0))

    def _write_data(d):
        count = len(d)
        for n in range(count):
            file.write("%16.9e" % d[n])
            if (n == count - 1) or ((n + 1) % 5 == 0):
                file.write('\n')

    _write_data(p["fpol"])
    _write_data(p["pres"])
    _write_data(p["ffprim"])
    _write_data(p["pprim"])
    _write_data(p["psirz"].reshape([nw * nh]))
    _write_data(p["qpsi"])
    file.write("%5i%5i\n" % (p["bbsrz"].shape[0], p["limrz"].shape[0]))
    _write_data(p["bbsrz"].reshape([p["bbsrz"].size]))
    _write_data(p["limrz"].reshape([p["limrz"].size]))

    return


def sp_imas_equilibrium_to_geqdsk(eq, nw=125, nh=125):
    from spdm.numlib import interpolate

    coord_r = eq.coordinate_system.r
    coord_z = eq.coordinate_system.z
    rleft = coord_r.min()
    rdim = coord_r.max() - coord_r.min()
    zdim = coord_z.max() - coord_z.min()

    # rdim = 0.0
    # zdim = 0.0
    rcentr = eq.boundary.geometric_axis.r
    # rleft = 0.0
    zmid = eq.boundary.geometric_axis.z
    rmaxis = eq.global_quantities.magnetic_axis.r
    zmaxis = eq.global_quantities.magnetic_axis.z
    simag = eq.global_quantities.psi_axis
    sibry = eq.global_quantities.psi_boundary
    bcentr = eq.global_quantities.magnetic_axis.b_field_tor
    current = eq.global_quantities.ip

    # boundary

    rbbs = eq.boundary.lcfs.r
    zbbs = eq.boundary.lcfs.z

    bbsrz = np.append(rbbs.reshape([1, rbbs.size]), zbbs.reshape(
        [1, rbbs.size]), axis=0).transpose()
    # psi

    grid_r, grid_z = np.mgrid[rleft:rleft + rdim: nw * 1j, zmid - zdim / 2: zmid + zdim / 2: nh * 1j]
    coord_r = np.append(coord_r[:, :], coord_r[:, 0].reshape(coord_r.shape[0], 1), axis=1)
    coord_z = np.append(coord_z[:, :], coord_z[:, 0].reshape(coord_z.shape[0], 1), axis=1)
    points = np.append(coord_r.reshape([coord_r.size, 1]), coord_z.reshape([coord_z.size, 1]), axis=1)
    psi = eq.profiles_2d[1].psi
    values = psi[:coord_r.shape[0], :coord_r.shape[1]].reshape(points.shape[0])
    psirz = interpolate.griddata(points, values, (grid_r, grid_z), method='cubic').transpose()

    # profile

    fpol = eq.profiles_1d.f
    pres = eq.profiles_1d.pressure
    ffprim = eq.profiles_1d.f_df_dpsi
    pprim = eq.profiles_1d.dpressure_dpsi
    qpsi = eq.profiles_1d.q

    return {
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        "bbsrz": bbsrz,
        "psirz": psirz,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprim": pprim,
        "qpsi": qpsi

    }


def sp_geqdsk_to_imas_equilibrium(geqdsk, eq: Dict = None) -> Dict:
    if eq is None:
        eq = Dict()

    # eq.time = 0.0
    eq["vacuum_toroidal_field.r0"] = geqdsk["rcentr"]
    eq["vacuum_toroidal_field.b0"] = geqdsk["bcentr"]

    # rleft = 0.0
    eq["global_quantities.magnetic_axis.r"] = geqdsk["rmaxis"]
    eq["global_quantities.magnetic_axis.z"] = geqdsk["zmaxis"]
    # eq["global_quantities.magnetic_axis.b_field_tor"] = geqdsk["bcentr"]
    eq["global_quantities.psi_axis"] = geqdsk["simag"]
    eq["global_quantities.psi_boundary"] = geqdsk["sibry"]
    eq["global_quantities.ip"] = geqdsk["current"]

    # boundary

    eq["boundary.outline.r"] = geqdsk["bbsrz"][:, 0]
    eq["boundary.outline.z"] = geqdsk["bbsrz"][:, 1]

    nw = geqdsk["nw"]
    nh = geqdsk["nh"]
    rmin = geqdsk["rleft"]
    rmax = geqdsk["rleft"] + geqdsk["rdim"]
    zmin = geqdsk["zmid"] - geqdsk["zdim"]/2
    zmax = geqdsk["zmid"] + geqdsk["zdim"]/2

    eq["profiles_2d.grid_type"] = "rectangular"
    eq["profiles_2d.grid_index"] = 1
    eq["profiles_2d.grid.dim1"] = np.linspace(rmin, rmax, nw)
    eq["profiles_2d.grid.dim2"] = np.linspace(zmin, zmax, nh)
    eq["profiles_2d.psi"] = geqdsk["psirz"].T

    # profile
    eq["profiles_1d.f"] = geqdsk["fpol"]
    eq["profiles_1d.f_df_dpsi"] = geqdsk["ffprim"]
    eq["profiles_1d.pressure"] = geqdsk["pres"]
    eq["profiles_1d.dpressure_dpsi"] = geqdsk["pprim"]
    eq["profiles_1d.q"] = geqdsk["qpsi"]
    eq["profiles_1d.psi"] = np.linspace(geqdsk["simag"], geqdsk["sibry"], nw)

    return eq


class GEQdskDocument(File):
    def __init__(self, path, *args, mode="r", **kwargs):
        super().__init__(path=path, mode=mode)
        self._data = None

    @property
    def entry(self):
        if self._data is None:
            self._data = self.load(self.path)
        return self._data._entry

    def flush(self, *args, **kwargs):
        if "x" in self.mode or "w" in self.mode:
            self.save(self.path)

    def load(self, p, eq=None):
        with open(p or self._path, mode="r") as fp:
            eq = sp_geqdsk_to_imas_equilibrium(sp_read_geqdsk(fp), eq)
        return eq

    def save(self, p):
        geqdsk = sp_imas_equilibrium_to_geqdsk(self._data)
        with open(p or self._path, mode="w") as fp:
            sp_write_geqdsk(geqdsk, fp)


__SP_EXPORT__ = GEQdskDocument

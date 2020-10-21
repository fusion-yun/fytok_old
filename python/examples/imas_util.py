import numpy as np
import imas
import pprint


def sp_read_geqdsk(file):
    """

    :param file: input file / file path
    :return: profile object
    """
    if type(file) is str:
        file = open(file, "r")

    # res = np.genfromtxt(file, dtype=np.dtype([
    #     ("description", 'S50'), ("idum", "i4"), ("nw", "i4"), ("nh", "i4")
    # ]))

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
        data = np.ndarray(shape=[count], dtype=float)

        for n in range(count):
            data[n] = float(file.read(width))
            if n >= count - 1 or ((n + 1) % 5 == 0):
                file.readline()
        return data

    #
    fpol = _read_data(nw)
    pres = _read_data(nw)
    ffprim = _read_data(nw)
    pprim = _read_data(nw)

    psirz = _read_data(nw * nh).reshape([nw, nh])

    qpsi = _read_data(nw)

    nbbs = int(file.read(5))
    limitr = int(file.read(5))
    file.readline()

    bbsrz = _read_data(nbbs * 2).reshape([nbbs, 2])
    limrz = _read_data(limitr * 2).reshape([limitr, 2])
    file.close()
    return {
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


def sp_write_geqdsk(p, file):
    """
    :param profile: object

    :param file: file path / file
    :return:
    """
    if type(file) is str:
        file = open(file, "w")

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

    file.close()

    return


def sp_imas_equilibrium_to_geqdsk(eq, nw=125, nh=125):
    from scipy.interpolate import griddata

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

    grid_r, grid_z = np.mgrid[rleft:rleft + rdim: nw *
                              1j, zmid - zdim / 2: zmid + zdim / 2: nh * 1j]
    coord_r = np.append(coord_r[:, :], coord_r[:, 0].reshape(
        coord_r.shape[0], 1), axis=1)
    coord_z = np.append(coord_z[:, :], coord_z[:, 0].reshape(
        coord_z.shape[0], 1), axis=1)
    points = np.append(coord_r.reshape(
        [coord_r.size, 1]), coord_z.reshape([coord_z.size, 1]), axis=1)
    psi = eq.profiles_2d[1].psi
    values = psi[:coord_r.shape[0], :coord_r.shape[1]].reshape(points.shape[0])
    psirz = griddata(points, values, (grid_r, grid_z),
                     method='cubic').transpose()

    # profile

    fpol = eq.profiles_1d.f
    pres = eq.profiles_1d.pressure
    ffprim = eq.profiles_1d.f_df_dpsi
    pprim = eq.profiles_1d.dpressure_dpsi
    qpsi = eq.profiles_1d.q
    print(qpsi.shape)

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


def sp_geqdsk_to_imas_equilibrium(geqdsk, eq):
    # rdim = 0.0
    # zdim = 0.0
    eq.boundary.geometric_axis.r = geqdsk["rcentr"]
    eq.boundary.geometric_axis.z = geqdsk["zmid"]
    # rleft = 0.0
    eq.global_quantities.magnetic_axis.r = geqdsk["rmaxis"]
    eq.global_quantities.magnetic_axis.z = geqdsk["zmaxis"]
    eq.global_quantities.psi_axis = geqdsk["simag"]
    eq.global_quantities.psi_boundary = geqdsk["sibry"]
    eq.global_quantities.magnetic_axis.b_field_tor = geqdsk["bcentr"]
    eq.global_quantities.ip = geqdsk["current"]

    # boundary

    eq.boundary.outline.r = geqdsk["bbsrz"][:, 0]
    eq.boundary.outline.z = geqdsk["bbsrz"][:, 1]

    eq.profiles_2d.resize(1)
    eq.profiles_2d[0].grid_type.name = "rectangular"
    eq.profiles_2d[0].grid_type.index = 1
    eq.profiles_2d[0].psi = geqdsk["psirz"]

    # coord_r = eq.coordinate_system.r
    # coord_z = eq.coordinate_system.z
    # rleft = coord_r.min()
    # rdim = coord_r.max() - coord_r.min()
    # zdim = coord_z.max() - coord_z.min()

    # bbsrz = np.append(rbbs.reshape([1, rbbs.size]), zbbs.reshape(
    #     [1, rbbs.size]), axis=0).transpose()
    # # psi

    # grid_r, grid_z = np.mgrid[rleft:rleft + rdim: nw *
    #                           1j, zmid - zdim / 2: zmid + zdim / 2: nh * 1j]
    # coord_r = np.append(coord_r[:, :], coord_r[:, 0].reshape(
    #     coord_r.shape[0], 1), axis=1)
    # coord_z = np.append(coord_z[:, :], coord_z[:, 0].reshape(
    #     coord_z.shape[0], 1), axis=1)
    # points = np.append(coord_r.reshape(
    #     [coord_r.size, 1]), coord_z.reshape([coord_z.size, 1]), axis=1)
    # psi = eq.profiles_2d[1].psi
    # values = psi[:coord_r.shape[0], :coord_r.shape[1]].reshape(points.shape[0])
    # psirz = griddata(points, values, (grid_r, grid_z),
    #                  method='cubic').transpose()

    # profile

    eq.profiles_1d.f = geqdsk["fpol"]
    eq.profiles_1d.pressure = geqdsk["pres"]
    eq.profiles_1d.f_df_dpsi = geqdsk["ffprim"]
    eq.profiles_1d.dpressure_dpsi = geqdsk["pprim"]
    eq.profiles_1d.q = geqdsk["qpsi"]


if __name__ == "__main__":
    import os
    os.environ['USER'] = 'fydev'

    pp = pprint.PrettyPrinter(indent=2)

    geqdsk = sp_read_geqdsk("/workspaces/example_data/g063982.04800")

    imas_obj = imas.ids(63982, 48)
    # Create a new instance of database
    imas_obj.create_env("fydev", "test", "3")
    imas_obj.equilibrium.ids_properties.homogeneous_time = 1
    imas_obj.equilibrium.grids_ggd.resize(1)
    imas_obj.equilibrium.grids_ggd[0].grid.resize(1)
    imas_obj.equilibrium.grids_ggd[0].grid[0].name = "unspecified"
    imas_obj.equilibrium.grids_ggd[0].grid[0].index = 0
    imas_obj.equilibrium.grids_ggd[0].time = 1.234

    imas_obj.equilibrium.time_slice.resize(1)
    imas_obj.equilibrium.time.resize(1)
    imas_obj.equilibrium.time[0] = 0.0

    sp_geqdsk_to_imas_equilibrium(geqdsk, imas_obj.equilibrium.time_slice[0])
    imas_obj.equilibrium.time_slice[0].time = 0.0
    imas_obj.equilibrium.put()

    imas_obj.core_profiles.ids_properties.homogeneous_time = 1
    imas_obj.core_profiles.time.resize(1)
    imas_obj.core_profiles.time[0] = 0.0
    imas_obj.core_profiles.put()

    imas_obj.ec_launchers.ids_properties.homogeneous_time = 1
    imas_obj.ec_launchers.time.resize(1)
    imas_obj.ec_launchers.time[0] = 0.0
    imas_obj.ec_launchers.put()

    imas_obj.close()

import numpy as np


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
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" % (p["rdim"], p["zdim"], p["rcentr"], p["rleft"], p["zmid"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" % (p["rmaxis"], p["zmaxis"], p["simag"], p["sibry"], p["bcentr"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" % (p["current"], p["simag"], 0, p["rmaxis"], 0))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" % (p["zmaxis"], 0, p["sibry"], 0, 0))

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


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=2)

    tokamak = sp_read_geqdsk("./g038300.03900")
    sp_write_geqdsk(tokamak, "./tmp.gfile")
    tokamak2 = sp_read_geqdsk("./tmp.gfile")
    pp.pprint(tokamak)

    pp.pprint(tokamak2)

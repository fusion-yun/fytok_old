import numpy as np
import fenics
import mshr

# Create empty Mesh


class EqSolver:

    class SubDomain(Enum):
        Boundary = 1
        Limter = 2
        Coil = 3

    def __init__(self):
        pass

    def create_mesh(self, eqdsk):
        self.domain = mshr.Rectangle(fenics.Point(eqdsk["rleft"], eqdsk["zmid"]-eqdsk["zdim"]/2.0),
                                     fenics.Point(eqdsk["rleft"]+eqdsk["rdim"], eqdsk["zmid"]+eqdsk["zdim"]/2.0))

        self.domain.set_subdomain(EqSolver.SubDomain.Limter, mshr.Polygon(
            [fenics.Point(p) for p in eqdsk["limrz"][::-1]]))
        self.domain.set_subdomain(EqSolver.SubDomain.Boundary, mshr.Polygon(
            [fenics.Point(p) for p in eqdsk["bbsrz"][::-1]]))

        coil_id = 0
        for coil in eqdsk["pf_active/coil"]:
            shape = coil["element/geometry/rectangle"]
            self.domain.set_subdomain(EqSolver.SubDomain.Coil+coil_id,
                                      mshr.Rectangle(fenics.Point(shape["r"], shape["z"]),
                                                     fenics.Point(shape["r"]+shape["width"], shape["z"]+shape["height"])))
            coil_id += 1

        self.mesh = mshr.generate_mesh(domain, 16)

        fenics.plot(mesh)

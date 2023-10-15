from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.utils.logger import logger

if __name__ == "__main__":

    equilibrium = Equilibrium()
    core_profiles = CoreProfiles()
    core_transport = CoreTransport({"model": [{"code": {"name": "neo"}}]})

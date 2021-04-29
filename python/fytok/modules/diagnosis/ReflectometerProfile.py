
# This is file is generated from template
from ..utilities.IDS import IDS

class ReflectometerProfile(IDS):
    r"""Profile reflectometer diagnostic. Multiple reflectometers are considered as independent diagnostics to be handled with different occurrence numbers
        
        Note: ReflectometerProfile is an ids
    """
    IDS="reflectometer_profile"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

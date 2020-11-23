
# This is file is generated from template
from fytok.IDS import IDS

class PulseSchedule(IDS):
    r"""Description of Pulse Schedule, described by subsystems waveform references and an enveloppe around them. The controllers, pulse schedule and SDN are defined in separate IDSs. All names and identifiers of subsystems appearing in the pulse_schedule must be identical to those used in the IDSs describing the related subsystems.
       
        .. note:: PulseSchedule is an ids
    """
    IDS="pulse_schedule"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

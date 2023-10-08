from spdm.data.AoS import AoS


def __getattr__(name: str):

    if name.startswith('_T_'):
        return AoS
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

import subprocess
__version__ = (subprocess.check_output(["git", "describe", "--always", "--dirty"]).strip().decode("utf-8"))

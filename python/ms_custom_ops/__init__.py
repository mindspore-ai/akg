from ms_custom_ops.ms_custom_ops import *

def _int_env():
    """init env."""
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_path, "vendors", "customize")
    origin_env_path = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path + ":" + origin_env_path

_int_env()

import os
import ctypes
import mindspore

def _init_env():
    """init env."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_path, "vendors", "customize")
    origin_env_path = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    if origin_env_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path + ":" + origin_env_path
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path
    
    if os.getenv("ASDOPS_LOG_LEVEL") is None:
        os.environ["ASDOPS_LOG_LEVEL"] = "ERROR"
    if os.getenv("ASDOPS_LOG_TO_STDOUT") is None:
        os.environ["ASDOPS_LOG_TO_STDOUT"] = "1"

    ms_path = os.path.dirname(os.path.abspath(mindspore.__file__))
    internal_lib_path = os.path.join(ms_path, "lib", "plugin", "ascend", "libmindspore_internal_kernels.so")
    ctypes.CDLL(internal_lib_path)

_init_env()

from .ms_custom_ops import *
# Import generated ops interfaces
try:
    from .gen_ops_def import *
except ImportError:
    pass  # Generated files may not exist during development

try:
    from .gen_ops_prim import *
except ImportError:
    pass  # Generated files may not exist during development

# Expose generated interfaces
__all__ = []

# Add ops from gen_ops_def if available
try:
    from . import gen_ops_def
    if hasattr(gen_ops_def, '__all__'):
        __all__.extend(gen_ops_def.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_def) if not name.startswith('_')])
except ImportError:
    pass

# Add ops from gen_ops_prim if available  
try:
    from . import gen_ops_prim
    if hasattr(gen_ops_prim, '__all__'):
        __all__.extend(gen_ops_prim.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_prim) if not name.startswith('_')])
except ImportError:
    pass

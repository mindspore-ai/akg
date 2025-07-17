from ms_custom_ops.ms_custom_ops import *

def _int_env():
    """init env."""
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_path, "vendors", "customize")
    origin_env_path = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    if origin_env_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path + ":" + origin_env_path
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path

_int_env()

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
    import ms_custom_ops.gen_ops_def as gen_ops_def
    if hasattr(gen_ops_def, '__all__'):
        __all__.extend(gen_ops_def.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_def) if not name.startswith('_')])
except ImportError:
    pass

# Add ops from gen_ops_prim if available  
try:
    import ms_custom_ops.gen_ops_prim as gen_ops_prim
    if hasattr(gen_ops_prim, '__all__'):
        __all__.extend(gen_ops_prim.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_prim) if not name.startswith('_')])
except ImportError:
    pass

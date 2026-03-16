"""e000-01-r3gan-baseline — clean R3GAN baseline experiment."""

try:
    from .src import train, get_config
except ImportError:
    pass  # running as a standalone script / pytest rootdir, not as a sub-package


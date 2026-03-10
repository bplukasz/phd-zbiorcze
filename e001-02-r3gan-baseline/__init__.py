"""e001-02-r3gan-baseline — R3GAN architecture baseline experiment."""
try:
    from .src import train, get_config
except ImportError:
    pass  # running as a standalone script / pytest rootdir, not as a sub-package


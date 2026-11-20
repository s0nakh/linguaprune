"""Old name of LinguaSqueeze, kept so code written for `linguaprune` keeps working.

`import linguaprune` and `import linguaprune.<module>` give the same module objects as `linguasqueeze`."""

import importlib
import pkgutil
import sys
import warnings

import linguasqueeze
from linguasqueeze import *  # noqa: F401,F403
from linguasqueeze import __all__, __version__  # noqa: F401

warnings.warn("linguaprune was renamed to linguasqueeze; import linguasqueeze instead", DeprecationWarning, stacklevel=2)

for _info in pkgutil.iter_modules(linguasqueeze.__path__):
    sys.modules[f"{__name__}.{_info.name}"] = importlib.import_module(f"linguasqueeze.{_info.name}")

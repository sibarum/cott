"""
Projection plugins for the COTT solver.

Each .py file in this directory (except base.py and __init__.py) is
auto-imported on package load, triggering its self-registration
into the global registry.
"""

import os
import importlib

_dir = os.path.dirname(__file__)
for filename in os.listdir(_dir):
    if filename.endswith('.py') and filename not in ('__init__.py', 'base.py'):
        modname = filename[:-3]
        importlib.import_module(f'projections.{modname}')

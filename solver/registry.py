"""
General-purpose function registry for the COTT solver.

Usage:
    from registry import register, get, list_all

    # Register anything
    register('projection', 'complex_lie', my_projection, description='...')

    # Retrieve
    proj = get('projection', 'complex_lie')

    # List all in a category
    for name, entry in list_all('projection').items():
        print(name, entry['description'])
"""

_registry = {}


def register(category, name, obj, **metadata):
    """Register an object under a category and name."""
    if category not in _registry:
        _registry[category] = {}
    _registry[category][name] = {'obj': obj, **metadata}


def get(category, name):
    """Retrieve a registered object. Returns None if not found."""
    if category in _registry and name in _registry[category]:
        return _registry[category][name]['obj']
    return None


def get_entry(category, name):
    """Retrieve the full registry entry (obj + metadata). Returns None if not found."""
    if category in _registry and name in _registry[category]:
        return _registry[category][name]
    return None


def list_all(category=None):
    """List all registered entries, optionally filtered by category."""
    if category is not None:
        return dict(_registry.get(category, {}))
    return {cat: dict(entries) for cat, entries in _registry.items()}


def names(category):
    """List all registered names in a category."""
    return list(_registry.get(category, {}).keys())

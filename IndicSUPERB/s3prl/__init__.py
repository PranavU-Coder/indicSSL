import collections, collections.abc
for _n in ("MutableMapping", "Mapping", "Sequence", "Iterable", "Callable", "Set"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

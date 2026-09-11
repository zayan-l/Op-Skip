"""Op-Skip: apply a fixed operator policy to an existing model."""

from .policy import Policy, load_policy


def apply_opskip(model, policy):
    from .patch import apply_opskip as apply

    return apply(model, policy)


def remove_opskip(model):
    from .patch import remove_opskip as remove

    return remove(model)


__all__ = ["Policy", "load_policy", "apply_opskip", "remove_opskip"]

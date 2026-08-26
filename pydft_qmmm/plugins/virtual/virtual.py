"""Plugins for implementing virtual sites.
"""
from __future__ import annotations

__all__ = ["Virtual"]

from typing import TYPE_CHECKING

from pydft_qmmm.utils import virtual_sites
from pydft_qmmm.integrators import IntegratorPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm import System


class Virtual(IntegratorPlugin):
    """Update virtual site positions after integration.
    """
    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to propagate
        virtual sites.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which propagates
            virtual sites.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            if len(system.virtual_site_indices):
                positions = virtual_sites.compute_positions(
                    system,
                    positions
                )
            return positions, velocities
        return inner

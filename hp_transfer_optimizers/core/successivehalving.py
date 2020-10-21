"""
Hyperband related stuff from original hpbandster code, we keep this as we might support
multi fidelity in the future.
"""

import numpy as np

from hp_transfer_optimizers.core.base_iteration import BaseIteration


class SuccessiveHalving(BaseIteration):
    # pylint: disable=unused-argument
    def _advance_to_next_stage(self, config_ids, losses):
        """
            SuccessiveHalving simply continues the best based on the current loss.
        """
        ranks = np.argsort(np.argsort(losses))
        return ranks < self.num_configs[self.stage]

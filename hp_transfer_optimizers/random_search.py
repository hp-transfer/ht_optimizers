import numpy as np

import hp_transfer_optimizers.core.master

from hp_transfer_optimizers.core.successivehalving import SuccessiveHalving


class _RandomSampler:
    """
        class to implement random sampling from a ConfigSpace
    """

    def __init__(self, configspace, logger=None):
        self.configspace = configspace
        self.logger = logger
        self.losses = []

    def new_result(self, job, config_info):  # pylint: disable=unused-argument
        if job.exception is not None:
            self.logger.warning(f"job {job.id} failed with exception\n{job.exception}")

        loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        self.losses.append(loss)

    def get_config(self, budget):  # pylint: disable=unused-argument
        return self.configspace.sample_configuration().get_dictionary(), {}


class RandomSearch(hp_transfer_optimizers.core.master.Master):
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

        self.config_generator = None

        # Hyperband related stuff from original hpbandster code, we keep this as we might
        # support multi fidelity in the future.
        self.eta = eta = 3
        self.min_budget = min_budget = 1
        self.max_budget = max_budget = 1
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(
            eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter)
        )

        self.config.update(
            {
                "eta": eta,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "budgets": self.budgets,
                "max_SH_iter": self.max_SH_iter,
            }
        )

    def run(
        self, configspace, n_iterations, previous_results, trials_until_loss, **kwargs,
    ):
        if previous_results is not None:
            self.logger.warning(
                f"You are using RandomSearch, but previous results is not None"
            )
        self.config_generator = _RandomSampler(
            configspace=configspace, logger=self.logger,
        )
        result = super()._run(
            n_iterations=n_iterations,
            trials_until_loss=trials_until_loss,
            configspace=configspace,
            **kwargs,
        )
        self.iterations.clear()
        return result

    def get_next_iteration(self, iteration, iteration_kwargs=None):
        # Hyperband related stuff from original hpbandster code, we keep this as we might
        # support multi fidelity in the future.
        if iteration_kwargs is None:
            iteration_kwargs = {}
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return SuccessiveHalving(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s - 1) :],
            config_sampler=self.config_generator.get_config,
            **iteration_kwargs,
        )

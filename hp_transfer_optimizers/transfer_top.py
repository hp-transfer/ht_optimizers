import numpy as np

import hp_transfer_optimizers.core.master

from hp_transfer_optimizers._transfer_utils import get_configspace_partitioning
from hp_transfer_optimizers._transfer_utils import project_config
from hp_transfer_optimizers._transfer_utils import project_configs
from hp_transfer_optimizers._transfer_utils import rank_configs
from hp_transfer_optimizers._transfer_utils import sortout_configs
from hp_transfer_optimizers.core.successivehalving import SuccessiveHalving
from hp_transfer_optimizers.gp import GPSampler
from hp_transfer_optimizers.tpe import TPESampler


class _TransferTopSampler:
    def __init__(
        self,
        configspace,
        top_n_percent,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        previous_results=None,
        logger=None,
        use_gp=False
    ):
        self.logger = logger

        self.configspace = configspace
        self.best_previous_config_projected = None
        self.tpe_current = None
        self.configspace_only_new = None
        if previous_results is not None and len(previous_results.batch_results) > 0:
            # Assume same-task changing-configspace trajectory for now
            results_previous_adjustment = previous_results.batch_results[-1]
            config_ranking_previous_adjustment = rank_configs(
                results_previous_adjustment.results[0]
            )

            # 2. Construct intersection / only_new configspace
            (
                configspace_intersection,
                configspace_only_new,
            ) = get_configspace_partitioning(
                self.configspace, results_previous_adjustment.configspace
            )
            self.configspace_only_new = configspace_only_new

            # 3. Project configs to the intersection configspace
            config_ranking_previous_projected = project_configs(
                config_ranking_previous_adjustment, configspace_intersection
            )
            config_ranking_previous_projected = sortout_configs(
                config_ranking_previous_projected, configspace_intersection
            )

            # 4. Read in best previous projected config
            self.best_previous_config_projected = config_ranking_previous_projected[0]

            # 5. Initialize tpe for the only_new configspace
            if len(self.configspace_only_new.get_hyperparameters()) > 0:
                tpe_configspace = self.configspace_only_new
            else:
                tpe_configspace = None
        else:
            tpe_configspace = configspace

        if tpe_configspace is not None:
            if use_gp:
                self.tpe_current = GPSampler(tpe_configspace, logger=self.logger)
            else:
                self.tpe_current = TPESampler(
                    tpe_configspace,
                    top_n_percent,
                    num_samples,
                    random_fraction,
                    bandwidth_factor,
                    min_bandwidth,
                    logger,
                )

    @property
    def configs(self):
        if self.tpe_current is None:
            return None
        return self.tpe_current.configs

    @property
    def losses(self):
        if self.tpe_current is None:
            return None
        return self.tpe_current.losses

    def get_config(self, budget):
        self.logger.debug("start sampling a new configuration.")

        if self.best_previous_config_projected is None:
            sample, _ = self.tpe_current.get_config(budget)
        elif len(self.configspace_only_new.get_hyperparameters()) > 0:
            sample_new, _ = self.tpe_current.get_config(budget)
            sample = {**sample_new, **self.best_previous_config_projected}
        else:
            sample = self.best_previous_config_projected

        info = {}
        self.logger.debug("done sampling a new configuration.")
        return sample, info

    def new_result(self, job, config_info):
        if self.configspace_only_new is not None:
            if len(self.configspace_only_new.get_hyperparameters()) > 0:
                job.kwargs["config"] = project_config(
                    job.kwargs["config"], self.configspace_only_new
                )
                self.tpe_current.new_result(job, config_info)
        else:
            self.tpe_current.new_result(job, config_info)


class TransferTop(hp_transfer_optimizers.core.master.Master):
    def __init__(
        self,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        use_gp=False, **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_gp = use_gp
        self.config_generator = None

        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

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
                "top_n_percent": top_n_percent,
                "num_samples": num_samples,
                "random_fraction": random_fraction,
                "bandwidth_factor": bandwidth_factor,
                "min_bandwidth": min_bandwidth,
            }
        )

    def run(
        self,
        configspace,
        task,
        n_iterations,
        previous_results,
        trials_until_loss,
        **kwargs,
    ):
        self.config_generator = _TransferTopSampler(
            configspace=configspace,
            top_n_percent=self.top_n_percent,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth,
            previous_results=previous_results,
            logger=self.logger,
            use_gp=self.use_gp,
        )
        result = super()._run(
            n_iterations=n_iterations,
            task=task,
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

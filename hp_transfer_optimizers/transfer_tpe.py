import random

import ConfigSpace
import ConfigSpace.util
import hp_transfer_optimizers.core.master
import numpy as np

from hp_transfer_optimizers._transfer_utils import get_configspace_partitioning
from hp_transfer_optimizers._transfer_utils import project_configs
from hp_transfer_optimizers._transfer_utils import rank_configs
from hp_transfer_optimizers._transfer_utils import sortout_configs
from hp_transfer_optimizers.core.successivehalving import SuccessiveHalving
from hp_transfer_optimizers.tpe import TPESampler


def _dict_config_to_array(config, configspace):
    return ConfigSpace.Configuration(configspace, config).get_array()


class _TransferTPESampler:
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
        best_first=True,
        do_ttpe=True,
    ):
        self.do_ttpe = do_ttpe
        self.logger = logger
        self.best_first = best_first

        self.tpe_current = TPESampler(
            configspace,
            top_n_percent,
            num_samples,
            random_fraction,
            bandwidth_factor,
            min_bandwidth,
            logger,
        )

        self.configspace = configspace

        self.first_sample = True
        self.best_previous_config_projected = None
        self.tpe_transfer = None
        if previous_results is not None and len(previous_results.batch_results) > 0:
            # Assume same-task changing-configspace trajectory for now
            results_previous_adjustment = previous_results.batch_results[-1]
            config_ranking_previous_adjustment = rank_configs(
                results_previous_adjustment.results[0]
            )

            # 2. Construct intersection configspace
            (
                self.configspace_intersection,
                _,
                self.configspace_range_only_new,
            ) = get_configspace_partitioning(
                self.configspace, results_previous_adjustment.configspace
            )

            # 3. Project configs to the intersection configspace
            config_ranking_previous_projected = project_configs(
                config_ranking_previous_adjustment, self.configspace_intersection
            )

            # 4. Delete configs which are not in the intersection
            config_ranking_previous_projected = sortout_configs(
                config_ranking_previous_projected, self.configspace_intersection
            )

            # 5. Read in best previous projected config
            self.best_previous_config_projected = config_ranking_previous_projected[0]

            # 6. Build TPE transfer
            config_ranking_previous_projected = [
                _dict_config_to_array(config, self.configspace_intersection)
                for config in config_ranking_previous_projected
            ]
            self.tpe_transfer = TPESampler(
                self.configspace_intersection,
                top_n_percent,
                num_samples,
                random_fraction,
                bandwidth_factor,
                min_bandwidth,
                logger,
                config_ranking_previous_projected,
                list(range(len(config_ranking_previous_projected))),
            )

            # 7.

    @property
    def configs(self):
        return self.tpe_current.configs

    @property
    def losses(self):
        return self.tpe_current.losses

    def get_config(self, budget):
        self.logger.debug("start sampling a new configuration.")

        def fill_intersection(intersection_sample):
            filler_sample = self.configspace.sample_configuration().get_dictionary()
            return {**filler_sample, **intersection_sample}

        if (
            self.best_first
            and self.first_sample
            and self.best_previous_config_projected is not None
        ):
            sample = fill_intersection(self.best_previous_config_projected)
        elif self.tpe_current.has_model:
            sample, _ = self.tpe_current.get_config(budget)
        elif (
            self.do_ttpe and self.tpe_transfer is not None and self.tpe_transfer.has_model
        ):
            intersection_sample, _ = self.tpe_transfer.get_config(budget)

            for hyperparameter in intersection_sample.keys():
                if self.configspace_range_only_new.has_non_empty_only_new_range(
                    hyperparameter
                ):
                    if random.random() < self.configspace_range_only_new.get_modification_probability(
                        hyperparameter
                    ):
                        intersection_sample[
                            hyperparameter
                        ] = self.configspace_range_only_new.modify_hyperparameter(
                            hyperparameter
                        )

            sample = fill_intersection(intersection_sample)
        else:  # no model available at all
            sample = self.configspace.sample_configuration()

        try:
            sample = sample.get_dictionary()
        except AttributeError:
            pass

        info = {}
        self.logger.debug("done sampling a new configuration.")
        self.first_sample = False
        return sample, info

    def new_result(self, job, config_info):
        self.tpe_current.new_result(job, config_info)


class TransferTPE(hp_transfer_optimizers.core.master.Master):
    def __init__(
        self,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        best_first=True,
        do_ttpe=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.do_ttpe = do_ttpe
        self.config_generator = None

        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.best_first = best_first

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
        self.config_generator = _TransferTPESampler(
            configspace=configspace,
            top_n_percent=self.top_n_percent,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth,
            previous_results=previous_results,
            logger=self.logger,
            best_first=self.best_first,
            do_ttpe=self.do_ttpe,
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

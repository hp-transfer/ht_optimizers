import random

import numpy as np
import ConfigSpace
from ConfigSpace import hyperparameters as CSH
from smac.configspace import convert_configurations_to_array
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.epm.gp_kernels import ConstantKernel, Matern, HammingKernel, WhiteKernel
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import LocalSearch
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import StatusType
import hp_transfer_optimizers.core.master
from hp_transfer_optimizers._transfer_utils import get_configspace_partitioning_cond
from hp_transfer_optimizers.core.successivehalving import SuccessiveHalving


def _configspace_to_types_and_bounds(configspace):
    types = []
    bounds = []
    for hyperparameter in configspace.get_hyperparameters():
        is_categorical = isinstance(hyperparameter, CSH.CategoricalHyperparameter)
        if is_categorical:
            types.append(len(hyperparameter.choices))
            bounds .append((len(hyperparameter.choices), np.nan))
        else:
            types.append(0)
            bounds.append((hyperparameter.lower, hyperparameter.upper))
    types = np.array(types, dtype=np.int)
    return types, bounds


def _impute_conditional_data(array, configspace):
    return_array = np.empty_like(array)
    for i in range(array.shape[0]):
        datum = np.copy(array[i])
        nan_indices = np.argwhere(np.isnan(datum.astype(np.float64))).flatten()
        while np.any(nan_indices):
            nan_idx = nan_indices[0]
            valid_indices = np.argwhere(
                np.isfinite(array.astype(np.float64)[:, nan_idx])
            ).flatten()
            if len(valid_indices) > 0:
                # Pick one of them at random and overwrite all NaN values
                row_idx = np.random.choice(valid_indices)
                datum[nan_indices] = array.astype(np.float64)[row_idx, nan_indices]
            else:
                # no good point in the data has this value activated, so fill it with a
                # valid but random value
                hparam_name = configspace.get_hyperparameter_by_idx(nan_idx)
                hparam = configspace.get_hyperparameter(hparam_name)
                if isinstance(hparam, CSH.CategoricalHyperparameter):
                    sample = hparam.sample(np.random.RandomState())
                    # Map to internal representation
                    datum[nan_idx] = hparam.choices.index(sample)
                elif isinstance(hparam, CSH.UniformFloatHyperparameter) or isinstance(hparam, CSH.UniformIntegerHyperparameter):
                    datum[nan_idx] = np.random.random()  # TODO, log sample
                else:
                    raise ValueError
            nan_indices = np.argwhere(np.isnan(datum.astype(np.float64))).flatten()
        return_array[i, :] = datum
    return return_array


def _construct_model(configspace, rng):
    types, bounds = _configspace_to_types_and_bounds(configspace)
    cont_dims = np.nonzero(types == 0)[0]
    cat_dims = np.nonzero(types != 0)[0]

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )
    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in
             range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )
    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in
             range(len(cat_dims))],
            operate_on=cat_dims,
        )
    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    def _impute_inactive(self, X):
        X = X.copy()
        return _impute_conditional_data(X, self.configspace)

    seed = random.randint(0, 100)
    GaussianProcess._impute_inactive = _impute_inactive
    return GaussianProcess(
        configspace=configspace, types=types, bounds=bounds, seed=seed, kernel=kernel
    )


class GPCondSampler:
    def __init__(
        self,
        configspace,
        random_fraction=1 / 2,
        logger=None,
        previous_results=None
    ):
        self.logger = logger
        self.random_fraction = random_fraction

        self.runhistory = RunHistory()
        self.configs = list()
        self.losses = list()
        rng = np.random.RandomState(random.randint(0, 100))

        if previous_results is not None and len(previous_results.batch_results) > 0:
            # Assume same-task changing-configspace trajectory for now
            results_previous_adjustment = previous_results.batch_results[-1]
            configspace_previous = results_previous_adjustment.configspace

            # Construct combined config space
            configspace_combined = ConfigSpace.ConfigurationSpace()
            development_step = CSH.CategoricalHyperparameter("development_step", choices=["old", "new"])
            configspace_combined.add_hyperparameter(
                development_step
            )

            configspace_only_old, configspace_both, configspace_only_new = get_configspace_partitioning_cond(configspace, configspace_previous)

            configspace_combined.add_hyperparameters(configspace_both.get_hyperparameters())
            configspace_combined.add_hyperparameters(configspace_only_old.get_hyperparameters())
            configspace_combined.add_hyperparameters(configspace_only_new.get_hyperparameters())

            for hyperparameter in configspace_only_old.get_hyperparameters():
                configspace_combined.add_condition(
                    ConfigSpace.EqualsCondition(hyperparameter, development_step, "old")
                )
            for hyperparameter in configspace_only_new.get_hyperparameters():
                configspace_combined.add_condition(
                    ConfigSpace.EqualsCondition(hyperparameter, development_step, "new")
                )

            # Read old configs and losses
            result_previous = results_previous_adjustment.results[0]
            all_runs = result_previous.get_all_runs(only_largest_budget=False)
            self.losses_old = [run.loss for run in all_runs]
            self.configs_old = [run.config_id for run in all_runs]
            id2conf = result_previous.get_id2config_mapping()
            self.configs_old = [id2conf[id_]["config"] for id_ in self.configs_old]

            # Map old configs to combined space
            for config in self.configs_old:
                config["development_step"] = "old"
            self.configs_old = [ConfigSpace.Configuration(configspace_combined, config) for config in self.configs_old]

            for config, cost in zip(self.configs_old, self.losses_old):
                self.runhistory.add(config, cost, 0, StatusType.SUCCESS)

            # Construct and fit model
            self.configspace = configspace_combined
            self.model = _construct_model(self.configspace, rng)
            self.acquisition_func = EI(model=self.model)
            self.acq_optimizer = LocalSearch(acquisition_function=self.acquisition_func,
                                             config_space=self.configspace, rng=rng)

            X = convert_configurations_to_array(self.configs_old)
            Y = np.array(self.losses_old, dtype=np.float64)
            self.model.train(X, Y)
            self.acquisition_func.update(
                model=self.model,
                eta=min(self.losses_old),
            )
        else:
            self.configspace = configspace
            self.model = _construct_model(self.configspace, rng)
            self.acquisition_func = EI(model=self.model)
            self.acq_optimizer = LocalSearch(acquisition_function=self.acquisition_func,
                                             config_space=self.configspace, rng=rng)

        self.min_points_in_model = len(self.configspace.get_hyperparameters())  # TODO

    @property
    def has_model(self):
        return len(self.configs) >= self.min_points_in_model

    def get_config(self, budget):  # pylint: disable=unused-argument
        self.logger.debug("start sampling a new configuration.")

        is_random_fraction = np.random.rand() < self.random_fraction
        if is_random_fraction or not self.has_model:
            if "development_step" in self.configspace.get_hyperparameter_names():
                while True:
                    sample = self.configspace.sample_configuration()
                    if sample["development_step"] == "new":
                        break
            else:
                sample = self.configspace.sample_configuration()
        else:
            # Use private _maximize to not return challenger list object
            sample = self.acq_optimizer._maximize(
                runhistory=self.runhistory,
                stats=None,
                num_points=1,
            )
            sample = sample[0][1]

        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(sample.get_dictionary(), self.configspace)
        sample = sample.get_dictionary()
        info = {}
        self.logger.debug("done sampling a new configuration.")
        return sample, info

    def new_result(self, job, config_info):  # pylint: disable=unused-argument
        if job.exception is not None:
            self.logger.warning(f"job {job.id} failed with exception\n{job.exception}")

        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf

        config = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        self.configs.append(config)
        self.losses.append(loss)

        if self.has_model:
            # TODO: include old
            X = convert_configurations_to_array(self.configs)
            Y = np.array(self.losses, dtype=np.float64)
            self.model.train(X, Y)
            self.acquisition_func.update(
                model=self.model,
                eta=min(self.losses),
            )


class GPCond(hp_transfer_optimizers.core.master.Master):
    def __init__(
        self,
        random_fraction=1 / 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config_generator = None

        self.random_fraction = random_fraction

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
                "random_fraction": random_fraction,
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
        self.config_generator = GPCondSampler(
            configspace=configspace,
            random_fraction=self.random_fraction,
            logger=self.logger,
            previous_results=previous_results,
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

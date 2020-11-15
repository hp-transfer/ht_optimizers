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

    seed = random.randint(0, 100)
    return GaussianProcess(
        configspace=configspace, types=types, bounds=bounds, seed=seed, kernel=kernel
    )


class GPSampler:
    def __init__(
        self,
        configspace,
        random_fraction=1 / 2,
        logger=None,
        configs=None,
        losses=None,
    ):
        self.logger = logger

        self.random_fraction = random_fraction
        self.configspace = configspace
        self.min_points_in_model = len(self.configspace.get_hyperparameters())

        rng = np.random.RandomState(random.randint(0, 100))

        self.model = _construct_model(configspace, rng)
        self.acquisition_func = EI(model=self.model)
        self.acq_optimizer = LocalSearch(acquisition_function=self.acquisition_func,
                                    config_space=configspace, rng=rng)
        self.runhistory = RunHistory()

        self.configs = configs or list()
        self.losses = losses or list()
        if self.has_model:
            for config, cost in zip(self.configs, self.losses):
                self.runhistory.add(config, cost, 0, StatusType.SUCCESS)

            X = convert_configurations_to_array(self.configs)
            Y = np.array(self.losses, dtype=np.float64)
            self.model.train(X, Y)
            self.acquisition_func.update(
                model=self.model,
                eta=min(self.losses),
            )

    @property
    def has_model(self):
        return len(self.configs) >= self.min_points_in_model

    def get_config(self, budget):  # pylint: disable=unused-argument
        self.logger.debug("start sampling a new configuration.")

        is_random_fraction = np.random.rand() < self.random_fraction
        if is_random_fraction:
            sample = self.configspace.sample_configuration()
        elif self.has_model:
            # Use private _maximize to not return challenger list object
            sample = self.acq_optimizer._maximize(
                runhistory=self.runhistory,
                stats=None,
                num_points=1,
            )
            sample = sample[0]
        else:
            sample = self.configspace.sample_configuration()

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
            X = convert_configurations_to_array(self.configs)
            Y = np.array(self.losses, dtype=np.float64)
            self.model.train(X, Y)
            self.acquisition_func.update(
                model=self.model,
                eta=min(self.losses),
            )


class GP(hp_transfer_optimizers.core.master.Master):
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
        if previous_results is not None:
            self.logger.warning(f"You are using TPE, but previous results is not None")
        self.config_generator = GPSampler(
            configspace=configspace,
            random_fraction=self.random_fraction,
            logger=self.logger,
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

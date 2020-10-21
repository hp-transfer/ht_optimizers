import ConfigSpace
import ConfigSpace.util
import numpy as np

from scipy import stats as sps
from statsmodels import api as sm

import hp_transfer_optimizers.core.master

from hp_transfer_optimizers.core.successivehalving import SuccessiveHalving


def _generate_candidate(bw_factor, kde_good, min_bandwidth, vartypes):
    idx = np.random.randint(0, len(kde_good.data))
    datum = kde_good.data[idx]
    vector = []
    for m, bw, t in zip(datum, kde_good.bw, vartypes):
        bw = max(bw, min_bandwidth)
        if t == 0:
            bw = bw_factor * bw
            vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
        else:
            if np.random.rand() < (1 - bw):
                vector.append(int(m))
            else:
                vector.append(np.random.randint(t))
    return vector


def _parse_categorical(best_vector, configspace):
    for i, _ in enumerate(best_vector):
        if isinstance(
            configspace.get_hyperparameter(configspace.get_hyperparameter_by_idx(i)),
            ConfigSpace.hyperparameters.CategoricalHyperparameter,
        ):
            best_vector[i] = int(np.rint(best_vector[i]))
    return best_vector


def ei_tpe(x, good_pdf, bad_pdf):
    return max(1e-32, good_pdf(x)) / max(bad_pdf(x), 1e-32)


def _sample_from_ei(
    kde_models, num_samples, vartypes, min_bandwidth, bw_factor, configspace
):
    good_pdf = kde_models["good"].pdf
    bad_pdf = kde_models["bad"].pdf

    candidates = [
        _generate_candidate(bw_factor, kde_models["good"], min_bandwidth, vartypes)
        for _ in range(num_samples)
    ]
    values = [ei_tpe(candidate, good_pdf, bad_pdf) for candidate in candidates]

    best_vector = None
    for value, candidate in sorted(zip(values, candidates), reverse=True):
        if not np.isfinite(value):
            # right now, this happens because a KDE does not contain all values for a
            # categorical parameter this cannot be fixed with the statsmodels KDE, so
            # for now, we are just going to evaluate this one if the good_kde has a
            # finite value, i.e. there is no config with that value in the bad kde, so
            # it shouldn't be terrible.
            if np.isfinite(good_pdf(candidate)):
                best_vector = candidate
                break
        else:
            best_vector = candidate
            break

    if best_vector is None:
        raise ValueError("EI optimization failed")

    best_vector = _parse_categorical(best_vector, configspace)
    return ConfigSpace.Configuration(configspace, vector=best_vector)


def _fit_kde(
    train_configs,
    train_losses,
    kde_vartypes,
    min_bandwidth,
    min_points_in_model,
    top_n_percent,
):
    # Refit KDE for the current budget
    n_good = max(min_points_in_model // 2, (top_n_percent * len(train_configs)) // 100)
    n_bad = len(train_configs) - n_good

    idx = np.argsort(train_losses)
    train_data_good = train_configs[idx[:n_good]]
    train_data_bad = train_configs[idx[-n_bad:]]

    # more expensive crossvalidation method
    # bw_estimation = 'cv_ls'

    # quick rule of thumb
    bw_estimation = "normal_reference"

    bad_kde = sm.nonparametric.KDEMultivariate(
        data=train_data_bad, var_type=kde_vartypes, bw=bw_estimation
    )
    good_kde = sm.nonparametric.KDEMultivariate(
        data=train_data_good, var_type=kde_vartypes, bw=bw_estimation
    )

    bad_kde.bw = np.clip(bad_kde.bw, min_bandwidth, None)
    good_kde.bw = np.clip(good_kde.bw, min_bandwidth, None)
    return good_kde, bad_kde


def _parse_vartypes(config_space):
    kde_vartypes = ""
    vartypes = []
    for h in config_space.get_hyperparameters():
        if hasattr(h, "sequence"):
            raise RuntimeError(
                "This version on BOHB does not support ordinal hyperparameters."
                f"Please encode {h.name} as an integer parameter!"
            )

        if hasattr(h, "choices"):
            kde_vartypes += "u"
            vartypes += [len(h.choices)]
        else:
            kde_vartypes += "c"
            vartypes += [0]
    vartypes = np.array(vartypes, dtype=int)
    return vartypes, kde_vartypes


class TPESampler:
    def __init__(
        self,
        configspace,
        top_n_percent,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        logger=None,
        configs=None,
        losses=None,
    ):
        self.logger = logger

        self.min_bandwidth = min_bandwidth
        self.random_fraction = random_fraction
        self.bw_factor = bandwidth_factor
        self.num_samples = num_samples
        self.top_n_percent = top_n_percent
        self.configspace = configspace

        self.min_points_in_model = 2 * (len(self.configspace.get_hyperparameters()) + 1)
        self.var_types, self.kde_var_types = _parse_vartypes(self.configspace)

        self.configs = configs or list()
        self.losses = losses or list()
        self.kde_models = dict()
        if self.has_model:
            good_kde, bad_kde = _fit_kde(
                np.array(self.configs),
                np.array(self.losses),
                self.kde_var_types,
                self.min_bandwidth,
                self.min_points_in_model,
                self.top_n_percent,
            )
            self.kde_models = {"good": good_kde, "bad": bad_kde}

    @property
    def has_model(self):
        return len(self.configs) >= self.min_points_in_model

    def get_config(self, budget):  # pylint: disable=unused-argument
        self.logger.debug("start sampling a new configuration.")

        model_available = len(self.kde_models.keys()) > 0
        is_random_fraction = np.random.rand() < self.random_fraction
        if is_random_fraction:
            sample = self.configspace.sample_configuration()
        elif model_available:
            sample = _sample_from_ei(
                self.kde_models,
                self.num_samples,
                self.var_types,
                self.min_bandwidth,
                self.bw_factor,
                self.configspace,
            )
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

        # We want to get a numerical representation of the configuration in the
        # original space
        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        self.configs.append(conf.get_array())
        self.losses.append(loss)

        # only fit kdes if enough points are available
        if self.has_model:
            good_kde, bad_kde = _fit_kde(
                np.array(self.configs),
                np.array(self.losses),
                self.kde_var_types,
                self.min_bandwidth,
                self.min_points_in_model,
                self.top_n_percent,
            )
            self.kde_models = {"good": good_kde, "bad": bad_kde}


class TPE(hp_transfer_optimizers.core.master.Master):
    def __init__(
        self,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        if previous_results is not None:
            self.logger.warning(f"You are using TPE, but previous results is not None")
        self.config_generator = TPESampler(
            configspace=configspace,
            top_n_percent=self.top_n_percent,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth,
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

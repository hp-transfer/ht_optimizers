import math
import random

from copy import deepcopy

import ConfigSpace


def rank_configs(result):
    all_runs = result.get_all_runs(only_largest_budget=False)
    ranked_runs = sorted(all_runs, key=lambda run: run.loss)
    ranked_config_ids = [run.config_id for run in ranked_runs]

    id2conf = result.get_id2config_mapping()
    return [id2conf[id_]["config"] for id_ in ranked_config_ids]


class RangeOnlyNewConfigspace:
    def __init__(self, configspace_new):
        self.configspace_new = configspace_new
        self.hyperparameter_to_range = dict()

    def add_categorical_hyperparameter(self, hyperparameter_new, hyperparameter_old):
        only_new = set(hyperparameter_new.choices).difference(
            set(hyperparameter_old.choices)
        )

        if len(only_new) > 0:
            self.hyperparameter_to_range[hyperparameter_new.name] = only_new

    def add_numerical_hyperparameter(self, hyperparameter_new, hyperparameter_old):
        if (
            hyperparameter_new.lower < hyperparameter_old.lower
            and hyperparameter_new.upper > hyperparameter_old.upper
        ):
            only_new = [
                (hyperparameter_new.lower, hyperparameter_old.lower),
                (hyperparameter_old.upper, hyperparameter_new.upper),
            ]
        elif hyperparameter_new.lower < hyperparameter_old.lower:
            only_new = (hyperparameter_new.lower, hyperparameter_old.lower)
        elif hyperparameter_new.upper > hyperparameter_old.upper:
            only_new = (hyperparameter_old.upper, hyperparameter_new.upper)
        else:  # only_new is empty
            return
        self.hyperparameter_to_range[hyperparameter_new.name] = only_new

    def has_non_empty_only_new_range(self, hyperparameter_name):
        return hyperparameter_name in self.hyperparameter_to_range.keys()

    def get_modification_probability(self, hyperparameter_name):
        hyperparameter_new = self.configspace_new.get_hyperparameter(hyperparameter_name)
        only_new_range = self.hyperparameter_to_range[hyperparameter_name]
        if isinstance(hyperparameter_new, ConfigSpace.CategoricalHyperparameter):
            return len(only_new_range) / len(hyperparameter_new.choices)
        else:
            if isinstance(only_new_range[0], tuple):
                if hyperparameter_new.log:
                    nominator = (
                        math.log(only_new_range[0][1]) - math.log(only_new_range[0][0]),
                    )
                    nominator += (
                        math.log(only_new_range[1][1]) - math.log(only_new_range[1][0]),
                    )
                    denominator = (
                        math.log(hyperparameter_new.upper)
                        - math.log(hyperparameter_new.lower),
                    )
                else:
                    nominator = only_new_range[0][1] - only_new_range[0][0]
                    nominator += only_new_range[1][1] - only_new_range[1][0]
                    denominator = hyperparameter_new.upper - hyperparameter_new.lower
                return nominator / denominator
            else:
                if hyperparameter_new.log:
                    return (math.log(only_new_range[1]) - math.log(only_new_range[0])) / (
                        math.log(hyperparameter_new.upper)
                        - math.log(hyperparameter_new.lower)
                    )
                else:
                    return (only_new_range[1] - only_new_range[0]) / (
                        hyperparameter_new.upper - hyperparameter_new.lower
                    )

    def modify_hyperparameter(self, hyperparameter_name):
        hyperparameter_new = self.configspace_new.get_hyperparameter(hyperparameter_name)
        only_new_range = self.hyperparameter_to_range[hyperparameter_name]
        if isinstance(hyperparameter_new, ConfigSpace.CategoricalHyperparameter):
            return random.choice(tuple(only_new_range))
        else:
            if isinstance(only_new_range[0], tuple):
                if hyperparameter_new.log:
                    size_left = math.log(only_new_range[0][1]) - math.log(
                        only_new_range[0][0]
                    )
                    size_right = math.log(only_new_range[1][1]) - math.log(
                        only_new_range[1][0]
                    )
                else:
                    size_left = only_new_range[0][1] - only_new_range[0][0]
                    size_right = only_new_range[1][1] - only_new_range[1][0]

                p_sampling_from_left = size_left / (size_left + size_right)
                sample_range = random.choices(
                    only_new_range,
                    weights=[p_sampling_from_left, 1 - p_sampling_from_left],
                )
                sample_range = sample_range[0]
            else:
                sample_range = only_new_range

            if isinstance(hyperparameter_new, ConfigSpace.UniformFloatHyperparameter):
                return random.uniform(sample_range[0], sample_range[1])
            else:
                return random.randint(sample_range[0], sample_range[1])


def get_configspace_partitioning(configspace_new, configspace_old):
    configspace_intersection = ConfigSpace.ConfigurationSpace()
    configspace_only_new = ConfigSpace.ConfigurationSpace()
    configspace_range_only_new = RangeOnlyNewConfigspace(configspace_new)

    hyperparameters_old = set(configspace_old.get_hyperparameter_names())
    for hyperparameter_new in configspace_new.get_hyperparameters():
        if hyperparameter_new.name in hyperparameters_old:

            hyperparameter_old = configspace_old.get_hyperparameter(
                hyperparameter_new.name
            )
            if isinstance(hyperparameter_new, ConfigSpace.CategoricalHyperparameter):
                both = set(hyperparameter_new.choices).intersection(
                    set(hyperparameter_old.choices)
                )
                if len(both) == 0:
                    configspace_only_new.add_hyperparameter(hyperparameter_new)
                    continue

                hyperparameter_both = ConfigSpace.CategoricalHyperparameter(
                    hyperparameter_new.name, list(both)
                )
                configspace_range_only_new.add_categorical_hyperparameter(
                    hyperparameter_new, hyperparameter_old
                )
            else:
                both_lower = max(hyperparameter_old.lower, hyperparameter_new.lower)
                both_upper = min(hyperparameter_old.upper, hyperparameter_new.upper)
                if both_lower >= both_upper:
                    configspace_only_new.add_hyperparameter(hyperparameter_new)
                    continue

                if isinstance(hyperparameter_new, ConfigSpace.UniformFloatHyperparameter):
                    hyperparameter_both = ConfigSpace.UniformFloatHyperparameter(
                        hyperparameter_new.name, lower=both_lower, upper=both_upper
                    )
                else:
                    hyperparameter_both = ConfigSpace.UniformIntegerHyperparameter(
                        hyperparameter_new.name, lower=both_lower, upper=both_upper
                    )

                configspace_range_only_new.add_numerical_hyperparameter(
                    hyperparameter_new, hyperparameter_old
                )
            configspace_intersection.add_hyperparameter(hyperparameter_both)
        else:
            configspace_only_new.add_hyperparameter(hyperparameter_new)
    return configspace_intersection, configspace_only_new, configspace_range_only_new


def project_config(config, projection_configspace):
    projection_hyperparameters = set(projection_configspace.get_hyperparameter_names())
    for hyperparameter in deepcopy(config).keys():
        if hyperparameter not in projection_hyperparameters:
            del config[hyperparameter]
    return config


def project_configs(configs, projection_configspace):
    return [project_config(config, projection_configspace) for config in configs]


def sortout_configs(configs, projection_configspace):
    def is_still_valid(config, projection_configspace):
        for hyperparameter in projection_configspace.get_hyperparameters():
            if not hyperparameter.is_legal(config[hyperparameter.name]):
                return False
        return True

    return [
        config for config in configs if is_still_valid(config, projection_configspace)
    ]

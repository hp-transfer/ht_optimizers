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


def get_configspace_partitioning(configspace_new, configspace_old):
    configspace_intersection = ConfigSpace.ConfigurationSpace()
    configspace_only_new = ConfigSpace.ConfigurationSpace()

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
            configspace_intersection.add_hyperparameter(hyperparameter_both)
        else:
            configspace_only_new.add_hyperparameter(hyperparameter_new)
    return configspace_intersection, configspace_only_new


def get_configspace_partitioning_cond(configspace_new, configspace_old):
    configspace_both = ConfigSpace.ConfigurationSpace()
    configspace_only_new = ConfigSpace.ConfigurationSpace()
    configspace_only_old = ConfigSpace.ConfigurationSpace()

    hyperparameters_old = set(configspace_old.get_hyperparameter_names())
    hyperparameters_new = set(configspace_new.get_hyperparameter_names())

    hyperparameters_only_old = hyperparameters_old - hyperparameters_new
    hyperparameters_only_new = hyperparameters_new - hyperparameters_old
    hyperparameters_both = hyperparameters_new.intersection(hyperparameters_old)

    for hyperparameter in configspace_old.get_hyperparameters():
        if hyperparameter.name in hyperparameters_only_old:
            configspace_only_old.add_hyperparameter(hyperparameter)

    for hyperparameter in configspace_new.get_hyperparameters():
        if hyperparameter.name in hyperparameters_only_new:
            configspace_only_new.add_hyperparameter(hyperparameter)

    for hyperparameter in hyperparameters_both:
        hyperparameter_new = configspace_new.get_hyperparameter(hyperparameter)
        hyperparameter_old = configspace_old.get_hyperparameter(hyperparameter)
        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            choices_new = set(hyperparameter_new.choices)
            choices_old = set(hyperparameter_old.choices)
            choices_combined = list(choices_new.add(choices_old))
            hyperparameter_both = ConfigSpace.CategoricalHyperparameter(
                hyperparameter, choices_combined
            )
        else:
            both_lower = min(hyperparameter_old.lower, hyperparameter_new.lower)
            both_upper = max(hyperparameter_old.upper, hyperparameter_new.upper)
            if isinstance(hyperparameter_new, ConfigSpace.UniformFloatHyperparameter):
                hyperparameter_both = ConfigSpace.UniformFloatHyperparameter(
                    hyperparameter, lower=both_lower, upper=both_upper
                )
            else:
                hyperparameter_both = ConfigSpace.UniformIntegerHyperparameter(
                    hyperparameter, lower=both_lower, upper=both_upper
                )
        configspace_both.add_hyperparameter(hyperparameter_both)

    return configspace_only_old, configspace_both, configspace_only_new


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

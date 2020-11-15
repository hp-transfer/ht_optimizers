import ConfigSpace as CS
from ConfigSpace import hyperparameters as CSH
import numpy as np

# %%
from smac.configspace import convert_configurations_to_array
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.epm.gp_kernels import ConstantKernel, Matern, HammingKernel, WhiteKernel
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, LocalSearch
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown
from smac.runhistory.runhistory import RunHistory

# %%
from smac.tae.execute_ta_run import StatusType

cs = CS.ConfigurationSpace()
nrounds = CSH.UniformIntegerHyperparameter("nrounds", lower=1, upper=5000)
subsample = CSH.UniformFloatHyperparameter("subsample", lower=0, upper=1)
a = CSH.CategoricalHyperparameter("a", choices=[1, 2])
cs.add_hyperparameters([nrounds, subsample, a])
types = []
bounds = []
for hyperparameter in cs.get_hyperparameters():
    is_categorical = isinstance(hyperparameter, CSH.CategoricalHyperparameter)
    if is_categorical:
        types.append(len(hyperparameter.choices))
        bounds .append((len(hyperparameter.choices), np.nan))
    else:
        types.append(0)
        bounds.append((hyperparameter.lower, hyperparameter.upper))
types = np.array(types, dtype=np.int)

seed = 1
rng = np.random.RandomState()

cont_dims = np.nonzero(types == 0)[0]
cat_dims = np.nonzero(types != 0)[0]


# %%
cov_amp = ConstantKernel(
    2.0,
    constant_value_bounds=(np.exp(-10), np.exp(2)),
    prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
)

if len(cont_dims) > 0:
    exp_kernel = Matern(
        np.ones([len(cont_dims)]),
        [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
        nu=2.5,
        operate_on=cont_dims,
    )

if len(cat_dims) > 0:
    ham_kernel = HammingKernel(
        np.ones([len(cat_dims)]),
        [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
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

# %%
model = GaussianProcess(configspace=cs, types=types, bounds=bounds, seed=seed, kernel=kernel)
acquisition_func = EI(model=model)
acq_optimizer = LocalSearch(acquisition_function=acquisition_func, config_space=cs, rng=rng)
initial_design = LHDesign(cs, rng, None, np.inf, init_budget=len(cs.get_hyperparameters()))
runhistory = RunHistory()

# %%
# Use private API to not have to use traj_logger
initial_configs = initial_design._select_configurations()
initial_costs = [0, 1, 2]
configs = initial_configs
costs = initial_costs
X = convert_configurations_to_array(configs)
Y = np.array(costs, dtype=np.float64)
best_observation = min(costs)

for config, cost in zip(configs, costs):
    runhistory.add(config, cost, 0, StatusType.SUCCESS)

# %%
model.train(X, Y)

# %%
acquisition_func.update(
    model=model,
    eta=best_observation,
)

# %%
# Use private to not return challenger list object
num_points = 2
samples = acq_optimizer._maximize(
    runhistory=runhistory,
    stats=None,
    num_points=num_points,
)

samples = [sample[1] for sample in samples][:num_points]

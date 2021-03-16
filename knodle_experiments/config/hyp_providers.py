from typing import List, Dict
import itertools


def grid_search_configs(base_configs: List[Dict], hyp_param_options=Dict[str, List]):
    configs = []

    keys, values = zip(*hyp_param_options.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for base_config in base_configs:
        for exp in experiments:
            new_config = base_config.copy()
            new_config["hyp_params"] = base_config["hyp_params"].copy()
            for param, value in exp.items():
                new_config["hyp_params"][param] = value
            configs.append(new_config)

    return configs

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

"""
ObjectiveFunction for optmizers
"""

import random
import copy
import numpy as np
from axc_lp import run_paths, plot_samples, token_scenario_baseline
from axc_algobot import AlgoBot, AlgoBotParams


class ObjectiveFunction:
    """
    ObjectiveFunction
    """

    def __init__(self, token_env=token_scenario_baseline):
        self.nround = 0
        self.token_env = copy.deepcopy(token_env)

    def get_function(self):
        """
        return objective function
        """

        def objfunc(x, seed=None):
            print(x)
            #    if (x[0], x[1], x[2], x[3]) in memo:
            #        return memo[(x[0], x[1], x[2], x[3])]
            if seed is not None:
                random.seed(seed)
            tenv = token_scenario_baseline
            tenv.samples = 400
            tenv.steps = 500
            tenv.swap_size = 1000

            samples = run_paths(
                tenv,
                [
                    [
                        [tenv.user_lp, "min_tick", "max_tick"],
                        [10.0 ** x[0], 0.95, 0.98],
                    ]
                ],
                [
                    [
                        [
                            AlgoBot,
                            AlgoBotParams(
                                price_down_gap=x[1],
                                price_down_reset=0.98,
                                price_up_gap=x[2],
                                price_up_reset=1.02,
                            ),
                        ]
                    ]
                ],
                display=False,
            )

            print(np.quantile(samples[0].price, 0.1, axis=0)[-10:-1].mean())
            meandiff = (
                tenv.nav - np.quantile(samples[0].price, 0.10, axis=0)[-50:-1].mean()
            )
            meanhigh = (
                np.quantile(samples[0].price, 0.90, axis=0)[-50:-1].mean() - tenv.nav
            )
            #  print(np.quantile(samples[0].price, 0.90, axis=0))
            reserve0low = np.quantile(samples[0].reserve0, 0.1, axis=0)
            reserve0low[reserve0low > 0.0] = 0.0
            reserve0 = reserve0low[-50:-1].mean()

            reserve1low = np.quantile(samples[0].reserve1, 0.1, axis=0)
            reserve1low[reserve1low > 0.0] = 0.0
            reserve1 = reserve1low[-50:-1].mean()
            print(meandiff, meanhigh, reserve0)
            out = (
                (10.0 ** x[0] / 10**4) ** 2.0
                + (meandiff / 0.01) ** 2.0
                + (meanhigh / 0.01) ** 2.0
                + (reserve0 / 10**4) ** 2.0
                + (reserve1 / 10**4) ** 2.0
            )
            print(
                (10.0 ** x[0] / 10**5) ** 2.0,
                (meandiff / 0.01) ** 2.0,
                (meanhigh / 0.01) ** 2.0,
                (reserve0 / 10**4) ** 2.0,
                (reserve1 / 10**4) ** 2.0,
            )
            if self.nround % 20 == 0:
                plot_samples([samples[0].price])
                plot_samples([samples[0].reserve0])
                plot_samples([samples[0].reserve1])
            print(out)
            #    memo[(x[0], x[1], x[2], x[3])] = out
            self.nround = self.nround + 1
            return out

        return objfunc

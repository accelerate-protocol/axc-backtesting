# %matplotlib ipympl
import ipywidgets as widgets
from uniswappy import *
from axc.lp import *
from axc.algobot import *
from axc.liquidity import *

tenv = token_scenario_baseline
tenv.swap_size = 1000
tenv.tkn_prob = 0.2
tenv.samples = 1
samples = run_paths(
    tenv,
    [
        [
            LiquidityBot(
                LiquidityBotParams(
                    pool_params=[
                        [tenv.user_lp, "min_tick", "max_tick"],
                        [tenv.reserve, tenv.reserve_lower, tenv.nav],
                    ]
                )
            ),
            AlgoBot(
                 AlgoBotParams(
                     redeem_amount0to1=1000,
                     
                 )
            ),
        ]
    ],
)
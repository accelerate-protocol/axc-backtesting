# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

"""
axc_liquidity - Liquidity routines
"""

from dataclasses import dataclass, field
from axc.algobot import AbstractAlgoBot


class LiquidityBot(AbstractAlgoBot):
    """
    add liquidity into bot
    """

    @dataclass
    class Params(AbstractAlgoBot.Params):
        """
        liquidity params
        """

        pool_params: list = field(default_factory=lambda: [])

    def __init__(self, params=Params()):
        super().__init__(params)
        self.account = "lbotuser"

    def init_algo(self, lp):
        return [
            {
                "addliquidity1": {
                    "account": self.account,
                    "amount": pool_param[0],
                    "min_tick": pool_param[1],
                    "max_tick": pool_param[2],
                }
            }
            for pool_param in self.params.pool_params
        ]


LiquidityBotParams = LiquidityBot.Params

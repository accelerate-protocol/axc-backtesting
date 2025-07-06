# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

"""
axc_liquidity - Liquidity routines
"""

from dataclasses import dataclass, field
from axc.algobot import AbstractAlgoBot
from typing import Optional

from uniswappy import (
    UniV3Helper,
    UniV3Utils,
)  # type: ignore


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
        account: str = "lbotuser"
        adapt_nav: Optional[float] = None

    def __init__(self, params=Params()):
        super().__init__(params)
        self.account = params.account
        self.adapt_nav = params.adapt_nav
        self.current_nav = None

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

    def run_algo(self, lp, input_data):
        if self.current_nav is None:
            self.current_nav = input_data["nav"]
        if self.adapt_nav is None:
            return []

        accounts = input_data["accounts"].get(self.account, [])
        nav = input_data["nav"]

        if abs(nav / self.current_nav - 1.0) >= self.adapt_nav:
            self.current_nav = nav
            retval = []
            for low_tick, high_tick, liquidity in accounts:
                amount = lp.convert_to_human(liquidity.liquidity)
                if amount == 0.0:
                    continue
                retval.append(
                    {
                        "removeliquidity": {
                            "account": self.account,
                            "amount": amount,
                            "min_tick_value": low_tick,
                            "max_tick_value": high_tick,
                        }
                    }
                )

                retval.append(
                    {
                        "addliquidity": {
                            "account": self.account,
                            "amount": amount,
                            "min_tick_value": low_tick,
                            "max_tick": input_data["nav"],
                        }
                    }
                )
            return retval
        return []


LiquidityBotParams = LiquidityBot.Params

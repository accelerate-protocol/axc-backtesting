# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

from dataclasses import dataclass, field
from axc_algobot import AbstractAlgoBot, AbstractAlgoBotParams


@dataclass
class LiquidityBotParams(AbstractAlgoBotParams):
    pool_params: list = field(default_factory=lambda: [])


default_params = LiquidityBotParams()


class LiquidityBot(AbstractAlgoBot):
    def __init__(self, params=default_params):
        super().__init__(params)

    def init_algo(self, lp):
        return [{"addliquidity1": pool_param} for pool_param in self.params.pool_params]

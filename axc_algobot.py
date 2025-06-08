# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

from dataclasses import dataclass
import copy
from uniswappy import SolveDeltas, Swap


@dataclass
class AlgoBotParams:
    reserve_tkn0: int
    reserve_tkn1: int


default_params = AlgoBotParams(reserve_tkn0=0, reserve_tkn1=0)


class AbstractAlgoBot:
    def __init__(self, params=default_params):
        self.params = copy.deepcopy(params)

    def init_algo(self, lp=None):
        return []

    def run_algo(self, lp, input_data):
        return []

    def change_reserves(self, amount0, amount1):
        self.params.reserve_tkn0 += amount0
        self.params.reserve_tkn1 += amount1

    @classmethod
    def factory(cls, params=default_params):
        return cls(params)


class NullAlgoBot(AbstractAlgoBot):
    def __init__(self, params=default_params):
        super().__init__(params)


class AlgoBot(AbstractAlgoBot):
    def __init__(self, params=default＿params):
        super().__init__(params)
        self.reserve_wait = False
        self.wait = 0

    def run_algo(self, lp, input_data):
        cmds = []
        if input_data["nsteps"] < self.wait:
            return {}

        nav = input_data["nav"] if lp else None
        if input_data["price"] is not None and nav is not None:
            price = input_data["price"]
            if price < nav - 0.05:
                (x, y) = SolveDeltas(lp).calc(0.98)
                cmds.append({"swap1to0": y})
            elif price > nav + 0.05:
                (x, y) = SolveDeltas(lp).calc(1.02)
                cmds.append({"swap0to1": x})
            if (
                price >= nav - 0.05
                and self.params.reserve_tkn1 <= -5000
                and not self.reserve_wait
            ):
                cmds.append({"redeem": 5000})
        return cmds


class AlgoBotAdapter:
    def __init__(self, lp, account, bot, tkn0, tkn1):
        self.lp = lp
        self.account = account
        self.bot = bot
        self.solver = SolveDeltas(lp)
        self.tkn0 = tkn0
        self.tkn1 = tkn1
        self.nsteps = 0
        self.delay = 50
        self.redeem_queue = {}
        self.nav = 1.0
        self.log = {"cmds": {}, "redemption": {}, "reserve0": [], "reserve1": []}

    def import_state(self):
        return {
            "nav": self.nav,
            "price": self.lp.get_price(self.tkn0),
            "nsteps": self.nsteps,
        }

    def exec(self, cmds):
        for cmd in cmds:
            for k, v in cmd.items():
                if k == "swap0to1":
                    try:
                        out = Swap().apply(self.lp, self.tkn0, self.account, v)
                        self.bot.change_reserves(-v, out)
                    except AssertionError:
                        pass
                if k == "swap1to0":
                    try:
                        out = Swap().apply(self.lp, self.tkn1, self.account, v)
                        self.bot.change_reserves(out, -v)
                    except AssertionError:
                        pass
                if k == "redeem":
                    self.redeem_queue[self.nsteps] = v
                    self.bot.change_reserves(-v, 0)
            new_redeem_queue = {}
            for k, v in self.redeem_queue.items():
                if self.nsteps >= k + self.delay:
                    self.bot.change_reserves(0, v * self.nav)
                    self.log["redemption"][self.nsteps] = v
                else:
                    new_redeem_queue[k] = v
            self.redeem_queue = new_redeem_queue

    def init_step(self):
        cmds = self.bot.init_algo(self.lp)
        self.exec(cmds)

    def run_step(self):
        state = self.import_state()
        cmds = self.bot.run_algo(self.lp, state)
        if cmds:
            self.log["cmds"][self.nsteps] = cmds
        self.exec(cmds)
        self.nsteps = self.nsteps + 1
        self.log["reserve0"].append(self.bot.params.reserve_tkn0)
        self.log["reserve1"].append(self.bot.params.reserve_tkn1)


__all__ = ["AlgoBot", "AlgoBotAdapter", "NullAlgoBot"]

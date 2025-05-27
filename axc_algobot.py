# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

from icecream import ic
from uniswappy import *
from dataclasses import dataclass

@dataclass
class AlgoBotParams:
    reserve_tkn0: int
    reserve_tkn1: int

default_params = AlgoBotParams(
    reserve_tkn0 =  10000,
    reserve_tkn1 = 10000
)

class AbstractAlgoBot:
    def __init__(self, params=default_params):
        self.reserve_tkn0 = params.reserve_tkn0
        self.reserve_tkn1 = params.reserve_tkn1
    def run_algo(self, lp, input_data):
        return {}
    def change_reserves(self, amount0, amount1):
        self.reserve_tkn0 += amount0
        self.reserve_tkn1 += amount1
    @classmethod
    def factory(cls, params=default_params):
        return cls(params)

class NullAlgoBot(AbstractAlgoBot):
    def __init__(self, params=default_params):
        super().__init__(params)

class AlgoBot(AbstractAlgoBot):
    def __init__(self, params=default＿params):
        super().__init__(params)
        self.reserve_tkn0 = params.reserve_tkn0
        self.reserve_tkn1 = params.reserve_tkn1
        self.reserve_wait = False
        self.wait = 0
    def run_algo(self, lp, input):
        cmds = []
        if input['nsteps'] < self.wait:
            return {}
        if input['price'] is not None and \
        input['price'] < input['nav'] - 0.05:
            (x, y) = SolveDeltas(lp).calc(0.98)
            cmds.append({'swap1to0': y})
        if input['price'] is not None and \
        input['price'] > input['nav'] + 0.05:
            (x, y) = SolveDeltas(lp).calc(1.02)
            cmds.append({'swap0to1': x})
        if input['price'] is not None and \
         input['price'] >= 0.95 and self.reserve_tkn1 <= 5000 and not \
        self.reserve_wait:
            cmds.append({'redeem': 5000})
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
        self.log = {
            'cmds': {},
            'redemption': {},
            'reserve0': [],
            'reserve1': []
        }
    def import_state(self):
        return  {
            "nav": self.nav,
            "price": self.lp.get_price(self.tkn0),
            "nsteps": self.nsteps
        }
    def exec(self, cmds):
        for cmd in cmds:
            for k, v in cmd.items():
                if k == "swap0to1":
                    try:
                        out = Swap().apply(
                            self.lp,
                            self.tkn0, 
                            self.account,
                            v
                        )
                        self.bot.change_reserves(-v, out)
                    except AssertionError:
                        pass
                if k == "swap1to0":
                    try:
                        out = Swap().apply(
                            self.lp,
                            self.tkn1, 
                            self.account,
                            v
                        )
                        self.bot.change_reserves(out, -v)
                    except AssertionError:
                        pass
                if k == "redeem":
                    self.redeem_queue[self.nsteps] = v
                    self.bot.change_reserves(0, v)
            new_redeem_queue = {}
            for k, v in self.redeem_queue.items():
                if self.nsteps >= k + self.delay:
                    self.bot.change_reserves(v * self.nav, 0)
                    ic(self.nsteps, "redeem out")
                else:
                    new_redeem_queue[k] = v
            self.redeem_queue = new_redeem_queue
    def run_step(self):
        state = self.import_state()
        cmds = self.bot.run_algo(self.lp, state)
        if cmds:
            self.log['cmds'][self.nsteps] = cmds
        self.exec(cmds)
        self.nsteps = self.nsteps + 1
        self.log['reserve0'].append(self.bot.reserve_tkn0)
        self.log['reserve1'].append(self.bot.reserve_tkn1)


__all__ = ['AlgoBot', 'AlgoBotAdapter', 'NullAlgoBot']
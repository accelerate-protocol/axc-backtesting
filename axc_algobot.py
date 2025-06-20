# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

from dataclasses import dataclass
from collections.abc import Iterable
import copy
from uniswappy import SolveDeltas, Swap, AddLiquidity, UniV3Helper, UniV3Utils


def get_tick(lp, x):
    if x == "min_tick":
        return UniV3Utils.getMinTick(lp.tickSpacing)
    if x == "max_tick":
        return UniV3Utils.getMaxTick(lp.tickSpacing)
    return UniV3Helper().get_price_tick(lp, 0, x)


@dataclass
class AbstractAlgoBotParams:
    reserve_tkn0: int = 0
    reserve_tkn1: int = 0


default_params = AbstractAlgoBotParams()


class AbstractAlgoBot:
    def __init__(self, params=default_params):
        self.params = copy.deepcopy(params)

    def init_algo(self, lp):
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
    pass


@dataclass
class AlgoBotParams(AbstractAlgoBotParams):
    price_down_gap: float = 0.03
    price_down_reset: float = 0.98
    price_down_frac: float = 1.0
    price_up_gap: float = 0.03
    price_up_reset: float = 1.02
    price_up_frac: float = 1.0
    price_redeem: float = 0.95
    redeem_threshold0: float = 0.2
    redeem_threshold1: float = 0.2
    redeem_amount1: float = 0.2
    max_reserve0: int = 50000
    max_reserve1: int = 50000
    redeem_amount0to1: int = 0


default_params_algobot = AlgoBotParams()


class AlgoBot(AbstractAlgoBot):
    def __init__(self, params=default＿params_algobot):
        super().__init__(params)
        self.reserve_wait = False
        self.wait = 0

    def run_algo(self, lp, input_data):
        cmds = []
        if input_data["nsteps"] < self.wait:
            return {}

        nav = input_data["nav"] if lp else None
        price_down = self.params.price_down_reset - self.params.price_down_gap
        price_up = self.params.price_up_reset + self.params.price_up_gap
        if input_data["price"] is not None and nav is not None:
            price = input_data["price"]
            if price < price_down * nav:
                (x, y) = SolveDeltas(lp).calc(nav * self.params.price_down_reset)
                if self.params.max_reserve1 is not None:
                    if self.params.reserve_tkn1 <= -self.params.max_reserve1:
                        y = 0
                    elif self.params.reserve_tkn1 - y <= 0:
                        ynew = abs(self.params.reserve_tkn1) + y
                        y = ynew * self.params.max_reserve1 / (
                            self.params.max_reserve1 + ynew
                        ) - abs(self.params.reserve_tkn1)
                cmds.append({"swap1to0": y * self.params.price_down_frac})
            elif price > nav * price_up:
                (x, y) = SolveDeltas(lp).calc(nav * self.params.price_up_reset)
                if self.params.max_reserve0 is not None:
                    if self.params.reserve_tkn0 <= -self.params.max_reserve0:
                        x = 0
                    elif self.params.reserve_tkn0 - x <= 0:
                        xnew = abs(self.params.reserve_tkn0) + x
                        x = xnew * self.params.max_reserve0 / (
                            self.params.max_reserve0 + xnew
                        ) - abs(self.params.reserve_tkn0)
                cmds.append({"swap0to1": x * self.params.price_up_frac})
            if (
                price >= nav * self.params.price_redeem
                and self.params.max_reserve1 is not None
                and self.params.reserve_tkn0 >= self.params.redeem_amount0to1
                and not self.reserve_wait
                and self.params.redeem_amount0to1 > 0
            ):
                cmds.append({"redeem0to1": self.params.redeem_amount0to1})
        return cmds

    @classmethod
    def factory(cls, params=default_params_algobot):
        return cls(params)


class BotSimulator:
    def __init__(self, lp, account, tkn0, tkn1, bots):
        self.lp = lp
        self.account = account
        self.bots = bots if isinstance(bots, Iterable) else [bots]
        self.solver = SolveDeltas(lp)
        self.tkn0 = tkn0
        self.tkn1 = tkn1
        self.nsteps = 0
        self.delay = 50
        self.redeem_queue = {}
        self.nav = 1.0
        self.log = {"cmds": {}, "pending_redemption0": [], "reserve0": [], "reserve1": [], "nav":[], "nav_net":[]}
        self.pending_redemption0 = 0

    def import_state(self):
        return {
            "nav": self.nav,
            "price": self.lp.get_price(self.tkn0),
            "nsteps": self.nsteps,
        }

    def exec(self, bot, cmds):
        for cmd in cmds:
            for k, v in cmd.items():
                if k == "swap0to1":
                    try:
                        out = Swap().apply(self.lp, self.tkn0, self.account, v)
                        bot.change_reserves(-v, out)
                    except AssertionError:
                        pass
                if k == "swap1to0":
                    try:
                        out = Swap().apply(self.lp, self.tkn1, self.account, v)
                        bot.change_reserves(out, -v)
                    except AssertionError:
                        pass
                if k == "addliquidity0":
                    try:
                        out = AddLiquidity().apply(
                            self.lp,
                            self.tkn0,
                            self.account,
                            v[0],
                            get_tick(self.lp, v[1]),
                            get_tick(self.lp, v[2]),
                        )
                    except AssertionError:
                        pass
                if k == "addliquidity1":
                    try:
                        out = AddLiquidity().apply(
                            self.lp,
                            self.tkn1,
                            self.account,
                            v[0],
                            get_tick(self.lp, v[1]),
                            get_tick(self.lp, v[2]),
                        )
                    except AssertionError:
                        pass
                if k == "redeem0to1":
                    self.redeem_queue[self.nsteps] = v
                    bot.change_reserves(-v, 0)
                    self.pending_redemption0 += v
            new_redeem_queue = {}
            for k, v in self.redeem_queue.items():
                if self.nsteps >= k + self.delay:
                    bot.change_reserves(0, v * self.nav)
                    self.pending_redemption0 -= v
                else:
                    new_redeem_queue[k] = v
            self.redeem_queue = new_redeem_queue

    def init_step(self):
        for bot in self.bots:
            cmds = bot.init_algo(self.lp)
            self.exec(bot, cmds)

    def run_step(self):
        state = self.import_state()
        reserve0 = 0.0
        reserve1 = 0.0
        for bot in self.bots:
            cmds = bot.run_algo(self.lp, state)
            if cmds:
                self.log["cmds"][self.nsteps] = cmds
            self.exec(bot, cmds)
            self.nsteps = self.nsteps + 1
            reserve0 += bot.params.reserve_tkn0
            reserve1 += bot.params.reserve_tkn1
        self.log["reserve0"].append(reserve0)
        self.log["reserve1"].append(reserve1)
        self.log["pending_redemption0"].append(self.pending_redemption0)
        self.log["nav"].append(self.nav)
        self.log["nav_net"].append(reserve0 * self.nav + reserve1) 


__all__ = ["AlgoBot", "BotSimulator", "NullAlgoBot", "AlgoBotParams"]

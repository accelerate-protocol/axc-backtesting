# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

import random
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
from axc_algobot import AlgoBot, BotSimulator, NullAlgoBot
from axc_liquidity import LiquidityBot, LiquidityBotParams
from tqdm.autonotebook import tqdm, trange
from uniswappy import (
    ERC20,
    AddLiquidity,
    EventSelectionModel,
    MockAddress,
    Swap,
    TokenDeltaModel,
    UniswapExchangeData,
    UniswapFactory,
    UniV3Helper,
    UniV3Utils,
)  # type: ignore

# The graphs were taken from notebooks/medium_articles/order_book.ipynb
# in the uniswappy distribution

FEE = UniV3Utils.FeeAmount.MEDIUM
TICK_SPACING = UniV3Utils.TICK_SPACINGS[FEE]
INIT_PRICE = UniV3Utils.encodePriceSqrt(1000, 1000)

@dataclass
class TokenScenario:
    user: str = "user"
    usdt_in: int = 10**6
    user_lp: int = 10000
    reserve: int = 50000
    name0: str = "TKN"
    name1: str = "USDT"
    address0: str = "0x111"
    address1: str = "0x09"
    tick_spacing: Any = TICK_SPACING
    fee: Any = FEE
    init_price: Any = INIT_PRICE
    nav: float = 1.0
    reserve_lower: float = 0.9
    seed: int = 42
    tkn_prob: float = 0.5
    swap_size: int = 1000
    samples: int = 50
    steps: int = 500


@dataclass
class SampleResults:
    price: np.array
    liquidity: np.array
    reserve0: np.array
    reserve1: np.array



token_scenario_baseline = TokenScenario()


def plotme(df, legend, annotation="", group="lower", title=""):
    plt.figure(figsize=(10, 6))
    for key, grp in df.groupby(group):
        plt.plot(grp["swap"], grp["price"], label=key)

    plt.text(25000, 0.6, annotation)
    plt.legend(title=legend)
    plt.title(title)
    plt.xlabel("Tokens sold (TKN)")
    plt.ylabel("Swap price (USDT/TKN)")
    plt.show()


def get_tick(lp, x):
    if x == "min_tick":
        return UniV3Utils.getMinTick(lp.tickSpacing)
    if x == "max_tick":
        return UniV3Utils.getMaxTick(lp.tickSpacing)
    return UniV3Helper().get_price_tick(lp, 0, x)


def setup_lp(tenv):
    factory = UniswapFactory("ETH pool factory", "0x%d")
    tkn0 = ERC20(tenv.name0, tenv.address0)
    tkn1 = ERC20(tenv.name1, tenv.address1)
    exchg_data = UniswapExchangeData(
        tkn0=tkn0,
        tkn1=tkn1,
        symbol="LP",
        address="0x011",
        version="V3",
        tick_spacing=tenv.tick_spacing,
        fee=tenv.fee,
    )
    lp = factory.deploy(exchg_data)
    lp.initialize(tenv.init_price)
    return (lp, tkn0, tkn1)


def do_calc2(tenv, params, names):
    results = []
    for param, name in zip(params, names):
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, tkn1) = setup_lp(tenv)
            adapter = BotSimulator(
                lp,
                tenv.user,
                tkn0,
                tkn1,
                [LiquidityBot(LiquidityBotParams(pool_params=param))],
            )
            adapter.init_step()
            try:
                out = Swap().apply(lp, tkn0, tenv.user, swap)
                results.append(
                    {
                        "lower": name,
                        "swap": swap,
                        "out": out,
                        "price": float(out) / float(swap),
                    }
                )
            except AssertionError:
                pass
    return pd.DataFrame(results)


def make_arrays(lists):
    return np.array(
        [
            np.pad(
                lst,
                (0, max(0, max_len - len(lst))),
                "constant",
                constant_values=(np.nan,),
            )
            for lst in lists
        ],
        dtype=float,
    )


def do_sim(tenv, lp, tkn0, tkn1, nsteps, bot_class=NullAlgoBot, lp_params=None):
    # Set up bot
    bot = bot_class.factory()
    bot_address = MockAddress().apply(1)
    bot_class_list = [bot_class] if not isinstance(bot_class, Iterable) else bot_class
    lp_params = [] if lp_params is None else lp_params

    adapters = [
        BotSimulator(
            lp,
            bot_address[0],
            tkn0,
            tkn1,
            [
                LiquidityBot.factory(LiquidityBotParams(pool_params=lp_params)),
                bot_class.factory(),
            ],
        )
        for bot_class in bot_class_list
    ]
    # Set up liquidity pool
    lp_prices = []
    lp_liquidity = []

    swap_size = tenv.swap_size
    deltas = TokenDeltaModel(swap_size)
    rnd_swap_amounts = [deltas.delta() for _ in range(nsteps)]
    # Run simulation
    for adapter in adapters:
        adapter.init_step()
    for step in range(nsteps):
        accounts = MockAddress().apply(50)
        select_tkn = EventSelectionModel().bi_select(tenv.tkn_prob)
        #        rnd_add_amt = TokenDeltaModel(tenv.swap_size).delta()
        #        user_add = random.choice(accounts)
        user_swap = random.choice(accounts)
        try:
            Swap().apply(
                lp, tkn0 if select_tkn == 0 else tkn1, user_swap, rnd_swap_amounts[step]
            )
        #            lp.summary()
        #            print(select_tkn, rnd_swap_amt, out, lp.get_price(tkn0))
        except AssertionError:
            #            print(traceback.format_exc())
            pass
        lp_prices.append(lp.get_price(tkn0))
        lp_liquidity.append(lp.get_liquidity())
        for adapter in adapters:
            adapter.run_step()
    return np.array(
        [lp_prices, lp_liquidity, adapter.log["reserve0"], adapter.log["reserve1"]]
    )


def do_paths(tenv, lp_params, bot_class=NullAlgoBot):
    samples = []
    for i in trange(tenv.samples):
        lp, tkn0, tkn1 = setup_lp(tenv)
        sample = do_sim(tenv, lp, tkn0, tkn1, tenv.steps, bot_class, lp_params)
        samples.append(sample)
    samples_array = np.transpose(np.array(samples), axes=[1, 0, 2])
    return SampleResults(
        price=samples_array[0],
        liquidity=samples_array[1],
        reserve0=samples_array[2],
        reserve1=samples_array[3],
    )


def plot_path(lp_prices, lp_liquidity):
    fig = plt.figure(figsize=(10, 5))

    fig, (price_ax, liq_ax) = plt.subplots(
        nrows=2, sharex=False, sharey=False, figsize=(12, 8)
    )

    x_val = np.arange(0, len(lp_prices) + 1)
    price_ax.plot(
        x_val[1:-1], lp_prices[1:], color="b", linestyle="dashed", label="lp price"
    )
    price_ax.set_ylabel("Price (TKN/USDT)", size=14)
    price_ax.set_xlabel("Time sample", size=10)
    price_ax.legend()

    liq_ax.plot(
        x_val[1:-1],
        lp_liquidity[1:],
        color="b",
        linestyle="dashed",
        label="lp liquidity",
    )
    liq_ax.set_ylabel("Liquidity", size=14)
    liq_ax.set_xlabel("Time sample", size=10)
    liq_ax.legend()
    plt.tight_layout()


def plot_distribution(samples, title="Price (TKN)", ylow=None, yhigh=None):
    fig, (p_ax) = plt.subplots(nrows=1, sharex=False, sharey=False, figsize=(10, 6))
    xaxis = np.arange(np.shape(samples)[1])

    pymc.gp.util.plot_gp_dist(
        ax=p_ax, x=xaxis, samples=samples, palette="cool", plot_samples=False
    )
    p_ax.plot(xaxis, np.mean(samples, axis=0), color="w", linewidth=3, label="Price")
    p_ax.set_title(title)
    p_ax.legend(facecolor="lightgray", loc="upper left")
    p_ax.set_xlabel("Trades")
    p_ax.set_ylabel("Price (TKN/USDT)")
    p_ax.set_ylim(bottom=ylow, top=yhigh)
    plt.show()


def dump_liquidity(lp, tkn0, tkn1) -> pd.DataFrame:
    """
    Dumps liquidity data from Uniswap V3 pool into DataFrame.

    Args:
        lp (LiquidityPool): The Uniswap V3 liquidity pool instance
        tkn0 (str): Token address for the first token
        tkn1 (str): Token address for the second token

    Returns:
        DataFrame: A DataFrame with columns 'price' and 'liquidity',
            where each row represents a position in the order book.
    """

    try:
        current_price = lp.get_price(tkn1)
    except AssertionError:
        print(f"Error getting price for {tkn1}")
        return pd.DataFrame()

    positions = lp.ticks
    prices = [UniV3Helper().tick_to_price(pos) for pos in positions]
    center_pos = UniV3Helper().price_to_tick(current_price)
    side_arr = []
    for pos in positions:
        if pos > center_pos:
            side_arr.append("asks")
        elif pos < center_pos:
            side_arr.append("bids")
        else:
            side_arr.append("center")

    df_liq = pd.DataFrame(
        {
            "price": prices,
            "side": side_arr,
            "liquidity": [lp.ticks[pos].liquidityGross / 10**18 for pos in positions],
        }
    )

    if "center" in side_arr:
        df_liq = df_liq[~df_liq["side"] == "center"]

    return df_liq


def plot_liquidity(lp, tkn0, tkn1, df_liq):
    fig = plt.figure(figsize=(10, 5))
    fig, (book_ax) = plt.subplots(nrows=1, figsize=(12, 8))
    current_price = lp.get_price(tkn1)
    prices = df_liq["price"].values
    liquidity = df_liq["liquidity"].values

    book_ax.bar(
        prices, liquidity, color="steelblue", width=0.0005, label="liquidity", alpha=0.7
    )
    book_ax.axvline(
        x=current_price,
        color="mediumvioletred",
        linewidth=1,
        linestyle="dashdot",
        label="current price",
    )
    book_ax.set_xlabel("Price (USD)", size=10)
    book_ax.set_ylabel("Liquidity", size=14)
    book_ax.set_title("Uniswap V3: Liquidity distribution")
    book_ax.legend()

    plt.tight_layout()


def run_paths(tenv, params=None, bots=None):
    if params is None:
        params = [
            [[tenv.user_lp, "min_tick", "max_tick"]],
            [
                [tenv.user_lp, "min_tick", "max_tick"],
                [tenv.reserve, tenv.reserve_lower, tenv.nav],
            ],
            [[tenv.user_lp, "min_tick", "max_tick"]],
            [
                [tenv.user_lp, "min_tick", "max_tick"],
                [tenv.reserve, tenv.reserve_lower, tenv.nav],
            ],
        ]

    if bots is None:
        bots = [NullAlgoBot, NullAlgoBot, AlgoBot, AlgoBot]

    samples = []
    for param, bot in tqdm(zip(params, bots), total=len(params)):
        random.seed(tenv.seed)
        sample = do_paths(tenv, param, bot)
        samples.append(sample)
    return samples


def plot_samples(lp_price_samples, ylow=None, yhigh=None):
    return [
        plot_distribution(sample, f"Scenario {i + 1}", ylow, yhigh)
        for (i, sample) in enumerate(lp_price_samples)
    ]


def runme(widgets):
    tenv = token_scenario_baseline
    tenv.token_prob = widgets["token_prob"].value
    tenv.swap_size = widgets["swap_size"].value
    widgets["output"].clear_output()
    with widgets["output"]:
        samples = run_paths(
            tenv,
            [
                [
                    [tenv.user_lp, "min_tick", "max_tick"],
                    [tenv.reserve, tenv.reserve_lower, tenv.nav],
                ]
            ],
            [AlgoBot],
        )
        plot_samples([samples[0].price])
        plot_samples([samples[0].reserve0])
        plot_samples([samples[0].reserve1])


__all__ = [
    "plotme",
    "do_calc2",
    "setup_lp",
    "TokenScenario",
    "do_sim",
    "do_paths",
    "dump_liquidity",
    "plot_liquidity",
    "plot_path",
    "plot_distribution",
    "token_scenario_baseline",
    "run_paths",
    "runme",
    "plot_samples",
]

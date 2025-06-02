# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

import random
import ipywidgets as widgets
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc

from tqdm.autonotebook import tqdm, trange
from uniswappy import (
    UniV3Utils,
    ERC20,
    UniswapFactory,
    AddLiquidity,
    MockAddress,
    TokenDeltaModel,
    EventSelectionModel,
    Swap,
    UniV3Helper,
    UniswapExchangeData,
)
from axc_algobot import NullAlgoBot, AlgoBotAdapter, AlgoBot


# The graphs were taken from notebooks/medium_articles/order_book.ipynb
# in the uniswappy distribution


@dataclass
class TokenScenario:
    user: str
    usdt_in: int
    user_lp: int
    reserve: int
    name0: str
    name1: str
    address0: str
    address1: str
    tick_spacing: any
    fee: any
    init_price: any
    nav: float
    reserve_lower: float
    seed: int
    tkn_prob: float
    swap_size: int


FEE = UniV3Utils.FeeAmount.MEDIUM
TICK_SPACING = UniV3Utils.TICK_SPACINGS[FEE]
INIT_PRICE = UniV3Utils.encodePriceSqrt(1000, 1000)
token_scenario_baseline = TokenScenario(
    user="user",
    user_lp=10000,
    reserve=50000,
    name0="TKN",
    name1="USDT",
    address0="0x111",
    address1="0x09",
    usdt_in=10**6,
    tick_spacing=TICK_SPACING,
    fee=FEE,
    init_price=INIT_PRICE,
    nav=1.0,
    reserve_lower=0.9,
    seed=42,
    tkn_prob=0.5,
    swap_size=1000,
)


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


def setup_lp(tenv, pool_params):
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
    for pool_param in pool_params:
        AddLiquidity().apply(
            lp,
            tkn1,
            tenv.user,
            pool_param[0],
            get_tick(lp, pool_param[1]),
            get_tick(lp, pool_param[2]),
        )
    #    lp.summary()
    return (lp, tkn0, tkn1)


def do_calc(tenv):
    results = []
    for lower in [0.95, 0.9, 0.8, 0.7, 0.5, 0.1, 0.000001]:
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, _) = setup_lp(
                tenv,
                [[tenv.user_lp, "min_tick", "max_tick"], [tenv.reserve, lower, 1.0]],
            )
            try:
                out = Swap().apply(lp, tkn0, tenv.user, swap)
                results.append(
                    {
                        "lower": lower,
                        "swap": swap,
                        "out": out,
                        "price": float(out) / float(swap),
                    }
                )
            except AssertionError:
                pass
    return pd.DataFrame(results)


def do_calc1(tenv):
    results = []
    insurance_lower = 0.95
    for frac_reserve in np.geomspace(0.0001, 0.99, num=10):
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, _) = setup_lp(
                tenv,
                [
                    [tenv.reserve * (1.0 - frac_reserve), "min_tick", "max_tick"],
                    [tenv.reserve * frac_reserve, insurance_lower, 1.0],
                ],
            )
            try:
                out = Swap().apply(lp, tkn0, tenv.user, swap)
                results.append(
                    {
                        "lower": frac_reserve * 100,
                        "swap": swap,
                        "out": out,
                        "price": float(out) / float(swap),
                    }
                )
            except AssertionError:
                pass
    return pd.DataFrame(results)


def do_calc2(tenv, params, names):
    results = []
    for param, name in zip(params, names):
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, _) = setup_lp(tenv, param)
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


def do_sim(tenv, lp, tkn0, tkn1, nsteps, bot_class=NullAlgoBot):
    # Set up bot
    bot = bot_class.factory()
    bot_address = MockAddress().apply(1)
    adapter = AlgoBotAdapter(lp, bot_address[0], bot, tkn0, tkn1)
    # Set up liquidity pool
    lp_prices = []
    lp_liquidity = []

    swap_size = tenv.swap_size
    deltas = TokenDeltaModel(swap_size)
    rnd_swap_amounts = [deltas.delta() for _ in range(nsteps)]
    # Run simulation
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
        adapter.run_step()
    return (np.array(lp_prices), np.array(lp_liquidity), adapter.log)


def do_paths(tenv, npaths, nsteps, lp_params, bot_class=NullAlgoBot):
    lp_price_samples = []
    lp_liquidity_samples = []
    adapter_logs = []
    for i in trange(npaths):
        lp, tkn0, tkn1 = setup_lp(tenv, lp_params)
        lp_prices, lp_liquidity, adapter_log = do_sim(
            tenv, lp, tkn0, tkn1, nsteps, bot_class
        )
        lp_price_samples.append(lp_prices)
        lp_liquidity_samples.append(lp_liquidity)
        adapter_logs.append(adapter_log)
    return (np.array(lp_price_samples), np.array(lp_liquidity_samples), adapter_logs)


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


def plot_distribution(samples, title="Price (TKN)", ylow=0.75, yhigh=1.5):
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
    p_ax.set_ylim([ylow, yhigh])
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
    except:
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

def run_paths(tenv):
    params = [
    [
            [tenv.user_lp, "min_tick", "max_tick"]
    ], [
            [tenv.user_lp, "min_tick", "max_tick"],
        [tenv.reserve, tenv.reserve_lower, tenv.nav]
    ], [
            [tenv.user_lp, "min_tick", "max_tick"]
    ], [
        [tenv.user_lp, "min_tick", "max_tick"],
        [tenv.reserve, tenv.reserve_lower, tenv.nav] 
    ]
    ]

    bots = [
        NullAlgoBot,
        NullAlgoBot,
        AlgoBot,
        AlgoBot
    ]

    random.seed(42)
    lp_price_samples = []
    lp_liquidity_samples = []
    adapter_log = []
    for (param, bot) in tqdm(zip(params, bots), total=len(params)):
        (lp_price_sample, lp_liquidity_sample, adapter_logs) = do_paths(
            tenv, 50, 500, param, bot
        )
        lp_price_samples.append(lp_price_sample)
        lp_liquidity_samples.append(lp_liquidity_sample)
        adapter_logs.append(adapter_log)
    return lp_price_samples, lp_liquidity_samples, adapter_logs

def plot_samples(lp_price_samples, lp_liquidity_samples, adapter_logs):
    return [
        plot_distribution(sample, f"Scenario {i+1}", 0.1, 3.0) \
        for (i, sample) in enumerate(lp_price_samples)
    ]
        
    
def runme(widgets):
    tenv = token_scenario_baseline
    tenv.token_prob = widgets['token_prob'].value
    tenv.swap_size = widgets['swap_size'].value
    with widgets['output']:
        lp_price_samples, lp_liquidity_samples, adapter_logs = \
            run_paths(tenv)
        plot_samples(lp_price_samples, lp_liquidity_samples, adapter_logs)

__all__ = [
    "plotme",
    "do_calc",
    "do_calc1",
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
    "runme"
]

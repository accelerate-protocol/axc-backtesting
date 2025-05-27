# SPDX-License-Identifier: BSD-2-Clause
# Copyright (C) 2025 AXC Laboratories

from uniswappy import *
import pandas as pd
import numpy as np
from icecream import ic
from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass
import traceback
from axc_algobot import *

import matplotlib.pyplot as plt
import pymc

#The graphs were taken from notebooks/medium_articles/order_book.ipynb 
#in the uniswappy distribution

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

def plotme(df, title, annotation):
    plt.figure(figsize=(10,6))
    for key, grp in df.groupby('lower'):
        plt.plot(grp['swap'], grp['price'], label=key)

    plt.text(25000, 0.6, annotation )
    plt.legend(title=title)
    plt.xlabel('Tokens sold (TKN)')
    plt.ylabel('Swap price (USDT/TKN)')
    plt.show()

def get_tick(lp, x):
    if x == "min_tick":
        return UniV3Utils.getMinTick(lp.tickSpacing)
    elif x == "max_tick":
        return UniV3Utils.getMaxTick(lp.tickSpacing)
    return UniV3Helper().get_price_tick(lp, 0, x)

def setup_lp(tenv, pool_params):
    factory = UniswapFactory("ETH pool factory", "0x%d" )
    tkn0 = ERC20(tenv.name0, tenv.address0)
    tkn1 = ERC20(tenv.name1, tenv.address1)
    exchg_data = UniswapExchangeData(
        tkn0 = tkn0, tkn1 = tkn1,
        symbol="LP", 
        address="0x011", version="V3",
        tick_spacing = tenv.tick_spacing,
        fee=tenv.fee
    )
    lp = factory.deploy(exchg_data)
    lp.initialize(tenv.init_price)
    for pool_param in pool_params:
        AddLiquidity().apply(
            lp, tkn1, tenv.user, pool_param[0],
            get_tick(lp, pool_param[1]), 
            get_tick(lp, pool_param[2])
        )
#    lp.summary()
    return (lp, tkn0, tkn1)

def do_calc(tenv):
    results = []
    for lower in [0.95, 0.9, 0.8, 0.7, 0.5, 0.1, 0.000001]:
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, tkn1) = setup_lp(tenv, [[
                tenv.reserve, lower, 1.0
            ]])
            try:
                out = Swap().apply(lp, tkn0, tenv.user, swap)
                results.append({
                    "lower": lower,
                    "swap": swap,
                    "out": out,
                    "price": float(out) / float(swap)
                })
            except AssertionError:
                pass
    return pd.DataFrame(results)

def do_calc1(tenv):
    results = []
    insurance_lower = 0.95
    insurance_upper = 0.98
    for frac_reserve in np.geomspace(0.0001, 0.99, num=10):
        for swap in np.geomspace(100, tenv.usdt_in, num=100):
            (lp, tkn0, tkn1) = setup_lp(
                tenv, [
                    [tenv.reserve * (1.0 - frac_reserve), "min_tick", 1.0],
                    [tenv.reserve * frac_reserve, insurance_lower, 1.0]
                ]
            )
            try:
                out = Swap().apply(lp, tkn0, tenv.user, swap)
                results.append({
                    "lower": frac_reserve * 100,
                    "swap": swap,
                    "out": out,
                    "price": float(out) / float(swap)
                })
            except AssertionError:
                pass
    return pd.DataFrame(results)

def do_sim(tenv, lp, tkn0, tkn1, nsteps, bot_class=NullAlgoBot):
    # Set up bot
    bot = bot_class.factory()
    bot_address = MockAddress().apply(1)
    adapter = AlgoBotAdapter(lp, bot_address[0], bot, tkn0, tkn1)
    # Set up liquidity pool
    frac_reserve = 0.05
    lp_prices = np.array([], dtype=np.float64)
    lp_liquidity = np.array([], dtype=np.float64)
    # Run simulation
    for i in range(nsteps):
        accounts = MockAddress().apply(50)
        select_tkn = EventSelectionModel().bi_select(0.5)
        rnd_add_amt = TokenDeltaModel(1000).delta()
        rnd_swap_amt = TokenDeltaModel(1000).delta()
        user_add = random.choice(accounts)
        user_swap = random.choice(accounts)
        try:
            out = Swap().apply(lp, tkn0 if select_tkn == 0 else tkn1, user_swap, rnd_swap_amt)
#            lp.summary()
#            print(select_tkn, rnd_swap_amt, out, lp.get_price(tkn0))
        except AssertionError:
#            print(traceback.format_exc())
            pass
        lp_prices = np.append(lp_prices, [lp.get_price(tkn0)])
        lp_liquidity = np.append(lp_liquidity, [lp.get_liquidity()])
        adapter.run_step()
    return (lp_prices, lp_liquidity, adapter.log)

def do_paths(tenv, npaths, nsteps, lp_params, bot_class=NullAlgoBot):
    lp_price_samples = np.zeros((npaths, nsteps), dtype=np.float64)
    lp_liquidity_samples = np.zeros((npaths, nsteps), dtype=np.float64)
    adapter_logs = []
    for i in range(npaths):
        (lp, tkn0, tkn1) = setup_lp(tenv, lp_params)
        (lp_prices, lp_liquidity, adapter_log) = do_sim(tenv, lp, tkn0, tkn1, nsteps, bot_class)
        if i % 10 == 0:
            print(i)
        lp_price_samples[i] = lp_prices
        lp_liquidity_samples[i] = lp_liquidity
        adapter_logs.append(adapter_log)
    return (lp_price_samples, lp_liquidity_samples, adapter_logs)

def plot_path(lp_prices, lp_liquidity):
    fig = plt.figure(figsize = (10, 5))

    fig, (price_ax, liq_ax) = plt.subplots(nrows=2, sharex=False, sharey=False, figsize=(12, 8))

    x_val = np.arange(0,len(lp_prices)+1)
    price_ax.plot(x_val[1:-1], lp_prices[1:], color = 'b',linestyle = 'dashed', label='lp price') 
    price_ax.set_ylabel('Price (TKN/USDT)', size=14)
    price_ax.set_xlabel('Time sample', size=10)
    price_ax.legend()

    liq_ax.plot(x_val[1:-1], lp_liquidity[1:], color = 'b',linestyle = 'dashed', label='lp liquidity') 
    liq_ax.set_ylabel('Liquidity', size=14)
    liq_ax.set_xlabel('Time sample', size=10)
    liq_ax.legend()
    plt.tight_layout()

def plot_distribution(samples, title='Price (TKN)', ylim=[0.75, 1.5] ):
    fig, (P_ax) = plt.subplots(nrows=1, sharex=False, sharey=False, figsize=(10, 6))
    xaxis = np.arange(np.shape(samples)[1])

    pymc.gp.util.plot_gp_dist( 
        ax=P_ax,
        x=xaxis,
        samples=samples,
        palette='cool',
        plot_samples=False
    )
    P_ax.plot(xaxis, np.mean(samples, axis=0), color = 'w', linewidth=3, label='Price')
    P_ax.set_title(title, fontsize=20)
    P_ax.legend(fontsize=16, facecolor="lightgray", loc='upper left')
    P_ax.set_xlabel("Trades", fontsize=16)
    P_ax.set_ylabel("Price (TKN/USDT)", fontsize=16)
    P_ax.set_ylim(ylim)

def dump_liquidity():
    liquidity = {}
    lower = 0.9
    (lp, tkn0, tkn1) = setup_lp(tenv, [[
        tenv.reserve, lower, 1.0
    ]])

    df_liq = pd.DataFrame(columns=['tick', 'price', 'liquidity'])
    for k, pos in enumerate(lp.ticks):
        price = UniV3Helper().tick_to_price(pos)
        liq = lp.ticks[pos].liquidityGross/10**18
        df_liq.loc[k] = [pos,price,liq]

    center_pos = UniV3Helper().price_to_tick(lp.get_price(tkn1))
    price = lp.get_price(tkn1)
    df_liq.loc[k+1] = [center_pos,price,0]

    df_liq.sort_values(by=['price'], inplace=True)
    df_liq.reset_index(drop=True, inplace=True)

    side_arr = []
    for tick in df_liq['tick'].values:
        if (tick > center_pos):
            side_arr.append('asks')
        elif (tick < center_pos):
            side_arr.append('bids')
        else:
            side_arr.append('center')
    df_liq['side'] = side_arr
    idx = df_liq.index[df_liq['side'] == 'center']
    df_liq.drop(idx[0], inplace=True)
    return df_liq

def plot_liquidity(df_liq):
    fig = plt.figure(figsize = (10, 5))
    fig, (book_ax) = plt.subplots(nrows=1,figsize=(12, 8))
    current_price = lp.get_price(tkn1)
    prices = df_liq['price'].values
    liquidity = df_liq['liquidity'].values

    book_ax.bar(prices, liquidity, color ='steelblue', width = 0.0005, label = 'liquidity', alpha=0.7)
    book_ax.axvline(x=current_price, color = 'mediumvioletred', linewidth = 1, linestyle = 'dashdot', label = 'current price')
    book_ax.set_xlabel("Price (USD)", size=10)
    book_ax.set_ylabel("Liquidity", size=14)
    book_ax.set_title("Uniswap V3: Liquidity distribution")
    book_ax.legend()

    plt.tight_layout()

__all__ = ['plotme', 'do_calc', 'do_calc1', 'setup_lp', 'TokenScenario',
          'do_sim', 'do_paths', 'dump_liquidity', 'plot_liquidity',
          'plot_path', 'plot_distribution']
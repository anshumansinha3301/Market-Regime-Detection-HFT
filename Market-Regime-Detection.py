"""
HFT Market Regime Detection & Dynamic Quoting Engine
Architecture: Asynchronous Event-Driven Node
Dependencies: hmmlearn, numpy, numba, pandas, scikit-learn
"""

import asyncio
import logging
import time
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import IntEnum

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from numba import njit

# ==============================================================================
# 1. CONFIGURATION & TYPES (simulate: core/config.py)
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HFT_Node")

class Regime(IntEnum):
    QUIET = 0
    NORMAL = 1
    VOLATILE = 2

@dataclass
class SystemConfig:
    symbol: str = "BTC/USDT"
    tick_size: float = 0.1
    lot_size: float = 0.001
    max_position: float = 1.0
    model_path: str = "./models/hmm_regime.pkl"
    feature_window: int = 100
    hmm_components: int = 3
    base_spread_ticks: int = 2

@dataclass
class TradeTick:
    ts: float
    price: float
    qty: float
    is_buyer_maker: bool

@dataclass
class DepthUpdate:
    ts: float
    is_bid: bool
    price: float
    qty: float

# ==============================================================================
# 2. NUMBA OPTIMIZED MATH CORE (simulate: math/microstructure.py)
# ==============================================================================

@njit(cache=True, fastmath=True)
def calc_hurst(prices: np.ndarray) -> float:
    n = len(prices)
    if n < 20: return 0.5
    lags = np.arange(2, min(20, n // 2))
    tau = np.zeros(len(lags))
    for i in range(len(lags)):
        lag = lags[i]
        diff = prices[lag:] - prices[:-lag]
        tau[i] = np.sqrt(np.mean(diff ** 2))
    x = np.log(lags)
    y = np.log(tau)
    # Fast linear regression
    n_lags = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    slope = (n_lags * sum_xy - sum_x * sum_y) / (n_lags * sum_xx - sum_x * sum_x)
    return slope * 2.0

@njit(cache=True, fastmath=True)
def calc_realized_vol(returns: np.ndarray) -> float:
    return np.sqrt(np.sum(returns ** 2))

@njit(cache=True, fastmath=True)
def calc_roll_spread(returns: np.ndarray) -> float:
    if len(returns) < 2: return 0.0
    ret_t = returns[1:]
    ret_t_1 = returns[:-1]
    mean_t = np.mean(ret_t)
    mean_t_1 = np.mean(ret_t_1)
    cov = np.sum((ret_t - mean_t) * (ret_t_1 - mean_t_1)) / (len(ret_t) - 1)
    return 2.0 * np.sqrt(max(0.0, -cov))

# ==============================================================================
# 3. FAST ORDER BOOK RECONSTRUCTION (simulate: data/orderbook.py)
# ==============================================================================

class LocalOrderBook:
    """Maintains an L2 book from delta updates. Optimized for Top-of-Book lookups."""
    def __init__(self, tick_size: float):
        self.tick_size = tick_size
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.best_bid: float = 0.0
        self.best_ask: float = float('inf')

    def apply_update(self, update: DepthUpdate):
        book = self.bids if update.is_bid else self.asks
        if update.qty <= 0:
            book.pop(update.price, None)
        else:
            book[update.price] = update.qty

        # Re-evaluate BBO (Best Bid/Offer)
        if update.is_bid:
            if update.qty > 0 and update.price > self.best_bid:
                self.best_bid = update.price
            elif update.price == self.best_bid and update.qty <= 0:
                self.best_bid = max(self.bids.keys(), default=0.0)
        else:
            if update.qty > 0 and update.price < self.best_ask:
                self.best_ask = update.price
            elif update.price == self.best_ask and update.qty <= 0:
                self.best_ask = min(self.asks.keys(), default=float('inf'))

    def get_bbo(self) -> Tuple[float, float, float, float]:
        """Returns best_bid, bid_qty, best_ask, ask_qty."""
        bq = self.bids.get(self.best_bid, 0.0)
        aq = self.asks.get(self.best_ask, 0.0)
        return self.best_bid, bq, self.best_ask, aq

    def get_microprice(self) -> float:
        bb, bq, ba, aq = self.get_bbo()
        if bq + aq == 0: return (bb + ba) / 2 if ba != float('inf') else bb
        imb = bq / (bq + aq)
        return (bb * (1 - imb)) + (ba * imb)

    def get_queue_imbalance(self) -> float:
        _, bq, _, aq = self.get_bbo()
        tot = bq + aq
        return (bq - aq) / tot if tot > 0 else 0.0

# ==============================================================================
# 4. STREAMING FEATURE ENGINE (simulate: models/features.py)
# ==============================================================================

class OnlineFeaturePipeline:
    """O(1) streaming updates for HMM inputs."""
    def __init__(self, window: int):
        self.window = window
        self.prices = deque(maxlen=window)
        self.returns = deque(maxlen=window)
        
        # Microstructure accumulators
        self.buy_vol = deque(maxlen=window)
        self.sell_vol = deque(maxlen=window)
        self.qimb_hist = deque(maxlen=window)
        
        # OFI tracking
        self.prev_bb, self.prev_bq = 0.0, 0.0
        self.prev_ba, self.prev_aq = float('inf'), 0.0
        self.ofi_buffer = deque(maxlen=window)

    def ingest_trade(self, trade: TradeTick):
        self.prices.append(trade.price)
        if len(self.prices) > 1:
            self.returns.append(np.log(self.prices[-1] / self.prices[-2]))
        else:
            self.returns.append(0.0)

        if trade.is_buyer_maker:
            self.sell_vol.append(trade.qty)
            self.buy_vol.append(0.0)
        else:
            self.buy_vol.append(trade.qty)
            self.sell_vol.append(0.0)

    def ingest_book_snapshot(self, ob: LocalOrderBook):
        bb, bq, ba, aq = ob.get_bbo()
        if ba == float('inf') or bb == 0: return

        # Order Flow Imbalance (OFI) update
        ofi = 0.0
        if self.prev_bb != 0:
            bid_flow = bq if bb > self.prev_bb else (-self.prev_bq if bb < self.prev_bb else bq - self.prev_bq)
            ask_flow = aq if ba < self.prev_ba else (-self.prev_aq if ba > self.prev_ba else aq - self.prev_aq)
            ofi = bid_flow - ask_flow
        
        self.ofi_buffer.append(ofi)
        self.qimb_hist.append(ob.get_queue_imbalance())
        
        self.prev_bb, self.prev_bq = bb, bq
        self.prev_ba, self.prev_aq = ba, aq

    def get_vector(self, current_mp: float) -> Optional[np.ndarray]:
        """Builds the 1D feature array for HMM inference."""
        if len(self.prices) < self.window or len(self.ofi_buffer) < self.window:
            return None

        p_arr = np.array(self.prices)
        r_arr = np.array(self.returns)
        
        realized_vol = calc_realized_vol(r_arr)
        hurst = calc_hurst(p_arr)
        roll = calc_roll_spread(r_arr)
        
        b_vol = sum(self.buy_vol)
        s_vol = sum(self.sell_vol)
        vpin = abs(b_vol - s_vol) / (b_vol + s_vol) if (b_vol + s_vol) > 0 else 0
        
        ofi_sum = sum(self.ofi_buffer)
        avg_qimb = sum(self.qimb_hist) / len(self.qimb_hist)
        
        mp_dev = np.log(current_mp / p_arr[-1]) if p_arr[-1] > 0 else 0.0

        return np.array([
            r_arr[-1], realized_vol, hurst, roll, vpin, ofi_sum, avg_qimb, mp_dev
        ])

# ==============================================================================
# 5. REGIME DETECTION ENGINE (simulate: models/hmm.py)
# ==============================================================================

class MarketRegimeEngine:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = GaussianHMM(
            n_components=config.hmm_components,
            covariance_type="diag", # Diag is more stable for streaming financial data
            n_iter=1000,
            tol=1e-4,
            init_params="stmc"
        )
        self.scaler = StandardScaler()
        self.state_map: Dict[int, Regime] = {}
        self.is_ready = False

    def train(self, historical_features: np.ndarray):
        """Offline training phase."""
        logger.info("Training HMM on historical dataset...")
        X_scaled = self.scaler.fit_transform(historical_features)
        self.model.fit(X_scaled)
        
        # Map hidden states to explicit Volatility Regimes (0=Quiet, 1=Normal, 2=Volatile)
        variances = np.array([np.sum(np.diag(cov)) for cov in self.model.covars_])
        sorted_idx = np.argsort(variances)
        
        self.state_map = {
            sorted_idx[0]: Regime.QUIET,
            sorted_idx[1]: Regime.NORMAL,
            sorted_idx[2]: Regime.VOLATILE
        }
        self.is_ready = True
        logger.info(f"HMM Training Complete. State Variance Mapping: {self.state_map}")

    def infer(self, features: np.ndarray) -> Optional[Regime]:
        """Sub-millisecond online inference."""
        if not self.is_ready: return None
        vec = self.scaler.transform(features.reshape(1, -1))
        # predict() runs Viterbi, which is fast enough for 1D arrays
        raw_state = self.model.predict(vec)[0]
        return self.state_map[raw_state]

# ==============================================================================
# 6. EXECUTION LOGIC (simulate: execution/market_maker.py)
# ==============================================================================

class DynamicMarketMaker:
    """Adjusts quotes based on current inventory and HMM Regime."""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.inventory: float = 0.0
        self.current_regime: Regime = Regime.NORMAL

    def update_regime(self, regime: Regime):
        if regime != self.current_regime:
            logger.info(f"🚨 REGIME SHIFT DETECTED: {self.current_regime.name} -> {regime.name}")
            self.current_regime = regime

    def generate_quotes(self, ob: LocalOrderBook) -> Tuple[float, float]:
        """Calculates Bid/Ask placements."""
        mid = (ob.best_bid + ob.best_ask) / 2
        
        # 1. Base spread determined by Regime
        if self.current_regime == Regime.QUIET:
            spread_mult = 1.0
            skew_aggression = 0.1
        elif self.current_regime == Regime.NORMAL:
            spread_mult = 2.0
            skew_aggression = 0.5
        else: # VOLATILE
            spread_mult = 4.0
            skew_aggression = 1.0 # Heavily skew to dump bad inventory

        half_spread = (self.config.base_spread_ticks * self.config.tick_size) * spread_mult
        
        # 2. Inventory Skew (Avellaneda-Stoikov proxy)
        inv_ratio = self.inventory / self.config.max_position
        skew = inv_ratio * skew_aggression * half_spread

        bid_price = mid - half_spread - skew
        ask_price = mid + half_spread - skew
        
        # Snap to tick size
        bid_price = math.floor(bid_price / self.config.tick_size) * self.config.tick_size
        ask_price = math.ceil(ask_price / self.config.tick_size) * self.config.tick_size

        return bid_price, ask_price

# ==============================================================================
# 7. ASYNC NODE EVENT LOOP (simulate: main.py)
# ==============================================================================

class TradingNode:
    def __init__(self):
        self.config = SystemConfig()
        self.ob = LocalOrderBook(self.config.tick_size)
        self.features = OnlineFeaturePipeline(self.config.feature_window)
        self.hmm = MarketRegimeEngine(self.config)
        self.algo = DynamicMarketMaker(self.config)
        
        self.trade_queue = asyncio.Queue()
        self.depth_queue = asyncio.Queue()
        
        # For mock training
        self._historical_buffer = []

    async def _mock_exchange_feed(self):
        """Simulates incoming high-frequency websocket data."""
        price = 60000.0
        while True:
            # Random walk with volatility clustering simulation
            vol = 0.5 if np.random.random() > 0.95 else 0.1
            price += np.random.normal(0, vol)
            
            # 1. Fire Depth Update
            await self.depth_queue.put(DepthUpdate(
                ts=time.time(), is_bid=True, price=round(price - 0.5, 1), qty=np.random.random()
            ))
            await self.depth_queue.put(DepthUpdate(
                ts=time.time(), is_bid=False, price=round(price + 0.5, 1), qty=np.random.random()
            ))
            
            # 2. Fire Trade occasionally
            if np.random.random() > 0.7:
                is_buyer = bool(np.random.choice([True, False]))
                await self.trade_queue.put(TradeTick(
                    ts=time.time(), price=round(price, 1), qty=0.01, is_buyer_maker=is_buyer
                ))
            
            await asyncio.sleep(0.01) # 10ms tick rate

    async def _process_depth(self):
        while True:
            update: DepthUpdate = await self.depth_queue.get()
            self.ob.apply_update(update)
            self.features.ingest_book_snapshot(self.ob)

    async def _process_trades_and_execute(self):
        while True:
            trade: TradeTick = await self.trade_queue.get()
            self.features.ingest_trade(trade)
            
            mp = self.ob.get_microprice()
            vec = self.features.get_vector(mp)
            
            if vec is not None:
                if not self.hmm.is_ready:
                    # Collect data to bootstrap model (In prod, load pre-trained)
                    self._historical_buffer.append(vec)
                    if len(self._historical_buffer) == 1000:
                        self.hmm.train(np.array(self._historical_buffer))
                else:
                    # Online Inference
                    regime = self.hmm.infer(vec)
                    self.algo.update_regime(regime)
                    
                    # Execute Quotes
                    b_quote, a_quote = self.algo.generate_quotes(self.ob)
                    if np.random.random() > 0.99: # Throttle logging for sanity
                        logger.debug(f"[{regime.name}] Quoting -> B: {b_quote:.1f} | A: {a_quote:.1f}")

    async def run(self):
        logger.info("Initializing Asynchronous HFT Node...")
        await asyncio.gather(
            self._mock_exchange_feed(),
            self._process_depth(),
            self._process_trades_and_execute()
        )

if __name__ == "__main__":
    node = TradingNode()
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        logger.info("Node shut down gracefully.")

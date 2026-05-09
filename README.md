# 🚀 HFT Market Regime Detection & Dynamic Quoting Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-Asynchronous-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-orange)

An institutional-grade, asynchronous High-Frequency Trading (HFT) node built in Python. This system reconstructs Level-2 (L2) Limit Order Book (LOB) updates, computes latency-critical microstructure features in O(1) time, and uses a **Gaussian Hidden Markov Model (HMM)** to infer latent market regimes. 

The inferred regime dynamically dictates the quoting behavior of an integrated Market Maker execution engine, adapting spread widths and inventory skews (inspired by Avellaneda-Stoikov) to survive structural market breaks.

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Mathematical Foundations](#-mathematical-foundations)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Directory Structure](#-directory-structure)
- [Disclaimer](#-disclaimer)

## 🏗 System Architecture

The core runs on a non-blocking `asyncio` event loop designed to handle thousands of tick and depth updates per second.

1. **L2 Book Reconstructor (`LocalOrderBook`)**: Maintains Top-of-Book state without locking overhead.
2. **Streaming Feature Engine (`OnlineFeaturePipeline`)**: Utilizes Ring Buffers (`collections.deque`) to avoid Pandas/NumPy array reallocation penalties during hot-loop execution.
3. **Machine Learning Core (`MarketRegimeEngine`)**: Employs `hmmlearn` to map streaming multidimensional vectors to discrete economic states (Quiet, Normal, Volatile).
4. **Execution Engine (`DynamicMarketMaker`)**: Calculates optimal bid/ask placements based on current microprice, HMM regime, and inventory delta.

## ✨ Key Features

* **Sub-Millisecond Math via Numba JIT**: Critical path calculations (Hurst, Realized Variance, Roll Measure) are compiled to LLVM machine code via `@njit(fastmath=True)`.
* **Latency-Optimized Data Structures**: No `pandas.rolling()` in the live environment. True O(1) updates using differential arithmetic for Order Flow Imbalance (OFI).
* **State-Aligned HMM**: The model automatically re-orders hidden states by variance to ensure consistent regime labeling across model retrains.
* **Adverse Selection Protection**: Dynamically widens spreads and skews quotes to dump inventory during "Volatile" (Regime 2) states.

## 🧮 Mathematical Foundations

The feature matrix fed into the HMM relies on advanced market microstructure metrics:

### 1. Hurst Exponent
Used to measure the long-term memory of the time series to determine if the market is mean-reverting or trending. Values < 0.5 imply mean-reversion, and > 0.5 imply trending behavior.

### 2. Order Flow Imbalance (OFI)
A proxy for buy/sell pressure at the Best Bid and Offer (BBO). It computes the differential changes in quote sizes and prices to identify short-term price pressure.

### 3. Effective Spread (Roll 1984)
Estimates the effective bid-ask spread purely from trade prices, identifying liquidity dry-ups by measuring the serial covariance of successive price changes.

## ⚙️ Installation

**Prerequisites:** Python 3.9+ and a Unix-based OS (Linux/macOS) are highly recommended for `asyncio` performance.

```bash
# Clone the repository
git clone [https://github.com/yourusername/hft-regime-engine.git](https://github.com/yourusername/hft-regime-engine.git)
cd hft-regime-engine

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

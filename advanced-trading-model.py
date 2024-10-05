import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from bayes_opt import BayesianOptimization
import backtrader as bt
from datetime import datetime
import tensorflow as tf
from collections import deque
import random
from arch import arch_model  # For volatility clustering

# 1. Data Preprocessing and Feature Engineering
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    prices = data['Close'].values
    prices = pd.Series(prices).fillna(method='ffill').values
    returns = np.diff(prices) / prices[:-1]
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
    return prices, returns, scaled_returns

def compute_rsi(series, period=14):
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50] * (period - 1), rsi])

def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    exp1 = pd.Series(series).ewm(span=fast_period, adjust=False).mean()
    exp2 = pd.Series(series).ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return macd.values, signal_line.values

def compute_bollinger_bands(series, period=20, num_std_dev=2):
    sma = pd.Series(series).rolling(window=period).mean()
    rstd = pd.Series(series).rolling(window=period).std()
    upper_band = sma + (rstd * num_std_dev)
    lower_band = sma - (rstd * num_std_dev)
    return upper_band.values, lower_band.values

# 2. LSTM Time Series Forecasting
def build_lstm_model(X):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(prices, window_size=50):
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    return np.array(X), np.array(y)

# 3. Monte Carlo Simulation with GBM including GARCH for volatility clustering
def simulate_gbm_garch(S0, mu, sigma, T, dt=1/252, n_steps=1000):
    N = int(T / dt)
    time = np.linspace(0, T, N)
    am = arch_model(sigma, vol='Garch', p=1, q=1)
    garch_fit = am.fit(disp="off")
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * time + sigma * W
    S = S0 * np.exp(X)
    return S

# 4. Bayesian Optimization for Strategy Parameters
def objective(take_profit, stop_loss):
    profits = []
    for i in range(len(prices) - 50):
        entry_price = prices[i]
        take_profit_price = entry_price * (1 + take_profit)
        stop_loss_price = entry_price * (1 - stop_loss)
        future_prices = prices[i:i + 50]

        if any(future_prices >= take_profit_price):
            profits.append(take_profit_price - entry_price)
        elif any(future_prices <= stop_loss_price):
            profits.append(stop_loss_price - entry_price)

    avg_profit = np.mean(profits)
    return avg_profit

# 5. Dynamic Risk Management
def position_sizing(volatility, account_balance, risk_per_trade=0.02):
    position_size = account_balance * risk_per_trade / (volatility * 100)
    return position_size

def trailing_stop(entry_price, current_price, trail_percent=0.02):
    stop_loss_price = entry_price * (1 - trail_percent)
    return max(stop_loss_price, current_price * (1 - trail_percent))

# 6. Backtesting with Backtrader
class MyStrategy(bt.Strategy):
    params = (
        ('take_profit', 0.02),
        ('stop_loss', 0.01),
    )

    def __init__(self):
        self.order = None
        self.buy_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            self.order = self.buy()
            self.buy_price = self.data.close[0]
        else:
            if self.data.close[0] >= self.buy_price * (1 + self.params.take_profit):
                self.sell()
            elif self.data.close[0] <= self.buy_price * (1 - self.params.stop_loss):
                self.sell()

# 7. Deep Q-Networks (DQN) for Reinforcement Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    prices, returns, scaled_returns = load_and_preprocess_data('historical_price_data.csv')
    rsi = compute_rsi(scaled_returns)
    macd, signal_line = compute_macd(scaled_returns)
    upper_band, lower_band = compute_bollinger_bands(scaled_returns)

    # LSTM Forecasting
    X, y = prepare_lstm_data(prices)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm_model(X)
    model.fit(X, y, batch_size=64, epochs=10)
    future_prices = model.predict(X[-1].reshape(1, X.shape[1], 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(future_prices, label='Forecasted Prices (LSTM)')
    plt.legend()
    plt.title("LSTM Price Forecast")
    plt.show()

    # Monte Carlo Simulation
    simulated_prices = simulate_gbm_garch(S0=prices[-1], mu=0.001, sigma=0.02, T=1, n_steps=100)
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_prices)
    plt.title("Simulated GBM with GARCH Price Path")
    plt.show()

    # Bayesian Optimization
    optimizer = BayesianOptimization(f=objective, pbounds={'take_profit': (0.01, 0.05), 'stop_loss': (0.01, 0.03)})
    optimizer.maximize(init_points=5, n_iter=20)
    best_params = optimizer.max['params']
    print(f"Best Take Profit: {best_params['take_profit']}, Best Stop Loss: {best_params['stop_loss']}")

    # Backtesting
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2022, 1, 1), todate=datetime(2023, 1, 1))
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy, take_profit=best_params['take_profit'], stop_loss=best_params['stop_loss'])
    cerebro.run()
    cerebro.plot()

    # DQN Reinforcement Learning
    state_size = len(scaled_returns)
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    batch_size = 32
    for e in range(1000):
        state = np.random.choice(scaled_returns)
        state = np.reshape(state, [1, state_size])
        for time in range(200):
            action = agent.act(state)
            next_state, reward, done = np.random.choice(scaled_returns), 1, False
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print("Training completed.")

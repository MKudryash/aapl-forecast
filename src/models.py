"""Модели прогнозирования. Общий интерфейс:

    model.fit(hist)      -- hist: DataFrame котировок с начала ряда до точки t включительно
    model.predict(hist)  -- массив прогнозов цены закрытия на 1..H_MAX дней вперед

Ни одна модель не видит данных после t: в walk-forward прогоне ей передается только hist.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from config import H_MAX, VAL_SIZE, SEED, NN_SEEDS, REFIT_EVERY
from features import make_features, make_targets

warnings.filterwarnings("ignore")
STEPS = np.arange(1, H_MAX + 1)


# ---------------------------------------------------------------- baseline
class Naive:
    """Случайное блуждание: прогноз равен последней известной цене."""
    name = "Naive"

    def fit(self, hist):
        pass

    def predict(self, hist):
        return np.repeat(hist["close"].iloc[-1], H_MAX)


class Drift:
    """Случайное блуждание со сносом: средний лог-прирост за всю историю."""
    name = "Drift"

    def fit(self, hist):
        c = np.log(hist["close"].values)
        self.mu = (c[-1] - c[0]) / (len(c) - 1)

    def predict(self, hist):
        return hist["close"].iloc[-1] * np.exp(self.mu * STEPS)


# ---------------------------------------------------------------- статистика
class ARIMA:
    """ARIMA(p,1,q) на логарифме цены. Порядок и наличие сноса выбираются по AIC
    один раз на начальной обучающей выборке, затем модель переоценивается в каждой точке."""
    name = "ARIMA"

    def __init__(self, order=None, trend=None):
        self.order, self.trend = order, trend

    def _select(self, y):
        from statsmodels.tsa.arima.model import ARIMA as SM
        best = (np.inf, None, None)
        self.grid_ = []
        for p, q in itertools.product(range(3), range(3)):
            for trend in ("n", "t"):
                try:
                    aic = SM(y, order=(p, 1, q), trend=trend).fit().aic
                except Exception:
                    continue
                self.grid_.append({"p": p, "q": q, "trend": trend, "aic": aic})
                if aic < best[0]:
                    best = (aic, (p, 1, q), trend)
        return best[1], best[2]

    def fit(self, hist):
        from statsmodels.tsa.arima.model import ARIMA as SM
        y = np.log(hist["close"].values)
        if self.order is None:
            self.order, self.trend = self._select(y)
        self.res = SM(y, order=self.order, trend=self.trend).fit()

    def predict(self, hist):
        return np.exp(self.res.forecast(H_MAX))


class ETS:
    """Экспоненциальное сглаживание Холта с затухающим трендом на логарифме цены."""
    name = "ETS"

    def fit(self, hist):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = np.log(hist["close"].values)
        self.res = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()

    def predict(self, hist):
        return np.exp(self.res.forecast(H_MAX))


# ---------------------------------------------------------------- ML
LGB_GRID = [
    dict(num_leaves=n, learning_rate=lr, n_estimators=ne, min_child_samples=mcs)
    for n in (4, 8) for lr in (0.03,) for ne in (100, 300) for mcs in (10, 20)
]


class LightGBM:
    """Градиентный бустинг, прямая стратегия: отдельная модель для каждого горизонта h,
    цель -- накопленная лог-доходность log(P[t+h]/P[t]). Гиперпараметры подбираются
    один раз на хвосте начальной обучающей выборки (VAL_SIZE точек)."""
    name = "LightGBM"

    def __init__(self, params=None):
        self.params = params

    @staticmethod
    def _xy(hist, h):
        X = make_features(hist)
        y = make_targets(hist, [h])[h]
        ok = X.notna().all(axis=1) & y.notna()
        return X[ok], y[ok]

    @staticmethod
    def _model(p):
        import lightgbm as lgb
        return lgb.LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 reg_lambda=1.0, random_state=SEED, verbose=-1, **p)

    def _tune(self, hist):
        cut = len(hist) - VAL_SIZE
        scores = []
        for p in LGB_GRID:
            err = []
            for h in (1, H_MAX):
                X, y = self._xy(hist, h)
                tr = X.index + h <= cut - 1          # цель обучения целиком до начала валидации
                va = (X.index >= cut) & (X.index + h <= len(hist) - 1)
                m = self._model(p).fit(X[tr], y[tr])
                err.append(np.mean(np.abs(m.predict(X[va]) - y[va])))
            scores.append(np.mean(err))
        self.tuning_ = pd.DataFrame(LGB_GRID).assign(val_mae_logret=scores)
        return LGB_GRID[int(np.argmin(scores))]

    def fit(self, hist):
        if self.params is None:
            self.params = self._tune(hist)
        self.models = {}
        for h in STEPS:
            X, y = self._xy(hist, h)
            self.models[h] = self._model(self.params).fit(X, y)

    def predict(self, hist):
        x = make_features(hist).iloc[[-1]]
        cum = np.array([self.models[h].predict(x)[0] for h in STEPS])
        return hist["close"].iloc[-1] * np.exp(cum)


# ---------------------------------------------------------------- нейросеть
class GRU:
    """Небольшая рекуррентная сеть GRU. Вход: окно из L последних дней
    (лог-доходность, лог-размах high/low), выход: сразу 5 накопленных лог-доходностей.
    Ансамбль из нескольких seed, ранняя остановка по последним 15% окон обучения."""
    name = "GRU"

    def __init__(self, lookback=None, hidden=None, seeds=NN_SEEDS, epochs=300, patience=25):
        self.L, self.hidden = lookback, hidden
        self.seeds, self.epochs, self.patience = seeds, epochs, patience

    @staticmethod
    def _inputs(hist):
        r = np.log(hist["close"]).diff()
        hl = np.log(hist["high"] / hist["low"])
        return np.column_stack([r, hl])

    def _windows(self, hist, L, with_y=True):
        Z = self._inputs(hist)
        c = np.log(hist["close"].values)
        X, Y = [], []
        last = len(hist) - 1 - (H_MAX if with_y else 0)
        for t in range(L, last + 1):
            X.append(Z[t - L + 1:t + 1])
            if with_y:
                Y.append(c[t + STEPS] - c[t])
        return np.array(X), np.array(Y)

    def _train(self, X, Y, hidden, seed):
        import torch
        from torch import nn
        torch.manual_seed(seed)
        np.random.seed(seed)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.gru = nn.GRU(X.shape[2], hidden, batch_first=True)
                s.drop = nn.Dropout(0.2)
                s.out = nn.Linear(hidden, H_MAX)

            def forward(s, x):
                _, hN = s.gru(x)
                return s.out(s.drop(hN[-1]))

        n_val = max(10, int(0.15 * len(X)))
        Xt, Yt = torch.tensor(X[:-n_val], dtype=torch.float32), torch.tensor(Y[:-n_val], dtype=torch.float32)
        Xv, Yv = torch.tensor(X[-n_val:], dtype=torch.float32), torch.tensor(Y[-n_val:], dtype=torch.float32)
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss_fn(net(Xt[b]), Yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xv), Yv).item()
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, best

    def _fit_cfg(self, hist, L, hidden, seeds):
        import torch
        X, Y = self._windows(hist, L)
        mu, sd = X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-12
        ysd = Y.std(0) + 1e-12
        nets, vals = [], []
        for s in seeds:
            net, v = self._train((X - mu) / sd, Y / ysd, hidden, s)
            nets.append(net)
            vals.append(v)
        return dict(nets=nets, mu=mu, sd=sd, ysd=ysd, L=L, val=float(np.mean(vals)))

    def _tune(self, hist):
        rows = []
        for L, hidden in itertools.product((10, 20), (8, 16)):
            st = self._fit_cfg(hist.iloc[:len(hist) - VAL_SIZE], L, hidden, [0])
            # ошибка на отложенной части начальной выборки (без утечки)
            err = []
            for t in range(len(hist) - VAL_SIZE, len(hist) - H_MAX):
                p = self._predict_state(st, hist.iloc[:t + 1])
                c = np.log(hist["close"].values)
                err.append(np.mean(np.abs(np.log(p) - c[t + STEPS])))
            rows.append({"lookback": L, "hidden": hidden, "val_mae_logprice": float(np.mean(err))})
        self.tuning_ = pd.DataFrame(rows)
        b = self.tuning_.iloc[self.tuning_["val_mae_logprice"].idxmin()]
        return int(b["lookback"]), int(b["hidden"])

    def _predict_state(self, st, hist, member=None):
        import torch
        Z = self._inputs(hist)[-st["L"]:]
        x = torch.tensor(((Z - st["mu"]) / st["sd"])[None], dtype=torch.float32)
        with torch.no_grad():
            outs = np.array([n(x).numpy()[0] * st["ysd"] for n in st["nets"]])
        self.last_members = hist["close"].iloc[-1] * np.exp(outs)
        return hist["close"].iloc[-1] * np.exp(outs.mean(0))

    def fit(self, hist):
        if self.L is None:
            self.L, self.hidden = self._tune(hist)
        self.state = self._fit_cfg(hist, self.L, self.hidden, self.seeds)

    def predict(self, hist):
        return self._predict_state(self.state, hist)


def all_models():
    """Модели основного эксперимента и частота переобучения (в точках прогноза, см. config.REFIT_EVERY)."""
    models = [Naive(), Drift(), ARIMA(), ETS(), LightGBM(), GRU()]
    return [(m, REFIT_EVERY[m.name]) for m in models]

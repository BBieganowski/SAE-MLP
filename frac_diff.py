import numpy as np
import pandas as pd

class FractionalDifferentiator:
    def __init__(self, d=0.5, thresh=1e-5):
        self.d = d
        self.thresh = thresh

    def get_weights(self, size: int):
        w = [1.0]
        k = 1
        while True:
            w_ = -w[-1] * (self.d - (k - 1)) / k
            if abs(w_) < self.thresh:
                break
            w.append(w_)
            k += 1
            if k > size:
                break
        return np.array(w[::-1])

    def frac_diff(self, series: pd.Series) -> pd.Series:
        w = self.get_weights(len(series))
        width = len(w)
        
        out = pd.Series(index=series.index, dtype=float)
        for i in range(width, len(series)):
            out.iloc[i] = np.dot(w, series.iloc[i - width : i])
        return out.ffill().fillna(0)

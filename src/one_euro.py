"""One-Euro filter for smoothing noisy landmark streams.

Rejects high-frequency jitter aggressively at rest and eases up during
fast motion, unlike a plain exponential filter which trades lag for
smoothness the same way at every speed. Reference:
Casiez, Roussel, Vogel (2012), "1 Euro Filter".
"""

import math


class _LowPass:
    __slots__ = ("_y", "_a", "initialized")

    def __init__(self):
        self._y = 0.0
        self._a = 1.0
        self.initialized = False

    def filter(self, x, alpha):
        if not self.initialized:
            self._y = x
            self.initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        self._a = alpha
        return self._y

    def last(self):
        return self._y


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroScalar:
    """Single-channel One-Euro filter."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff",
                 "_x", "_dx", "_t_prev")

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev = None

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev = None

    def filter(self, x, t):
        if self._t_prev is None:
            self._t_prev = t
            self._x.filter(x, 1.0)
            self._dx.filter(0.0, 1.0)
            return x
        dt = max(t - self._t_prev, 1e-3)
        self._t_prev = t

        prev_x = self._x.last()
        dx = (x - prev_x) / dt
        edx = self._dx.filter(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(x, _alpha(cutoff, dt))


class OneEuroVec3:
    """Three-channel One-Euro filter for (x, y, z) points."""

    __slots__ = ("_fx", "_fy", "_fz")

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self._fx = OneEuroScalar(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroScalar(min_cutoff, beta, d_cutoff)
        self._fz = OneEuroScalar(min_cutoff, beta, d_cutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()

    def filter(self, xyz, t):
        return (
            self._fx.filter(xyz[0], t),
            self._fy.filter(xyz[1], t),
            self._fz.filter(xyz[2], t),
        )

import numpy as np

class OrnsteinUhlenbeckNoise():
    """
    Ornstein-Uhlenbeck noise process approximated with the Euler-Maruyama numerical scheme.
    Based on the Stable Baselines3 implementation 
    found at https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/noise.py (under MIT License)
    """

    def __init__(self, mean, sigma, theta, dt, init_noise):
        self._mu = mean
        self._theta = theta
        self._sigma = sigma
        self._dt = dt
        self.noise_prev = np.zeros_like(self._mu)

    def sample(self, scaling=1.0):

        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self.noise_prev = noise

        return noise * scaling
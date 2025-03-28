import george
import numpy as np
from scipy import integrate, optimize


class Model(object):

    def update(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GPModel(Model):
    """
    TODO: add error if no data is attached
    """

    def __init__(self, data=None, kernel=None, yerr=0.1):
        self.kernel = kernel
        self.gp = george.GP(kernel)
        self.data = data
        self.yerr = yerr
        self.half_log_2pi = 0.9189385332046727

    def nll(self, params):
        self.gp.set_parameter_vector(params)
        ll = self.gp.log_likelihood(self.data.y, quiet=True)
        grad = self.gp.grad_log_likelihood(self.data.y, quiet=True)
        return -ll, -grad

    def update(self, hyperparameter_optimization=True, initial_jitter=1e-6, max_jitter=1e+300):
        """
        更新高斯过程模型的状态，包括重新计算协方差矩阵和优化超参数。

        参数:
            hyperparameter_optimization (bool): 是否进行超参数优化。
            initial_jitter (float): 初始噪声强度（用于增强数值稳定性）。
            max_jitter (float): 最大允许的噪声强度。
        """
        if self.data:
            # 动态调整噪声强度，确保协方差矩阵正定
            current_jitter = initial_jitter
            while current_jitter < max_jitter:
                try:
                    # 添加噪声项并重新计算协方差矩阵
                    yerr_with_jitter = np.sqrt(self.yerr ** 2 + current_jitter)
                    self.gp.compute(self.data.x, yerr=yerr_with_jitter)
                    break  # 如果成功，退出循环
                except np.linalg.LinAlgError:
                    # 如果 Cholesky 分解失败，增加噪声
                    current_jitter *= 1e+5
            else:
                # 如果噪声强度超过最大值仍然失败，抛出异常
                raise ValueError("即使增加噪声也无法使协方差矩阵正定！")

    # def update(self, hyperparameter_optimization=True):
    #         if self.data:
    #             self.gp.compute(self.data.x, yerr=self.yerr)
            # 超参数优化（保持原有逻辑不变）
            if hyperparameter_optimization:
                params = self.gp.get_parameter_vector()
                soln = optimize.minimize(self.nll, params, jac=True)
                self.soln = soln
                self.gp.set_parameter_vector(soln.x)

    def predict(self, x):
        return self.gp.predict(self.data.y, x, return_var=True)

    def log_likelihood(self):
        return self.gp.log_likelihood(self.data.y) + self.gp.log_prior()

    def log_evidence(self, bic=False):
        if self.data:
            n = len(self.data.y)     # number of observations
        else:
            n = 1
        k = len(self.gp.parameter_vector)    # number of parameters

        # Negative of Bayesian information criterion (BIC)
        if bic:
            return 2 * self.log_likelihood() - k * np.log(n)

        # Laplace Approximation to the model evidence
        eigenvalues, eigenvectors = np.linalg.eigh(self.soln.hess_inv)  # 对称矩阵的特征值分解
        eigenvalues = np.maximum(eigenvalues, 1e-6)  # 将所有特征值限制为 >= 1e-6
        hess_inv_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        chol_hess_inv = np.linalg.cholesky(hess_inv_fixed)
        half_log_det_hess_inv = np.sum(np.log(np.diag(chol_hess_inv)))
        return self.log_likelihood() + k * self.half_log_2pi + half_log_det_hess_inv
        # chol_hess_inv = np.linalg.cholesky(self.soln.hess_inv)
        # half_log_det_hess_inv = np.sum(np.log(np.diag(chol_hess_inv)))
        # return self.log_likelihood() + k * self.half_log_2pi + half_log_det_hess_inv

    def entropy(self, points):
        (mean, covariance) = self.predict(points)
        return 0.5 + self.half_log_2pi + np.log(covariance) * 0.5

    def sample(self, points):
        return self.gp.sample(points)

    def __repr__(self):
        return str(self.gp.kernel)
    


class GrammarModels(object):
    """A collections of models from the grammar of kernels."""

    def __init__(self, base_kernels=["LIN", "PER", "K"], ndim=1, max_depth=2, data=None):
        self.max_depth = max_depth
        self.base_kernels = base_kernels
        self.kernels = self._build_kernels(self.base_kernels, ndim)
        self.data = data
        self._models = [GPModel(kernel=k, data=self.data)
                        for k in self.kernels]

    @property
    def _kernel_lookup(self):
        return {
            "RQ": (george.kernels.RationalQuadraticKernel,
                   {"metric": 5.0, "log_alpha": 2}),
            "M32": (george.kernels.Matern32Kernel, {"metric": 5.0}),
            "M52": (george.kernels.Matern52Kernel, {"metric": 5.0}),
            "E": (george.kernels.ExpKernel, {"metric": 5.0}),
            "SE": (george.kernels.ExpSquaredKernel, {"metric": 5.0}),
            "ES2": (george.kernels.ExpSine2Kernel,
                    {"gamma": 0.1, "log_period": -1}),
            "PER": (george.kernels.CosineKernel, {"log_period": 0.25}),
            "K": (george.kernels.ConstantKernel, {"log_constant": 0}),
            "LIN": (george.kernels.LinearKernel,
                    {"order": 1, "log_gamma2": 1}),
            "DP": (george.kernels.DotProductKernel, {}),
            "LG": (george.kernels.LocalGaussianKernel,
                   {"location": 0.5, "log_width": -1}),
            "POLY": (george.kernels.PolynomialKernel,
                     {"order": 2, "log_sigma2": 2}),
        }

    def _build_kernels(self, kernel_names, ndim=1):
        # TODO as suggestion: change for namedtuple
        # TODO: Remove duplicates due to commutativity of + and *.

        kernels = [self._kernel_lookup[name] for name in kernel_names]

        operators = [
            george.kernels.Product,
            george.kernels.Sum,
        ]

        models = []

        # Add base kernels.
        for kernel in kernels:
            for dim in range(ndim):
                models.append(kernel[0](ndim=ndim, axes=dim, **kernel[1]))

        # Add all compositions of the base kernels up to the max depth.
        for _ in range(1, self.max_depth):
            previous_level_models = models[:]
            for model in previous_level_models:
                for operator in operators:
                    for kernel in kernels:
                        for dim in range(ndim):
                            models.append(
                                operator(
                                    kernel[0](
                                        ndim=ndim,
                                        axes=dim,
                                        **kernel[1]
                                    ), model))

        return models

    def update(self):
        for model in self._models:
            model.update()

    def posteriors(self):
        """Compute posterior probabilities of the models."""
        # Compute log model evidence.
        log_evidences = np.zeros(len(self._models))
        for i, model in enumerate(self._models):
            model.update()
            log_evidences[i] = model.log_evidence()

        # Compute model posteriors.
        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        model_posterior = model_posterior / np.sum(model_posterior)
        return model_posterior

    def marginal_entropy(self, points, model_posterior):

        # Compute predictions and means for the test points
        means = np.zeros((len(self._models), len(points)))
        stds = np.ones((len(self._models), len(points)))
        for i, model in enumerate(self._models):
            model.update()
            (mean, var) = model.predict(points)
            means[i, :] = mean
            stds[i, :] = np.sqrt(var)

        # Compute an upper and lower bounds for y
        max_range = 4
        upper_values = means.max(0) + max_range * stds.max(0)
        lower_values = means.min(0) - max_range * stds.max(0)

        # Compute the entropy of a mixture of Gaussians for a single y
        def entropy(y, mu, sigma, model_posterior):
            sqrt_2pi = 2.5066282746310002
            prob = np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sqrt_2pi * sigma)
            prob = np.dot(model_posterior, prob)
            eps = np.spacing(1)
            return -prob * np.log(prob + eps)

        # Numerically compute the entropy of y for each test point
        y_entropy = np.zeros(len(points))
        for i in range(len(points)):

            def func(x):
                return entropy(x, means[:, i], stds[:, i], model_posterior)

            y_entropy[i] = integrate.quad(
                func,
                lower_values[i],
                upper_values[i]
            )[0]

        return y_entropy

    def __getitem__(self, index):
        return self._models[index]

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        return str(self.kernels)

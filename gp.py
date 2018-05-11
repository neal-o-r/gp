import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

np.random.seed(123)

def f(x):

        return 2*np.sin(3*x) - np.cos(5*x) + x**2

def get_data(n_points):

        x = np.random.uniform(-2, 2, n_points)
        x.sort()
        u = 0.5
        y =  f(x) + np.random.randn(n_points) * u

        return x, y, u


def plot_data_and_draws(x, y, u, x_t, G, y_b):

        plt.errorbar(x, y, yerr=u, fmt='.k')
        x_l = np.linspace(x[0], x[-1], 75)
        ax = plt.axes()
        ax.plot(x_t, G, color='#4682b4', lw=1, alpha=0.4)
        plt.xlim((x.min(), x.max()))

        plt.show()


def kernel(x, x_d, theta=[1,1]):

        sig = theta[0]
        l   = theta[1]

        sqdist = np.sum(x**2,1).reshape(-1,1) + np.sum(x_d**2,1) - 2*np.dot(x, x_d.T)
        return sig**2 * np.exp(-0.5 * (1/l**2) * sqdist)


def gpr(x, y, u, x_test, params):

        n_draws = 10

        x = x.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

        K    = kernel(x, x, params) + np.eye(len(x))*u**2
        K_s  = kernel(x, x_test, params)
        K_ss = kernel(x_test, x_test, params) + u**2

        L = np.linalg.cholesky(K)
        Lk = np.linalg.solve(L, K_s)
        mu = np.dot(Lk.T, np.linalg.solve(L, y.reshape(-1, 1)))
        L = np.linalg.cholesky(K_ss + 1e-6*np.eye(len(x_test)) - np.dot(Lk.T, Lk))
        G = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(len(x_test), n_draws)))
        y_bar = np.dot(np.dot(K_s.T, np.linalg.inv(K)), y.reshape(-1, 1))

        plot_data_and_draws(x, y, u, x_test, G, y_bar)


def logl(theta, x, y, u, x_test):

        x = x.reshape(-1, 1)
        K = kernel(x, x, theta) + np.eye(len(x))*u**2
        y = y.reshape(-1, 1)

        L =    ( - 0.5 * np.dot(np.dot(y.T, np.linalg.inv(K)), y)
                 - 0.5 * np.log(np.linalg.det(K + 1e-6 * np.eye(len(x))))
                 - 0.5 * len(x) * np.log(2 * np.pi))[0][0]

        return L if np.isfinite(L) else 1e25


def opt_params(L, x, y, u, x_test, params):

        ll = lambda *args: -L(*args)

        result = opt.minimize(ll, params, args=(x, y, u, x_test))
        return result['x']


if __name__ == '__main__':

        # Given 10 points, with some uncertainty
        n_train = 25
        x, y, u = get_data(n_train)

        # we want to estimate the function at each of these points
        n_test = 100
        x_test = np.linspace(x[0], x[-1], n_test)

        gpr(x, y, u, x_test, [1,1])

        theta = opt_params(logl, x, y, u, x_test, [1,1])

        gpr(x, y, u, x_test, theta)


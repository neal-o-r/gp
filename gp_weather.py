import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from gp import opt_params

plt.style.use('ggplot')
np.random.seed(123)


def get_data():

	df = pd.read_csv('carlow_weather.csv', parse_dates=['date'])
	df['maxtp'] = df.maxtp.apply(lambda x: 0 if x ==' ' else float(x))
	df.index = df.date
	
	df_g = df.resample('1W').max()

	d = np.array(df_g.index)
	y = np.array(df_g.maxtp)
	u = 0.5 * np.random.randn(len(d))
	x = np.arange(len(d))
	y[~np.isfinite(y.ravel())] = 20.
	y[np.where(y == 0)] += 21
	

	return x, d, y, u


def periodic(theta, x, x_d):
	
	l, p = theta

	dists = cdist(x, x_d, metric='euclidean')
	K = np.exp(- 2 * (np.sin(np.pi / p * dists) / l) ** 2)	

	return K


def kernel(theta, x, x_d):

	kp = periodic(theta, x, x_d)

	return kp


def prior(theta):
	
	l, p = theta

	if (-1000 < l < 1000) and (25 < p < 100):
		return 0 

	return -np.inf  


def log_post(theta, x, y, u, x_test):

	x = x.reshape(-1, 1)
	y = y.reshape(-1, 1)

	L = logl(theta, x, y, u)

	P = prior(theta)

	Lpost = L + P
	return Lpost if np.isfinite(Lpost) else 1e25


def logl(theta, x, y, u):

	x = x.reshape(-1, 1)
	y = y.reshape(-1, 1)
	
	K = kernel(theta, x, x) + np.eye(len(x))*u**2

	L =    ( - 0.5 * np.dot(np.dot(y.T, np.linalg.inv(K)), y)
                 - 0.5 * np.log(np.linalg.det(K + 1e-6 * np.eye(len(x)))) 
                 - 0.5 * len(x) * np.log(2 * np.pi))[0][0]

	return L if np.isfinite(L) else 1e25


def gpr(x, y, u, x_test, theta):

	n_draws = 10

	x = x.reshape(-1, 1)
	x_test = x_test.reshape(-1, 1)

	K    = kernel(theta, x, x) + np.eye(len(x))*u**2
	K_s  = kernel(theta, x, x_test)
	K_ss = kernel(theta, x_test, x_test) + u**2

	L = np.linalg.cholesky(K)
	Lk = np.linalg.solve(L, K_s)
	mu = np.dot(Lk.T, np.linalg.solve(L, y.reshape(-1, 1)))
	L = np.linalg.cholesky(K_ss + 1e-6*np.eye(len(x_test)) - np.dot(Lk.T, Lk))
	G = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(len(x_test), n_draws)))
	y_bar = np.dot(np.dot(K_s.T, np.linalg.inv(K)), y.reshape(-1, 1))

	return y_bar, G


def plot_data_and_draws(d, y, u, dt, G, y_b, t=0):


	plt.errorbar(d, y, yerr=u, fmt='.k')
	plt.plot(dt, G, color='#4682b4', lw=1, alpha=0.4)
	plt.plot(dt, y_b, label='Mean', color='r')
	plt.title('Periodicity: {0:.1f} weeks'.format(t))
	plt.legend()

	plt.show()



if __name__ == '__main__':

	x, d, y, u = get_data()
	sig_u = 0.5

	n_train = 400
	n_preds = 30

	x  = x[:n_train]
	y  = y[:n_train]
	dt = d[:n_train]
	u  = u[:n_train]

	x_t = np.linspace(x[0], x[-1] + n_preds, n_train + n_preds)

	init = [10, 50]
	print("Initial Parameters: \n", init)
	print("Initial marginalized ln-likelihood: {0}".format(
		log_post(init, x, y, sig_u, x_t)))


	theta = opt_params(log_post, x, y, sig_u, x_t, init)
	yb, GP = gpr(x, y, sig_u, x_t, theta)

	print("Final parameters: \n", theta)
	print("Final marginalized ln-likelihood: {0}".format(
		log_post(theta, x, y, sig_u, x_t)))

	plot_data_and_draws(dt, y, u, 
		d[:(n_train+n_preds)], GP, yb, t=theta[1])


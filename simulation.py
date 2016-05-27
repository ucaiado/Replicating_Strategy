#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Model different stochastic processes, options pricing and execute simulations

@author: ucaiado

Created on 05/27/2016
"""
# bibliotecas necessarias
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import time


'''
Begin help functions
'''


def plot_simulations(df, s_title, f_steps):
    '''
    Plot a line chart to each path in the dataframe passed
    and a vertical histogram summarizing all paths
    :param df: dataframe. the data with all paths simulated
    :param s_title: string. The title of the chart
    :param f_steps: float. the number of steps in the simulation
    '''
    # crio o grid
    sns.set_context(rc={'figure.figsize': (15, 6)})
    g = sns.JointGrid(x=df.columns, y=df.iloc[-1, :].values, size=7)
    g.ax_marg_x.set_visible(False)
    # ploto as linhas
    g.ax_joint.plot(df, linewidth=2)
    g.ax_joint.set_xlim([0, int(f_steps)])
    # ploto o histograma
    g.plot_marginals(sns.distplot, kde=True)
    # arrumo o grafico
    g.fig.suptitle(s_title, fontsize=18, y=0.92)
    g.set_axis_labels('Passo', u'Preço')


def plot_payoff(MyOption, d_param):
    '''
    PLot the payoff of the option at the expiration
    :param MyOption: Derivative Object. A contract/option instance
    :param d_param: dict. The initial parameter pof the MyOption
    '''
    # calcula Preco no vencimeno
    option = MyOption(**d_param)
    f_min = d_param['f_St']*0.2
    f_max = d_param['f_St']*1.8
    na_ST = np.linspace(f_min, f_max, 50)
    l_price = []
    for f_p in na_ST:
        # pega peco no vencimento
        option.update(f_p, 0.)
        l_price.append(option.f_price)
    # arruma dados para plotar
    df_plot = pd.DataFrame(l_price, index=na_ST)
    df_plot.columns = ['Preco']
    # plota Precos
    ax = df_plot.plot(legend=False)
    ax.set_xlabel(u'Preço do Subjacente')
    ax.set_ylabel(u'PnL')
    s_title = u'Pay-Off no Vencimento {} {}\n'
    s_prep = 'do'
    if 'Opcao' in option.s_name or 'Call' in option.s_name:
        s_prep = 'da'
    ax.set_title(s_title.format(s_prep, option.s_name), fontsize=18)


def plot_payoff_all_simulations(MyOption, df):
    '''
    Plot a payoff scatter plot with the final returns of a stochastic
    simulation of a replicationg strategy using different rebalancing
    frequencies. Compare to the last price of the option/contract simulated
    :param MyOption: Derivative Object. A contract/option instance
    :param df: dataframe. The output of strategy the simulation
    '''
    # exlui os valores do ativo que estiverem nos 2 porcento superior da distr
    f_98perc = np.percentile(df.St, 98)
    df2 = df[(df.St <= f_98perc)]
    f_max = df2.St.max()
    option = MyOption(1., 1., 1., 1., 1.)
    # plota grafico
    g = sns.FacetGrid(df2, col="N", col_wrap=3, margin_titles=True,
                      sharex=False, size=3, aspect=1.3)
    g.map(plt.scatter, 'St', 'Ht', s=2, linewidth=.0, edgecolor='white',
          alpha=0.6, label=u'$H(T)$')
    g.map(plt.scatter, 'St', 'Vt', s=3, linewidth=.0, edgecolor='white',
          color='g', label=u'$V(T)$')
    # define titutlos
    s_title = u'Pay-Off da Simulacao {} {}\n'
    s_prep = 'do'
    if 'Opcao' in option.s_name or 'Call' in option.s_name:
        s_prep = 'da'
    g.fig.suptitle(s_title.format(s_prep, option.s_name), fontsize=16, y=1.03)
    g.set_ylabels(u'Preço')
    g.despine(left=True).add_legend()
    g.set(xlim=(5., f_max + 5))


def box_plot_err(MyOption, df):
    '''
    Plot a boxplot of the hedging erro of the strategy passed
    '''
    # plota dados
    df2 = df.copy()
    df2 = df2.assign(Err=df2['Vt'] - df2['Ht'])
    option = MyOption(1., 1., 1., 1., 1.)
    ax = sns.boxplot(x='N', y='Err', data=df2)
    ax.set_ylabel('$V_T - H_T$')
    ax.set_xlabel(u'Número de Rebalanceamentos')
    # define titutlos
    s_title = u'Box-Plot do Erro de Hedging na Replicação\n {} {}'
    s_prep = 'do'
    if 'Opcao' in option.s_name or 'Call' in option.s_name:
        s_prep = 'da'
    # insere titulo
    ax.set_title(s_title.format(s_prep, option.s_name), fontsize=16, y=1.03)


def plot_payoff_simulation(MyOption, df):
    '''
    Plot a payoff scatter plot with the final returns of a stochastic
    simulation of a replicationg strategy. Compare to the last price of the
    option/contract simulated
    :param MyOption: Derivative Object. A contract/option instance
    :param df: dataframe. The output of strategy the simulation
    '''
    # exlui os valores do ativo que estiverem nos 2 porcento superior da distr
    f_98perc = np.percentile(df.St, 98)
    df2 = df[(df.St <= f_98perc)]
    f_max = df2.St.max()
    option = MyOption(1., 1., 1., 1., 1.)
    # plota grafico
    g = sns.FacetGrid(df2, palette='Set1', size=4, aspect=1.8)
    g.map(plt.scatter, 'St', 'Ht', s=7, linewidth=.0, edgecolor='white',
          alpha=0.6, label=u'$H(T)$')
    g.map(plt.scatter, 'St', 'Vt', s=9, linewidth=.0, edgecolor='white',
          color='g', label=u'$V(T)$')
    g.ax.set_xlabel(u'Preço do Subjacente')
    g.ax.set_ylabel('PnL')
    # define titutlos
    s_title = u'Pay-Off da Simulacao {} {}\n'
    s_prep = 'do'
    if 'Opcao' in option.s_name or 'Call' in option.s_name:
        s_prep = 'da'
    g.ax.set_title(s_title.format(s_prep, option.s_name), fontsize=16)
    # arruma limites e legenda
    g.add_legend()
    g.ax.set_xlim([0., f_max + 5])


def get_d1_and_d2(f_St, f_sigma, f_time, f_r, f_K):
    '''
    Calculate the d1 and d2 parameter used in Digital and call options
    '''
    f_d2 = (np.log(f_St/f_K) - (f_r - 0.5 * f_sigma ** 2)*f_time)
    f_d2 /= (f_sigma * f_time**0.5)
    f_d1 = f_d2 + f_sigma*f_time**0.5
    return f_d1, f_d2

'''
End help functions
'''
# MODELAGEM PROCESSO ESTOCASTICOS


class StochasticProcess(object):
    '''
    A general representation of a Stochastic Process
    '''
    def __init__(self, f_sigma, f_time, f_steps, b_random_state=True):
        '''
        Initialize a StochasticProcess object. Save all parameters as
        attributes
        '''
        self._f_sigma = 0
        # guarda parametros
        self.f_sigma = f_sigma
        self.f_time = f_time
        self.f_steps = f_steps
        self.b_random_state = b_random_state
        # checa se seta um seed para o random state
        if not b_random_state:
            np.random.seed(0)

    @property
    def f_sigma(self):
        '''
        Return the variance of the process
        '''
        return self._f_sigma

    @f_sigma.setter
    def f_sigma(self, f_value):
        '''
        set the f_sigma attribute. ensure that the number is positive
        :param f_value: float. A non negative float number
        '''
        # TODO: I need to check this property
        assert f_value >= 0., u"A volatilidade deve ser positiva"
        self._f_sigma = f_value

    def _random_number(self):
        '''
        Return a float drown from a distribution
        and standard deviation 1
        '''
        raise NotImplementedError()

    def __call__(self):
        '''
        Return a random drow from the Stochastic Process
        '''
        raise NotImplementedError()


class WienerProcess(StochasticProcess):
    '''
    A representation of a Wiener process, also called Brownian motion.
    '''
    def __init__(self, f_sigma, f_time, f_steps, b_random_state=True):
        '''
        Initialize a WienerProcess object. Save all parameters as attributes
        :param f_sigma: float. The standard deviation of ALL distribution
        :param f_time: float. the time at each step, in years
        :param f_steps: float. number of steps to simulate for
        :*param f_random_state: float. Random state to be used to
            reproduciability
        '''
        super(WienerProcess, self).__init__(f_sigma, f_time, f_steps,
                                            b_random_state=True)

    def _random_number(self):
        '''
        Return a float of drown from a normal distribution with mean 0 and
        standard deviation 1
        '''
        return np.random.standard_normal()

    def __call__(self):
        '''
        Return a random drow from the Wiener process with mean 0 and variance t
        '''
        delta_t = self.f_time/self.f_steps
        sqrt_delta_sigma = math.sqrt(delta_t) * self.f_sigma
        return sqrt_delta_sigma * self._random_number()


class GeometricBrownianMotion(StochasticProcess):
    '''
    A generic representation of a Geometric Brownian Motion (GBM). Besides the
    variance, this process might present a drift
    '''
    def __init__(self, f_s0, f_sigma, f_time, f_steps, b_random_state=True,
                 f_r=None, f_mu=None):
        '''
        Initialize a GeometricBrownianMotion object. Save all parameters as
        attributes
        :param f_s0: float. The initial price of the process
        :param f_sigma: float. The standard deviation of ALL distribution
        :param f_time: float. the time at each step, in years
        :param f_steps: float. number of steps to simulate for
        :*param f_random_state: float. Random state to be used to
            reproduciability
        :*param f_r: float. risk free interest rate
        :*param f_mu: float. the average return of the process
        '''
        # inicia parametros
        self._original = f_s0
        self.f_St = f_s0
        self.f_mu = f_mu
        self.f_r = f_r
        self._current_step = -1
        super(GeometricBrownianMotion, self).__init__(f_sigma, f_time, f_steps,
                                                      b_random_state=True)
        # inicia o processo de winner presente no BGM
        self.wiener_process = None

    @property
    def current_step(self):
        '''
        Return the variance of the process
        '''
        return self._current_step

    @current_step.setter
    def current_step(self, i_value):
        '''
        set the current_step attribute. ensure that the current step is
        not greater than the total steps defined to the process
        :param i_value: integer. A increment step to the current path
        '''
        self._current_step = i_value
        if self._current_step > self.f_steps:
            self._current_step = 0
            self.f_St = self._original

    def __call__(self):
        '''
        Return a PRICE selected from a specif path randomly generated by a
        Geometric Brownian Motion
        '''
        raise NotImplementedError()


class GBM_Exact_Solution(GeometricBrownianMotion):
    '''
    A representation of the Exact solution of a Geometric Brownian Motion (GBM)
    Besides de variance, this process might present a drift
    '''
    def __init__(self, f_s0, f_mu, f_sigma, f_time, f_steps,
                 b_random_state=True):
        '''
        Initialize a GBM_Exact_Solution object. Save all parameters as
        attributes
        :param f_s0: float. The initial price of the process
        :param f_mu: float. the average return of the process
        :param f_sigma: float. The standard deviation of ALL distribution
        :param f_time: float. the time at each step, in years
        :param f_steps: float. number of steps to simulate for
        :*param f_random_state: float. Random state to be used to
            reproduciability
        '''
        # inicia variaveis da GeometricBrownianMotion
        super(GBM_Exact_Solution, self).__init__(f_s0=f_s0,
                                                 f_mu=f_mu,
                                                 f_sigma=f_sigma,
                                                 f_time=f_time,
                                                 f_steps=f_steps,
                                                 b_random_state=True)
        # inicia o processo de winner presente no BGM
        self.wiener_process = WienerProcess(f_sigma, f_time, f_steps)

    def __call__(self):
        '''
        Return a PRICE selected from a specif path randomly generated by the
        Exact solution of the Geometric Brownian Motion
        '''
        # seta a quantidade de passos ja dados por esse processo
        # se for maior que limite, retorna para estado inicial e
        # comeca novamente
        self.current_step += 1
        s_entrei = "Nao"
        if self.current_step > 0:
            s_entrei = "Sim"
            # calcula preco pelo movimento browniano geometrico
            dt = self.f_time/self.f_steps
            sigma_pow_mu_delta = (self.f_mu - 0.5 * self.f_sigma**2.0) * dt
            f_log_rtn = sigma_pow_mu_delta + self.wiener_process()
            f_rtn = np.exp(f_log_rtn)
            self.f_St *= f_rtn
            # print self.current_step, self.f_St, s_entrei
        return self.f_St


class GBM_By_Euler(GeometricBrownianMotion):
    '''
    A representation of the Geometric Brownian Motion (GBM) simulated using
    Euler Method. Besides de variance, this process might present a drift
    '''
    def __init__(self, f_s0, f_mu, f_sigma, f_time, f_steps,
                 b_random_state=True):
        '''
        Initialize a GBM_Exact_Solution object. Save all parameters as
        attributes
        :param f_s0: float. The initial price of the process
        :param f_mu: float. the average return of the process
        :param f_sigma: float. The standard deviation of ALL distribution
        :param f_time: float. the time at each step, in years
        :param f_steps: float. number of steps to simulate for
        :*param f_random_state: float. Random state to be used to
            reproduciability
        '''
        # inicia variaveis da GeometricBrownianMotion
        super(GBM_By_Euler, self).__init__(f_s0=f_s0,
                                           f_mu=f_mu,
                                           f_sigma=f_sigma,
                                           f_time=f_time,
                                           f_steps=f_steps)
        # inicia o processo de winner, que vou dar o nome dos termos da equacao
        self.sigma_sqrt_t_eps = WienerProcess(f_sigma, f_time, f_steps)

    def __call__(self):
        '''
        Return a PRICE selected from a specif path randomly generated by a
        Geometric Brownian Motion
        '''
        # seta a quantidade de passos ja dados por esse processo
        # se for maior que limite, retorna para estado inicial e
        # comeca novamente
        self.current_step += 1
        if self.current_step > 0:
            # calcula preco pelo movimento browniano geometrico
            dt = self.f_time/self.f_steps
            mu_St_dt = self.f_mu * self.f_St * dt
            sigma_St_sqrt_t_eps = self.f_St * self.sigma_sqrt_t_eps()
            self.f_St += mu_St_dt + sigma_St_sqrt_t_eps
            # print self.current_step, self.f_St, s_entrei
        return self.f_St


class GBM_Risk_Neutral_By_Euler(GeometricBrownianMotion):
    '''
    A representation of the Geometric Brownian Motion (GBM) simulated using
    Euler Method. As it is risk neutral, the asset grows by the risk free
    interest rate
    '''
    def __init__(self, f_s0, f_r, f_sigma, f_time, f_steps,
                 b_random_state=True):
        '''
        Initialize a GBM_Exact_Solution object. Save all parameters as
        attributes
        :param f_s0: float. The initial price of the process
        :param f_r: float. the risk free interest rate
        :param f_sigma: float. The standard deviation of ALL distribution
        :param f_time: float. the time at each step, in years
        :param f_steps: float. number of steps to simulate for
        :*param f_random_state: float. Random state to be used to
            reproduciability
        '''
        # inicia variaveis da GeometricBrownianMotion
        super(GBM_Risk_Neutral_By_Euler, self).__init__(f_s0=f_s0,
                                                        f_r=f_r,
                                                        f_sigma=f_sigma,
                                                        f_time=f_time,
                                                        f_steps=f_steps)
        # inicia o processo de winner, que vou dar o nome dos termos da equacao
        self.sigma_sqrt_t_eps = WienerProcess(f_sigma, f_time, f_steps)

    def __call__(self):
        '''
        Return a PRICE selected from a specif path randomly generated by a
        Geometric Brownian Motion
        '''
        # seta a quantidade de passos ja dados por esse processo
        # se for maior que limite, retorna para estado inicial e
        # comeca novamente
        self.current_step += 1
        if self.current_step > 0:
            # calcula preco pelo movimento browniano geometrico
            dt = self.f_time/self.f_steps
            r_St_dt = self.f_r * self.f_St * dt
            sigma_St_sqrt_t_eps = self.f_St * self.sigma_sqrt_t_eps()
            self.f_St += r_St_dt + sigma_St_sqrt_t_eps
            # print self.current_step, self.f_St, s_entrei
        return self.f_St


# MODELAGEM DE OPCOES
# a classe basica para as opcoes

class Derivative(object):
    '''
    A general representation of a Derivative contract. The volatility and the
    interest rate are constant
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K=None):
        '''
        Initialize a Derivative object
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :*param f_K: float. The strike, if applyable
        '''
        # inicia variaveis
        self.s_name = 'General'
        self.f_price = 0
        self.f_delta = 0
        self.f_K = f_K
        self.f_r = f_r
        self.f_sigma = f_sigma
        # define preco e delta
        self.update(f_St, f_time)

    def update(self, f_St, f_time):
        '''
        Update the price of the Derivative contract and its Delta
        :param f_St: float. The price of the underline asset
        :param f_time: float. The time remain until the expiration
        '''
        # salva novos atributos
        self.f_St = f_St
        self.f_time = f_time
        # calcula preco e delta
        self._set_price()
        self._set_delta()

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        raise NotImplementedError()

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        raise NotImplementedError()

    def __str__(self):
        '''
        Return a string describing the option
        '''
        s = u'Um(a) {} baseado em um subjacente com preco {:.2f}'
        s += u', {:.1f}% de volatilidade, juros de {:.1f}%'
        if self.f_K:
            s += u', com Strike de {}'
            l_val = [self.s_name, self.f_St, self.f_sigma * 100,
                     self.f_r*100, self.f_K, self.f_time,
                     self.f_price, self.f_delta]
        else:
            l_val = [self.s_name, self.f_St, self.f_sigma * 100,
                     self.f_r*100, self.f_time, self.f_price,
                     self.f_delta]
        s += u' e vencimento em {:.2f} anos tem o preco de R$ {:.2f} '
        s += u'e Delta de {:.2f}'
        s = s.format(*l_val)
        return s

# implementacao dos cinco contratos


class LogContract(Derivative):
    '''
    A representation of a Log Contract
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K=None):
        '''
        Initialize a LogContract object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :*param f_K: float. The strike, if applyable
        '''
        # inicia variaveis da Derivativo
        super(LogContract, self).__init__(f_St=f_St,
                                          f_sigma=f_sigma,
                                          f_time=f_time,
                                          f_r=f_r,
                                          f_K=None)
        self.s_name = 'Contrato Log'

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        exp_r_t = np.exp(-1*self.f_r*self.f_time)
        ln_S = np.log(self.f_St)
        r_var_t = (self.f_r - (self.f_sigma**2)/2) * self.f_time
        ln_S_r_var_t = ln_S + r_var_t
        self.f_price = exp_r_t * ln_S_r_var_t

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        exp_r_t = np.exp(-1*self.f_r*self.f_time)
        self.f_delta = exp_r_t / self.f_St


class SquaredLogContract(Derivative):
    '''
    A representation of a Squared Log Contract
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K=None):
        '''
        Initialize a SquaredLogContract object. Save all parameters as
        attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :*param f_K: float. The strike, if applyable
        '''
        # inicia variaveis da Derivativo
        super(SquaredLogContract, self).__init__(f_St=f_St,
                                                 f_sigma=f_sigma,
                                                 f_time=f_time,
                                                 f_r=f_r,
                                                 f_K=None)
        self.s_name = 'Contrato Log Quadratico'

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        exp_r_t = np.exp(-1*self.f_r*self.f_time)
        ln_S_r_var_t_sq = (np.log(self.f_St) + (self.f_r -
                           (self.f_sigma**2)/2) * self.f_time)**2
        var_t = self.f_sigma**2 * self.f_time
        self.f_price = exp_r_t * (ln_S_r_var_t_sq + var_t)

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        two_exp_r_t_over_S = 2 * np.exp(-1*self.f_r*self.f_time) / self.f_St
        ln_S_r_var_t = (np.log(self.f_St) + (self.f_r -
                        (self.f_sigma**2)/2) * self.f_time)
        self.f_delta = two_exp_r_t_over_S * ln_S_r_var_t


class SquaredExotic(Derivative):
    '''
    A representation of a exotic suqared contract. The Strike is given
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K):
        '''
        Initialize a SquaredExotic object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param f_K: float. The strike, if applyable
        '''
        # inicia variaveis da Derivativo
        super(SquaredExotic, self).__init__(f_St=f_St,
                                            f_sigma=f_sigma,
                                            f_time=f_time,
                                            f_r=f_r,
                                            f_K=f_K)
        self.s_name = 'Exotico Quadratico'

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        exp_r_var_t = np.exp((self.f_r + self.f_sigma**2)*self.f_time)
        S_sq_exp_r_var_t = self.f_St**2 * exp_r_var_t
        K_sq_exp_r_t = self.f_K**2 * np.exp(-self.f_r * self.f_time)
        two_S_K = 2 * self.f_St * self.f_K
        self.f_price = S_sq_exp_r_var_t - two_S_K + K_sq_exp_r_t

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        exp_r_var_t = np.exp((self.f_r + self.f_sigma**2)*self.f_time)
        two_K = 2 * self.f_K

        self.f_delta = 2*self.f_St * exp_r_var_t - two_K


class DigitalOption(Derivative):
    '''
    A representation of a Digital Option.
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K):
        '''
        Initialize a DigitalOption object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param f_K: float. The strike, if applyable
        '''
        # inicia variaveis da Derivativo
        super(DigitalOption, self).__init__(f_St=f_St,
                                            f_sigma=f_sigma,
                                            f_time=f_time,
                                            f_r=f_r,
                                            f_K=f_K)
        self.s_name = 'Opcao Digital'

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        f_d1, f_d2 = get_d1_and_d2(self.f_St, self.f_sigma, self.f_time,
                                   self.f_r, self.f_K)
        exp_r_t = np.exp(-self.f_r * self.f_time)
        cdf_d2 = stats.norm.cdf(f_d2, 0., 1.)
        self.f_price = exp_r_t * cdf_d2

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        f_d1, f_d2 = get_d1_and_d2(self.f_St, self.f_sigma, self.f_time,
                                   self.f_r, self.f_K)
        exp_r_t = np.exp(-self.f_r * self.f_time)
        pdf_d2 = stats.norm.pdf(f_d2, 0., 1.)
        sig_S_sqtr_t = self.f_sigma * self.f_St * (self.f_time**0.5)
        self.f_delta = exp_r_t * pdf_d2 / sig_S_sqtr_t


class EuropianCall(Derivative):
    '''
    A representation of a Europian Call Option
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, f_K):
        '''
        Initialize a EuropianCall object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param f_K: float. The strike, if applyable
        '''
        # inicia variaveis da Derivativo
        super(EuropianCall, self).__init__(f_St=f_St,
                                           f_sigma=f_sigma,
                                           f_time=f_time,
                                           f_r=f_r,
                                           f_K=f_K)
        self.s_name = 'Call Europeia'

    def _set_price(self):
        '''
        Return the price of the contract
        '''
        f_d1, f_d2 = get_d1_and_d2(self.f_St, self.f_sigma, self.f_time,
                                   self.f_r, self.f_K)
        exp_r_t = np.exp(-self.f_r * self.f_time)
        S_cdf_d1 = self.f_St * stats.norm.cdf(f_d1, 0., 1.)
        K_cdf_d2 = self.f_K * stats.norm.cdf(f_d2, 0., 1.)
        self.f_price = S_cdf_d1 - K_cdf_d2 * exp_r_t

    def _set_delta(self):
        '''
        Return the delta of the contract
        '''
        f_d1, f_d2 = get_d1_and_d2(self.f_St, self.f_sigma, self.f_time,
                                   self.f_r, self.f_K)
        cdf_d1 = stats.norm.cdf(f_d1, 0., 1.)
        self.f_delta = cdf_d1


# SIMULACOES


def do_simulations(MySrochasticProcess, i_nsiml, f_s0, f_mu, f_sigma, f_time,
                   f_steps, f_r=None):
    '''
    Simulate a number of paths using some Stochastic Process and return a
    dataframe with all paths
    :param MySrochasticProcess: StochasticProcess object. The stochastic
        process used
    :param i_nsiml: integer. the number of paths contructed
    :param f_s0: float. The initial price of the process
    :param f_mu: float. the average return of the process
    :param f_sigma: float. The standard deviation of ALL distribution
    :param f_time: float. the time at each step, in years
    :param f_steps: float. number of steps to simulate for
    '''
    st = time.time()
    # Crio objeto para simulacao
    if f_r:
        gbm = MySrochasticProcess(f_s0=f_s0,
                                  f_r=f_r,
                                  f_sigma=f_sigma,
                                  f_time=f_time,
                                  f_steps=f_steps)
    else:
        gbm = MySrochasticProcess(f_s0=f_s0,
                                  f_mu=f_mu,
                                  f_sigma=f_sigma,
                                  f_time=f_time,
                                  f_steps=f_steps)
    # crio i_nsiml caminhos com f_steps simulacoes
    l = []
    l_aux = []
    for idx in range(int(i_nsiml * (1 + gbm.f_steps))):
        if gbm.current_step == gbm.f_steps:
            l.append(l_aux)
            l_aux = []
        l_aux.append(gbm())
    l.append(l_aux)
    # imprimo tempo e simulacoes
    print ('\nNumero de sorteios: {}'.format(idx + 1))
    print ('Levou {:0.2f} segundos'.format(time.time() - st))
    # crio dataframe com os passos nas linhas e as simulacoes nas colunas
    df = pd.DataFrame(l).T
    return df


def replicate_portfolio(i_nsiml, MyOption, d_param):
    '''
    Simulate portfolio when selling a given option/contract and using a
    money market account and the the underline asset to replicate it
    Return a dataframe with the last observed value of the derivative,
    de asset and the strategy
    :param MyOption: Derivative Object. A contract/option instance
    :param i_nsiml: integer. the number of paths contructed
    :param d_param: dictionary. The parameter of the simulation
    '''
    st = time.time()
    # Crio objeto para simulacao
    gbm = GBM_By_Euler(f_s0=d_param['f_St'],
                       f_mu=d_param['f_mu'],
                       f_sigma=d_param['f_sigma'],
                       f_time=d_param['f_time'],
                       f_steps=d_param['f_steps'])
    # cria dicionario para opcoes
    d_par_opt = {'f_St': d_param['f_St'],
                 'f_r': d_param['f_r'],
                 'f_sigma': d_param['f_sigma'],
                 'f_time': d_param['f_time'],
                 'f_K': d_param['f_K']
                 }
    # crio opcao
    option = MyOption(**d_par_opt)
    f_dt = d_param['f_time'] / d_param['f_steps']
    # crio i_nsiml caminhos com f_steps simulacoes e calculo
    # os deltas e phis em cada passo
    d = {'St': [], 'Vt': [], 'Delta': [],
         'PhiB': [], 'DeltaS': [], 'Ht': []}
    d_aux = {'St': [], 'Vt': [], 'Delta': [],
             'PhiB': [], 'DeltaS': [], 'Ht': []}
    for idx in range(int(i_nsiml * (1 + gbm.f_steps))):
        if gbm.current_step == gbm.f_steps:
            # terminou este path. Comeca outro
            for s_key in ['St', 'Vt', 'Delta', 'PhiB', 'DeltaS', 'Ht']:
                # d[s_key].append(d_aux[s_key])
                d[s_key].append(d_aux[s_key][-1])
                d_aux[s_key] = []
        d_aux['St'].append(gbm())
        # calculo tempo para vencimento
        f_tnow = d_param['f_time'] - gbm.current_step*f_dt
        # calculo preco da opcao
        option.update(d_aux['St'][-1], f_tnow)
        d_aux['Vt'].append(option.f_price)
        # trata valor de delta quando ativo vencer
        if f_tnow < 10e-6:
            option.f_delta = d_aux['Delta'][-1]
        # guardo valor do delta
        d_aux['Delta'].append(option.f_delta)
        # Calculo posicao no ativo para portfolio replicante
        d_aux['DeltaS'].append(d_aux['St'][-1] * d_aux['Delta'][-1])
        # calculo valor no money market account
        if gbm.current_step == 0:
            d_aux['PhiB'].append(d_aux['Vt'][-1] - d_aux['DeltaS'][-1])
        else:
            # carrego juros da aplicacao anterior
            f_aux = d_aux['PhiB'][-1] * (1 + f_dt * d_param['f_r'])
            # pago delta hedge
            f_chg_delta = d_aux['Delta'][-1] - d_aux['Delta'][-2]
            f_chg_delta_s = f_chg_delta * d_aux['St'][-1]
            # guardo valor no money account
            d_aux['PhiB'].append(f_aux - f_chg_delta_s)
        # calculo valor da estrategia
        d_aux['Ht'].append(d_aux['DeltaS'][-1] + d_aux['PhiB'][-1])

    # guardo ultimo path
    for s_key in ['St', 'Vt', 'Delta', 'PhiB', 'DeltaS', 'Ht']:
        # d[s_key].append(d_aux[s_key])
        d[s_key].append(d_aux[s_key][-1])
        d_aux[s_key] = []

    # imprimo tempo e simulacoes e monto tabela
    print ('\nNumero de sorteios: {}'.format(idx + 1))
    print ('Levou {:0.2f} segundos'.format(time.time() - st))
    # crio dataframe com os passos nas linhas e as simulacoes nas colunas
    df = pd.DataFrame(d)
    df = df.loc[:, ['St', 'Ht', 'Vt']]
    return df

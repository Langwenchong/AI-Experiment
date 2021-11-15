import numpy as np
from hmm_meta import HMM_Meta


class HMM(HMM_Meta):
    """Implementation of the HMM filtering and smoothing algorithms for indoor localization.

    Parameters
    ----------
    width : int
        The width of the building

    length : int
        The length of the building

    rssi_range : int
        The range of signal strength (RSSI) values

    n_beacons : int
        The number of beacons

    sigma : float, optional
        The average step size

    init_pos : a pair of ints, optional
        The initial position (x,y) of the mobile sensor


    Attributes
    ----------
    n_states : int
        The number of states

    rssi_range : int
        The range of signal strength (RSSI) values. The RSSI can take values from 0 to rssi_range-1

    n_beacons : int
        The number of beacons

    init_probs : array, shape(n_states)
        The initial probabilities, P(S_0=i); e.g. self.init_probs[s] stores the probability P(S_0=s)

    trans_probs : array, shape(n_states, n_states)
        The transition matrix (probabilities), P(S_{t+1}=i|S_{t}=j);
        e.g. self.trans_probs[s1,s2] stores the probability P(S_{t+1}=s1|S_{t}=s2)

    obs_probs : array, shape(n,beacons, rssi_range, n_states)
        The array of observation probabilities for each beacon, P(O^{b}_{t}=i|S_{t}=j);
        e.g. self.obs_probs[b,o_b,s] stores the probability P(O^{b}_{t}=o_b|S_{t}=s)


    Examples
    ----------
    Initialization:
    >>> kwargs = {
    ...     'width'     : 32,
    ...     'length'    : 32,
    ...     'rssi_range': 4,
    ...     'n_beacons' : 32,
    ...     'init_pos'  : (0,0)
    ... }
    >>> model = HMM(**kwargs)

    """

    def __init__(self, **kwargs):
        super(HMM, self).__init__(**kwargs)

    def predict(self, probs):
        """Predicts the next state. It computes P(S_{t+1}|O_{1:t}) using probs, which is P(S_{t}|O_{1:t});
        and the transition probabilities P(S_{t+1}|S_{t}).
        Note that none of the probabilities needs to be normalized.

        Parameters
        ----------
        probs : array, shape(n_states)
            The probability vector of the states, which represents P(S_{t}|O_{1:t})

        Returns
        -------
        array, shape(n_states)
            The predicted probabilities, P(S_{t+1}|O_{1:t})

        """
        predicted_probs = np.empty(self.n_states)

        # TODO: add your code here
        # Multiply probs with the transition matrix (matrix multiplication) and store the result in predicted_probs
        # You can do that with one line of code using NumPy

        # 首先我们要知道预测公式是P(S_t+1|O1:t)=Σ_t[P(S_t=s_t|O1:t)*P(S_t+1|S_t=s_t)*P(O_t=o_t|S_t)]

        # 根据题干现在已经给出在O1:t基础上了S_t取不同的状态值k的概率向量了即P(S_t=k|O1:t)(k取任意状态值)
        # todo提示告诉我们现在要做的就是当前状态矩阵*状态转移矩阵刚好就是预测公式=右边的前两项
        # 那么我们只需要用P(S_t+1|S_t)的矩阵去和给出的probs向量相乘即得到了在O1:t基础上S_t+1取不同状态值k的概率了如下
        predicted_probs = np.matmul(self.trans_probs, probs)
        return predicted_probs

    def update(self, probs, o):
        """Updates the probabilities using the observations. It computes probs*P(O_{t}|S_{t}).
        If it is called from the method monitor, then probs represents P(S_{t}|O_{1:t-1})
        and the resulting vector will be proportional to P(S_{t}|O_{1:t}).
        Similarly, if it is called from the method backwards, probs represents P(O_{t+1:T}|S_{t})
        and it will return a vector that is proportional to P(O_{t:T}|S_{t}).

        Parameters
        ----------
        probs : array, shape(n_states)
            The probability vector of the states

        o : array, shape(n_beacons)
            The observation vector, the RSSI values received from each beacon

        Returns
        -------
        array, shape(n_states)
            The updated probabilities, P(S_{t}|O_{1:t}) or P(O_{t:T}|S_{t}) depending on the context

        """
        updated_probs = np.empty(self.n_states)

        # TODO: add your code here
        # Multiply (element-wise) probs with the observation probabilities P(O^{b}_{t}=o[b]|S_{t}) for each beacon b
        # and store the result in updated_probs.
        # 根据提示，这里我们要更新probs向量
        # 实际上这里就是再求预测公式的前两项乘积后的中间结果和最后一项的乘积
        # 首先我们知道O[b]是我们在状态t时从不同位置获取到的观测值向量，他存储了此时当前状态不同位置的观测值
        # 然后我们要根据提示完成更新，做法就是用当前的probs向量去乘S_t即当前状态基础上得到的不同的观察值O^b_t=o[b]的概率向量
        # 同时t时刻得到不同的观测值b概率可以根据题干self.obs probs[b,o[b],s]获得，因此代码如下
        #注意，初始时要先把所有的概率设置为1,
        for i in range(self.n_states):
            updated_probs[i] = 1;
        for i in range(self.n_states):
            for j in range(self.n_beacons):
                # 先求b不同时的概率之积
                updated_probs[i] *= self.obs_probs[j, o[j], i]
            # 然后在和probs[i]当前状态概率相乘
            updated_probs[i] *= probs[i]
        return updated_probs

    def monitor(self, T, observations):
        """Returns the monitoring probabilities for T time steps using the given sequence of observations.
        In other words, it computes P(S_{t}|O_{1:t}) for each t.
        This procedure is also called filtering, and the algorithm is known as forward algorithm.

        Parameters
        ----------
        T : int
            The number of time steps

        observations: array, shape(T, n_beacons)
            The sequence of observations received

        Returns
        -------
        array, shape(T, n_states)
            The monitoring probabilities, P(S_{t}|O_{1:t}) for each t

        """
        monitoring_probs = np.empty((T, self.n_states))

        # TODO: add your code here
        # Store the initial probabilities in monitoring_probs[0]
        # For t=1,2,...,T-1;
        #     Copy monitoring_probs[t-1] to monitoring_probs[t]
        #     Predict monitoring_probs[t]
        #     Update monitoring_probs[t]
        #     Normalize monitoring_probs[t]

        # 现在我们已经完成了预测公式P(S_t+1|O1:t)=Σ_t[P(S_t=s_t|O1:t)*P(S_t+1|S_t=s_t)*P(O_t=o_t|S_t)]核心的乘积运算步骤
        # 那么接下来我们需要将两个核心的运算步骤组合起来完成最终的预测公式
        # 首先我们初始化0状态
        monitoring_probs[0] = self.init_probs

        # 然后我们使用之前完成的predict函数方法进行第一步前两项的运算
        # 然后再使用这个结果去调用update完成与P(O_t|S_t)的运算，一定要注意传递的O[b]观测向量要保证切片正确
        # 然后我们再进行归一化处理记得到了预测的概率结果
        for i in range(1, T):
            # 前两项的乘积运算得到的中间值
            monitoring_probs[i] = self.predict(monitoring_probs[i-1])
            # 然后再和后一项进行相乘得到结果
            monitoring_probs[i] = self.update(
                monitoring_probs[i], observations[i])
            # 再归一化处理得到答案
            monitoring_probs[i] /= np.sum(monitoring_probs[i])

        return monitoring_probs

    def postdict(self, probs):
        """Predicts the previous state. It computes P(O_{t:T}|S_{t-1}) using probs, which is P(O_{t:T}|S_{t});
        and the transition probabilities P(S_{t}|S_{t-1}).
        Note that none of the probabilities needs to be normalized.

        Parameters
        ----------
        probs : array, shape(n_states)
            The probability vector of the states, which represents P(O_{t:T}|S_{t})

        Returns
        -------
        array, shape(n_states)
            The postdicted probabilities, P(O_{t:T}|S_{t-1})

        """
        postdicted_probs = np.empty(self.n_states)
        # Matrix multiplication
        postdicted_probs[:] = self.trans_probs.T @ probs
        return postdicted_probs

    def backwards(self, T, observations):
        """Returns the backwards probabilities for T time steps using the sequence of observations.
        In other words, it computes P(O_{t+1:T}|S_{t}) for each t.

        Parameters
        ----------
        T : int
            The number of time steps

        observations: array, shape(T, n_beacons)
            The sequence of observations received

        Returns
        -------
        array, shape(T, n_states)
            The backwards probabilities, P(O_{t+1:T}|S_{t}) for each t

        """
        backwards_probs = np.empty((T, self.n_states))
        backwards_probs[T - 1] = np.ones(self.n_states)
        for t in range(T - 2, -1, -1):
            backwards_probs[t] = backwards_probs[t + 1]
            backwards_probs[t] = self.update(
                backwards_probs[t], observations[t + 1])
            backwards_probs[t] = self.postdict(backwards_probs[t])
            backwards_probs[t] /= np.sum(backwards_probs[t])  # Normalization
        return backwards_probs

    def hindsight(self, T, observations):
        """Computes the hindsight probabilities by combining the monitoring and backwards probabilities.
        It returns P(S_{t}|O_{1:T}) for each t.

        Parameters
        ----------
        T : int
            The number of time steps

        observations: array, shape(T, n_beacons)
            The sequence of observations received

        Returns
        -------
        array, shape(T, n_states)
            The hindsight probabilities, P(S_{t}|O_{1:T}) for each t

        """
        hindsight_probs = np.empty((T, self.n_states))
        hindsight_probs[:, :] = self.monitor(
            T, observations) * self.backwards(T, observations)
        hindsight_probs[:, :] = (
            hindsight_probs.T / np.sum(hindsight_probs, axis=1)).T  # Normalization
        return hindsight_probs

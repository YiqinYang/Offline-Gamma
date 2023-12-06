from math import gamma
from utils import *


my_gamma = 0.95
def bcq_operator(Q, mdp, mu, mu_hat, zeta, noise, *args):
    """
    BCQ Bellman Operator: Q(s, a) = r(s, a) + γ * max_{a' | μ > 0} Q(s', a')
    """
    Q = (Q + (np.random.rand(mdp.N_S, mdp.N_A) - 0.5) * 2 * noise).clip(0, 10 / (1 - my_gamma))
    Q_update = mdp.r + my_gamma * np.dot(mdp.P, np.max((mu_hat > zeta) * Q, axis=-1))
    return (mu > 0) * Q_update + (mu == 0) * Q


c = 0.5
res_bcq_6 = []
zeta = 0
for _ in range(100):
    mdp = RandomMdp()
    idxs = np.random.permutation(mdp.N_S * mdp.N_A)
    Q_rand = np.ones((mdp.N_S, mdp.N_A)) / (1 - GAMMA)
    Q_true = get_fixed_point(optimal_operator, Q_rand, mdp)
    for part in (0.56, ): # 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,
        for noise in (0.1,):
            mask = np.ones((mdp.N_S * mdp.N_A)) > 0
            mask[idxs[int(mdp.N_S * mdp.N_A * c):]] = False
            mask = mask.reshape(mdp.N_S, mdp.N_A)
            mask[Q_true == np.max(Q_true, axis=1, keepdims=True)] = True

            mu = softmax(Q_true, 1)
            mu = mu * mask
            mu /= mu.sum(-1, keepdims=True)

            mask = np.ones((mdp.N_S * mdp.N_A)) > 0
            mask[idxs[int(mdp.N_S * mdp.N_A * part):]] = False
            mask = mask.reshape(mdp.N_S, mdp.N_A)
            mask[Q_true == np.max(Q_true, axis=1, keepdims=True)] = True
            mu_hat = softmax(Q_true, 1)
            mu_hat = mu_hat * mask
            mu_hat /= mu_hat.sum(-1, keepdims=True)
            Q_rand = np.ones((mdp.N_S, mdp.N_A)) / (1 - GAMMA)
            Q_bcq = get_fixed_point(bcq_operator, Q_rand, mdp, mu, mu_hat, zeta, noise)
            error = np.abs(Q_bcq[mu > 0] - Q_true[mu > 0]).max()
            res_bcq_6.append(dict(part=part, noise=noise, error=error))
res_bcq_6 = pd.DataFrame(res_bcq_6)

for part in (0.56,): # 0.55, 0.6, 0.7, 0.8, 0.9
    data = res_bcq_6[(res_bcq_6['noise']==noise) & (res_bcq_6['part']==part)]['error']
    print(f"{(part - 0.5) / 0.5 * 100:.0f}\\% & ${data.mean():.2f} \pm {data.std():.2f}$ \\\\", my_gamma)
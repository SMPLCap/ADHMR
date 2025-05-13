import torch
import math
import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
    return a


def generalized_steps(x_0, x_1, seq, model, b, eta, ctx=None, gen_multi=False):
    '''
        x_0: joint gaussian noise X_T
        x_1: twist gaussian noise X_T
    '''
    with torch.no_grad():
        n = x_0.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = [[],[]]
        xs = [[x_0],[x_1]]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x_0.device)
            next_t = (torch.ones(n) * j).to(x_0.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt_0 = xs[0][-1].to(x_0.device)
            xt_1 = xs[1][-1].to(x_0.device)
            
            et = model(xinj=xt_0,xint=xt_1,t=t.float(),ctx =ctx, gen_multi= gen_multi)          # estimated noise for current timestep
            x0_t_0 = (xt_0 - et[0] * (1 - at).sqrt()) / at.sqrt()
            x0_t_1 = (xt_1 - et[1] * (1 - at).sqrt()) / at.sqrt()
            x0_preds[0].append(x0_t_0)                                                          # estimated x_0 of current timestep
            x0_preds[1].append(x0_t_1)
            c1 = (                                                      # signma_t
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next_0 = at_next.sqrt() * x0_t_0 + c1 * torch.randn_like(x_0) + c2 * et[0]
            xt_next_1 = at_next.sqrt() * x0_t_1 + c1 * torch.randn_like(x_1) + c2 * et[1]
            xs[0].append(xt_next_0)
            xs[1].append(xt_next_1)
    return xs, x0_preds

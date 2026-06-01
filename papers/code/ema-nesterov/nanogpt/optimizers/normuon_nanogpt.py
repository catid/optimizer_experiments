import torch
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95, nesterov=True, backend_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2, nesterov=nesterov, backend_steps=backend_steps)
        # assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        # params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        zeropower_backend = zeropower_via_newtonschulz5
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    had_grad = p.grad is not None
                    if not had_grad:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])

                    state["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
                    update = p.grad.lerp_(state["momentum_buffer"], group["momentum"]) if group['nesterov'] else state["momentum_buffer"]
                    if update.size(0) == 3 * update.size(1): # split grouped QKV parameters
                        update = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in update.split(update.size(1))])
                        scale = update.size(1)**0.5
                    else:
                        update = zeropower_backend(update, steps=group['backend_steps'])
                        scale = max(update.size(0), update.size(1))**0.5 # scale to have update.square().mean() == 1
                    update = update.to(p.grad.dtype)
                    
                    ################ NorMuon added ###################
                    vnorm = update.norm(dim=(-2,-1), keepdim=True)
                    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
                    state["second_momentum_buffer"].lerp_(v_mean, 1 - group["beta2"])
                    step_size = 1 / state["second_momentum_buffer"].sqrt().add_(1e-10)
                    update.mul_(step_size)
                    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
                    update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
                    ##################################################


                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.data.add_(update, alpha=-group["lr"] * scale)
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss
    
    
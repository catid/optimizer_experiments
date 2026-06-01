import torch
import torch.distributed as dist
from torch.optim import Optimizer

import os
import random

def set_seed_deterministic(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # for reproducibility
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)


class Lookahead(Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_step_size, local_steps_K=100):

        defaults = dict()
        super(Lookahead, self).__init__(params, defaults)
        self.inner_optimizer = inner_optimizer
        self.lookahead_step_size = lookahead_step_size
        self.local_steps_K = local_steps_K
        self.it = 0
        self.initialize_buffers()

    def __setstate__(self, state):
        super(Lookahead, self).__setstate__(state)
    
    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'current_params' not in param_state:
                    param_state['current_params'] = (p.clone(), 0)

    @torch.no_grad()
    def lookahead_step(self):
        prin = True
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                if prin and dist.get_rank() == 0:
                    print("Lookahead: {}_iter consuming ({}_iter - {}_iter) as lookahead".format(param_state['current_params'][1], param_state['current_params'][1], self.it))
                    prin = False
            
                # accumulated inner step udpates
                lookahead = p.add(param_state['current_params'][0], alpha=-1)
                # apply lookahead
                p.copy_(param_state['current_params'][0]).add_(lookahead, alpha=self.lookahead_step_size)
                # shift current_params
                param_state['current_params'] = (param_state['current_params'][0].copy_(p), self.it)
   

    @torch.no_grad()
    def step(self):
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()
        if self.it % self.local_steps_K == 0 and self.it > 0:
            self.lookahead_step()
        self.it += 1


class GPA(Optimizer):
    def __init__(self, params, inner_optimizer, momentum_x, momentum_y): 
        super(GPA, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.momentum_x = momentum_x
        self.momentum_y = momentum_y
        self.xy_exchanged = False
        self.initialize_buffers()
    
    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'x_params_buffer' not in param_state:
                    param_state['x_params_buffer'] = p.clone()
                
                if 'y_params_buffer' not in param_state:
                    param_state['y_params_buffer'] = p.clone()

                if 'z_params' not in param_state:
                    param_state['z_params'] = p.clone()
    
    @torch.no_grad()
    def nesterov_step(self):
        if not self.xy_exchanged:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # calculate y
                    param_state["y_params_buffer"].copy_(self.momentum_y * p).add_(param_state["z_params"], alpha=1-self.momentum_y)
                    # exchange p and y so that p contains the lookahead position
                    param_state["x_params_buffer"].copy_(p)
                    p.copy_(param_state["y_params_buffer"])

            self.xy_exchanged = True
    
    @torch.no_grad()
    def step(self):
        if not self.xy_exchanged:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass.")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()

        for group in self.param_groups:
            weight_decay = group.get("weight_decay", 0)
            assert weight_decay == 0, "check implementation for weight_decay > 0."
            for p in group['params']:
                param_state = self.state[p]
                # update z
                # param_state['z_params'].mul_(1 - group['lr'] * weight_decay).add_(p - param_state["y_params_buffer"])
                param_state['z_params'].add_(p - param_state["y_params_buffer"])
                # update x
                p.copy_(self.momentum_x * param_state["x_params_buffer"]).add_(param_state['z_params'], alpha=1-self.momentum_x)

        self.xy_exchanged = False


class EMA_Nesterov(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_stepsize=0, use_scheduled_lookahead_stepsize=True, lookahead_ema=0.9, warmup_step=0, rest_step=0):
        if lookahead_stepsize < 0.0:
            raise ValueError("Invalid momentum value: {}".format(lookahead_stepsize))

        super(EMA_Nesterov, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.warmup_step = warmup_step
        self.rest_step = rest_step
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0
        self.initialize_buffers()

    def __setstate__(self, state):
        super(EMA_Nesterov, self).__setstate__(state)

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'prev_params' not in param_state:
                    param_state['prev_params'] = (p.clone(), self.it)

                if 'lookahead_buffer' not in param_state:
                    param_state['lookahead_buffer'] = (torch.zeros_like(p), -1)
    
    def get_lr_lambda(self):
        if isinstance(self.inner_optimizer, list):
            lr_lambda = self.inner_optimizer[0].param_groups[0]["lr"] / self.inner_optimizer[0].param_groups[0]["initial_lr"]
        else:
            lr_lambda = self.inner_optimizer.param_groups[0]["lr"] / self.inner_optimizer.param_groups[0]["initial_lr"]
        return lr_lambda

    @torch.no_grad()
    def lookahead_step(self):
        """Performs nesterov's lookahead."""
        if self.use_scheduled_lookahead_stepsize:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda()
        else:
            lookahead_stepsize = self.lookahead_stepsize
        self.current_lookahead_stepsize = lookahead_stepsize

        for group in self.param_groups:
            for p in group['params']:

                param_state = self.state[p]         

                lookahead = param_state['lookahead_buffer'][0]
       
                p.add_(lookahead, alpha=lookahead_stepsize)



    @torch.no_grad()
    def accum_lookahead(self):
        """Update nesterov's lookahead direction."""
        lookahead_ema = self.lookahead_ema
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                look = p.add(param_state['prev_params'][0], alpha=-1)

                # update lookahead buffer
                buf = param_state['lookahead_buffer'][0]

                param_state['lookahead_buffer'] = (buf.lerp_(look, 1 - lookahead_ema), self.it) # m^{t+1} = beta * m^t + (1-beta) * look

                # update prev_params buffer
                param_state['prev_params'] = (param_state['prev_params'][0].copy_(p), self.it)

    
    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.warmup_step and self.it < self.rest_step and not self.lookahead_status:
            self.lookahead_step()
        else:
            self.current_lookahead_stepsize = 0
        self.lookahead_status = True

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass.")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()
        
        self.accum_lookahead()

        self.lookahead_status = False
        self.it += 1
    
    @torch._disable_dynamo
    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                    }
                ]
            }

        """

        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)

        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }

        state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }

        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        if isinstance(self.inner_optimizer, list):
            state_dict["inner_state_dict"] = [opt.state_dict() for opt in self.inner_optimizer]
        else:
            state_dict["inner_state_dict"] = self.inner_optimizer.state_dict()
        return state_dict


class SNOO(Optimizer):
    def __init__(self, params, inner_optimizer, lr, momentum, local_steps_K=100):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict()
        super(SNOO, self).__init__(params, defaults)
        self.inner_optimizer = inner_optimizer
        self.lr = lr
        self.momentum = momentum
        self.local_steps_K = local_steps_K
        self.it = 0
        self.initialize_buffers()

    def __setstate__(self, state):
        super(SNOO, self).__setstate__(state)
    
    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'current_params' not in param_state:
                    param_state['current_params'] = (p.clone(), 0)
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def nesterov_buffer_step(self):
        prin = True
        for group in self.param_groups:
            momentum = self.momentum

            for p in group['params']:
                param_state = self.state[p]

                if prin and dist.get_rank() == 0:
                    print("SNOO: {}_iter consuming ({}_iter - {}_iter) as lookahead momentum".format(self.it, param_state['current_params'][1], self.it))
                    prin = False
            
                # accumulated inner step udpates
                accum_update = param_state['current_params'][0].add(p, alpha=-1)
                # update Nesterov's momentum
                param_state['momentum_buffer'].mul_(momentum).add_(accum_update)
                # apply Nesterov update
                p.copy_(param_state['current_params'][0]).add_(momentum * param_state['momentum_buffer'], alpha=-self.lr).add_(accum_update, alpha=-self.lr)
                # shift current_params
                param_state['current_params'] = (param_state['current_params'][0].copy_(p), self.it)
   

    @torch.no_grad()
    def step(self):
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()
        if self.it % self.local_steps_K == 0 and self.it > 0:
            self.nesterov_buffer_step()
        self.it += 1

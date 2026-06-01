import torch

import os
import random
from optimizers.muon import MuonWithAuxAdam

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

def get_trainable_parameters(model, args):
    if "muon" in args.optimizer.lower():
        muon_params = []
        adam_params = []
        for name, prm in model.named_parameters():
            if prm.ndim == 2 and not "embed_tokens" in name and not "lm_head" in name:
                muon_params.append(prm)
            else:
                adam_params.append(prm)
        trainable_params = [
            dict(params=muon_params, use_muon=True, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
            dict(params=adam_params, use_muon=False, lr=args.adam_lr, betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.adam_weight_decay),
        ]
    else:
        trainable_params = []
        for name, prm in model.named_parameters():
            trainable_params.append({"params": prm, "lr": args.lr})
    return trainable_params


def build_optimizer(trainable_params, args):   
    if args.optimizer.lower() == "muon":
        optimizer = MuonWithAuxAdam(trainable_params)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer


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


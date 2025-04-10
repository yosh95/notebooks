import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):

    def __init__(self,
                 d_model=128,
                 n_layers=1,
                 d_state=4,
                 expand_factor=1,
                 d_conv=3,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 rms_norm_eps=1e-5,
                 bias=False,
                 conv_bias=True):

        super().__init__()

        d_inner = expand_factor * d_model
        dt_rank = math.ceil(d_model / 16)

        self.layers = nn.ModuleList(
                [ResidualBlock(d_model,
                               d_inner,
                               d_state,
                               d_conv,
                               dt_rank,
                               dt_min,
                               dt_max,
                               dt_scale,
                               dt_init_floor,
                               bias,
                               conv_bias,
                               rms_norm_eps)
                 for _ in range(n_layers)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 d_state,
                 d_conv,
                 dt_rank,
                 dt_min,
                 dt_max,
                 dt_scale,
                 dt_init_floor,
                 bias,
                 conv_bias,
                 rms_norm_eps):

        super().__init__()

        self.mixer = MambaBlock(d_model,
                                d_inner,
                                d_state,
                                d_conv,
                                dt_rank,
                                dt_min,
                                dt_max,
                                dt_scale,
                                dt_init_floor,
                                bias,
                                conv_bias)
        self.norm = RMSNorm(rms_norm_eps)

    def forward(self, x):

        output = self.mixer(self.norm(x)) + x

        return output

    def step(self, x, cache):

        output, cache = self.mixer.step(self.norm(x), cache)

        output = output + x

        return output, cache


class MambaBlock(nn.Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 d_state,
                 d_conv,
                 dt_rank,
                 dt_min,
                 dt_max,
                 dt_scale,
                 dt_init_floor,
                 bias,
                 conv_bias):

        super().__init__()

        self.d_state = d_state
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(
                d_model,
                2 * d_inner,
                bias=bias)

        self.conv1d = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=d_inner,
                padding=d_conv - 1)

        self.x_proj = nn.Linear(
                d_inner,
                dt_rank + 2 * d_state,
                bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
                torch.rand(d_inner) *
                (math.log(dt_max) - math.log(dt_min)) +
                math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(
                1,
                d_state + 1,
                dtype=torch.float32).repeat(d_inner, 1)

        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
                d_inner, d_model, bias=bias)

    def forward(self, x):

        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x, z)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        return output

    def ssm(self, x, z):

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(
                deltaBC,
                [self.dt_rank,
                 self.d_state,
                 self.d_state],
                dim=-1)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)

        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        hs = PScan.apply(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y


class RMSNorm(nn.Module):

    def __init__(self,
                 eps=1e-5):

        super().__init__()

        self.eps = eps

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return output


class PScan(torch.autograd.Function):

    @staticmethod
    def npo2(len):

        return 2 ** math.ceil(math.log2(len))

    @staticmethod
    def pad_npo2(X):

        len_npo2 = PScan.npo2(X.size(1))
        pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
        return F.pad(X, pad_tuple, "constant", 0)

    @staticmethod
    def pscan(A, X):

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                    Aa[:, :, 3].mul(
                        Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                    Aa[:, :, 0].mul(Xa[:, :, 1].add(
                        Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):

        L = X_in.size(1)

        if L == PScan.npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = PScan.pad_npo2(A_in)
            X = PScan.pad_npo2(X_in)

        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == PScan.npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = PScan.pad_npo2(grad_output_in)
            A_in = PScan.pad_npo2(A_in)

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))

        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]

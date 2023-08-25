"""Example of a one layer QG model"""
import numpy as np
import time
import torch

from helmholtz import solve_helmholtz_dst, solve_helmholtz_dst_cmm
from qgm import QGFV

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64


# grid
nx = 1024
ny = 1024
nl = 1
L = 100000 # 100km
dx = L / nx
dy = L / ny
xv = torch.linspace(-L/2, L/2, nx+1, dtype=torch.float64, device=device)
yv = torch.linspace(-L/2, L/2, ny+1, dtype=torch.float64, device=device)
x, y = torch.meshgrid(xv, yv, indexing='ij')

H = torch.zeros(nl,1,1, dtype=dtype, device=device)
if nl == 1:
    H[0,0,0] = 1000.

# gravity
g_prime = torch.zeros(nl,1,1, dtype=dtype, device=device)
if nl == 1:
    g_prime[0,0,0] = 10

## create rankine vortex
# Burger and Rossby numbers, coriolis set with Bu Number
Bu = 1
Ro = 0.01
r0, r1, r2 = 0.1*L, 0.1*L, 0.14*L
f0 = torch.sqrt(g_prime[0,0,0] * H[0,0,0] / Bu / r0**2)
beta = 0
f = f0 + beta * (y - L/2)

# wind forcing, bottom drag
tau0 = 0.
bottom_drag_coef = 0.

apply_mask = True

# create rankine vortex in PV
xc = 0.5 * (xv[1:] + xv[:-1])
yc = 0.5 * (yv[1:] + yv[:-1])
x, y = torch.meshgrid(xc, yc, indexing='ij')
r = torch.sqrt(x**2 + y**2)
# circular domain mask
mask = (r < L/2).type(torch.float64) if apply_mask else torch.ones_like(x)

z = x + 1j*y
theta = torch.angle(z)
# slight pertubation of initial vortex to create tripole
epsilon = 1e-3
r *= (1+epsilon*torch.cos(theta*3+torch.pi/4))
soft_step = lambda x: torch.sigmoid(x/100)
mask_core = soft_step(r0 - r)
mask_ring = soft_step(r-r1) * soft_step(r2-r)
pv = 1. * (-mask_core / mask_core.mean() + mask_ring / mask_ring.mean()) * mask

param = {
    'nx': nx,
    'ny': ny,
    'nl': nl,
    'mask': mask,
    'n_ens': 1,
    'Lx': L,
    'Ly': L,
    'flux_stencil': 5,
    'H': H,
    'g_prime': g_prime,
    'tau0': tau0,
    'f0': f0,
    'beta': beta,
    'bottom_drag_coef': bottom_drag_coef,
    'device': device,
    'dt': 0, # time-step (s)
}


qg = QGFV(param)
qg.q = pv.unsqueeze(0).unsqueeze(0)
# compute p from q_over_f0
q_i = qg.interp_TP(qg.q)
helmholtz_rhs = torch.einsum('lm,...mxy->...lxy', qg.Cl2m, q_i)
if apply_mask:
    psi_modes = solve_helmholtz_dst_cmm(
                helmholtz_rhs*qg.masks.psi[...,1:-1,1:-1],
                qg.helmholtz_dst, qg.cap_matrices,
                qg.masks.psi_irrbound_xids,
                qg.masks.psi_irrbound_yids,
                qg.masks.psi)
else:
    psi_modes = solve_helmholtz_dst(helmholtz_rhs, qg.helmholtz_dst)
qg.psi = torch.einsum('lm,...mxy->...lxy', qg.Cm2l, psi_modes)

# set amplitude to have correct Rossby number
u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
u_norm_max = max(torch.abs(u).max().item(), torch.abs(v).max().item())
factor = Ro * f0 * r0 / u_norm_max
qg.psi *= factor
qg.q *= factor

u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
u_max = u.max().cpu().item()
v_max = v.max().cpu().item()
print(f'u_max {u_max:.2e}, v_max {v_max:.2e}')

# set time step with CFL
cfl = 0.5
dt = cfl * min(dx / u_max, dy / v_max)
qg.dt = dt


# time params
t = 0
w_0 = qg.laplacian_h(qg.psi, qg.dx, qg.dy).squeeze()
tau = 1. / torch.sqrt(w_0.pow(2).mean()).cpu().item()
print(f'tau = {tau *f0:.2f} f0-1')
t_end = 30. * tau
n_steps = int(t_end / dt) + 1

# plot, log, check nans
freq_plot = int(t_end / 25 / dt) + 1
freq_checknan = 10
freq_log = int(t_end / 50 / dt) + 1

ns_store = [int(t / dt)+1 for t in np.linspace(0, t_end, 100)]
qs = []

if freq_plot > 0:
    ims = []
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})

    palette = plt.cm.bwr.with_extremes(bad='grey')
    plt.ion()
    f,a = plt.subplots(1, 2, figsize=(16,9))
    a[0].set_title('q (units of $f_0$)')
    a[1].set_title('$\\psi$ $(m^2 s^{-1})$')
    a[0].set_xticks([]), a[0].set_yticks([])
    a[1].set_xticks([]), a[1].set_yticks([])
    plt.tight_layout()
    plt.pause(0.1)

    # intermediate plots
    ts_plot = [0*tau, 5*tau, 7.5*tau, 9*tau, 13*tau, 18*tau, 24*tau, 30*tau]
    ns_plot = [int(t / dt)+1 for t in ts_plot]
    f2,a2 = plt.subplots(2, 4, figsize=(16,9))
    f2.suptitle('Evolution of the potential vorticity q (units of $f_0$)')

wM, pM = None, None

t0 = time.time()
for n in range(1, n_steps+1):
    if n in ns_store:
        qs.append((qg.q / qg.f0 * qg.masks.q).cpu())

    if n in ns_plot:
        ind = ns_plot.index(n)
        q_over_f0 = (qg.q / qg.f0 * qg.masks.q)[0,0].cpu().numpy()
        q_over_f0 = np.ma.masked_where((1-qg.masks.q[0,0].cpu().numpy()), q_over_f0)
        if wM is None:
            wM = np.abs(q_over_f0).max()
        im = a2[ind//4, ind % 4].imshow(q_over_f0.T, cmap=palette, origin='lower', vmin=-wM, vmax=wM, animated=True)
        a2[ind//4, ind % 4].set_title(f't={t/tau:.1f}$\\tau$')
        a2[ind//4, ind % 4].set_xticks([]), a2[ind//4, ind % 4].set_yticks([])
        plt.pause(0.05)

    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        q_over_f0 = (qg.q / qg.f0 * qg.masks.q)[0,0].cpu().numpy()
        q_over_f0 = np.ma.masked_where((1-qg.masks.q[0,0].cpu().numpy()), q_over_f0)
        psi = qg.psi.cpu().numpy()
        if wM is None or pM is None:
            wM = np.abs(q_over_f0).max()
            pM = np.abs(psi).max()
        im0 = a[0].imshow(q_over_f0.T, cmap=palette, origin='lower', vmin=-wM, vmax=wM, animated=True)
        im1 = a[1].imshow(psi[0,0].T, cmap='bwr', origin='lower', vmin=-pM, vmax=pM, animated=True)
        if n // freq_plot == 1:
            f.colorbar(im0, ax=a[0])
            f.colorbar(im1, ax=a[1])
        f.suptitle(f'Ro={Ro:.2f}, Bu={Bu:.2f}, t={t/tau:.2f}$\\tau$')
        plt.pause(0.05)

    qg.step()
    t += dt

    if n % freq_checknan == 0 and torch.isnan(qg.psi).any():
        raise ValueError(f'Stopping, NAN number in psi at iteration {n}.')


    if freq_log > 0 and n % freq_log == 0:
        u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
        u, v = u.cpu().numpy(), v.cpu().numpy()
        log_str = f'{n=:06d}, qg t={t/tau:.2f} tau, ' \
                    f'u: {u.mean():+.1E}, {np.abs(u).max():.1E}, ' \
                    f'v: {v.mean():+.1E}, {np.abs(v).max():.2E}'
        print(log_str)

f2.tight_layout()
f2.subplots_adjust(right=0.92)
cbar_ax = f2.add_axes([0.93, 0.05, 0.02, 0.9])
f2.colorbar(im, cax=cbar_ax)

plt.figure()
plt.plot(np.linspace(0, t_end/tau, 100), [q.pow(2).mean() for q in qs])
plt.ylabel('Enstrophy (units of $f_0^2$)')
plt.xlabel('Time (units of $\\tau$)')

total_time = time.time() - t0
print(f'{total_time // 60}min {(total_time % 60)}sec')

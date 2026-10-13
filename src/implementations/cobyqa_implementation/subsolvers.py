import numpy as np
from scipy.linalg import qr
from typing import Callable

TINY = np.finfo(float).tiny
EPS = np.finfo(float).eps

# ==========================================
# Helper Functions
# ==========================================

def get_arrays_tol(*arrays) -> float:
    """Get a relative tolerance for a set of arrays."""
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    size = max(array.size for array in arrays)
    weight = max(np.max(np.abs(array[np.isfinite(array)]), initial=1.0) for array in arrays)
    return 10.0 * EPS * max(size, 1.0) * weight

def _alpha_tr(step: np.ndarray, sd: np.ndarray, delta: float) -> float:
    """Calculates the step size to the trust-region boundary."""
    step_sd = step @ sd
    sd_sq = sd @ sd
    dist_tr_sq = delta**2.0 - step @ step
    temp = np.sqrt(max(step_sd**2.0 + sd_sq * dist_tr_sq, 0.0))
    if step_sd <= 0.0 and sd_sq > TINY * abs(temp - step_sd):
        alpha_tr = max((temp - step_sd) / sd_sq, 0.0)
    elif abs(temp + step_sd) > TINY * dist_tr_sq:
        alpha_tr = max(dist_tr_sq / (temp + step_sd), 0.0)
    else:
        raise ZeroDivisionError
    return alpha_tr

def _argmin(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(x <= np.min(x))

# ==========================================
# Geometry-Improving Steps
# ==========================================

def _cauchy_geom(const: float, grad: np.ndarray, curv: Callable[[np.ndarray], float], xl: np.ndarray, xu: np.ndarray, delta: float):
    """Calculates the Cauchy step for geometry improvement."""
    fixed_xl = (xl < 0.0) & (grad > 0.0)
    fixed_xu = (xu > 0.0) & (grad < 0.0)

    cauchy_step = np.zeros_like(grad)
    cauchy_step[fixed_xl] = xl[fixed_xl]
    cauchy_step[fixed_xu] = xu[fixed_xu]
    
    if np.linalg.norm(cauchy_step) > delta:
        working = fixed_xl | fixed_xu
        while True:
            g_norm = np.linalg.norm(grad[working])
            delta_reduced = np.sqrt(delta**2.0 - cauchy_step[~working] @ cauchy_step[~working])
            if g_norm > TINY * abs(delta_reduced):
                mu = max(delta_reduced / g_norm, 0.0)
            else:
                break
            cauchy_step[working] = mu * grad[working]

            fixed_xl = working & (cauchy_step < xl)
            fixed_xu = working & (cauchy_step > xu)
            if not np.any(fixed_xl) and not np.any(fixed_xu):
                break
            cauchy_step[fixed_xl] = xl[fixed_xl]
            cauchy_step[fixed_xu] = xu[fixed_xu]
            working = working & ~(fixed_xl | fixed_xu)

    grad_step = grad @ cauchy_step
    if grad_step >= 0.0:
        s_norm = np.linalg.norm(cauchy_step)
        alpha_tr = max(delta / s_norm, 0.0) if s_norm > TINY * delta else 0.0

        curv_step = curv(cauchy_step)
        alpha_quad = max(-grad_step / curv_step, 0.0) if curv_step < -TINY * grad_step else np.inf

        i_xl = (xl > -np.inf) & (cauchy_step < TINY * xl)
        i_xu = (xu < np.inf) & (cauchy_step > TINY * xu)
        alpha_xl = np.min(xl[i_xl] / cauchy_step[i_xl], initial=np.inf)
        alpha_xu = np.min(xu[i_xu] / cauchy_step[i_xu], initial=np.inf)
        alpha_bd = min(alpha_xl, alpha_xu)

        alpha = min(alpha_tr, alpha_quad, alpha_bd)
        step = np.clip(alpha * cauchy_step, xl, xu)
        q_val = const + alpha * grad_step + 0.5 * alpha**2.0 * curv_step
    else:
        step = np.zeros_like(grad)
        q_val = const

    return step, q_val

def cauchy_geometry(const: float, grad: np.ndarray, curv: Callable[[np.ndarray], float], xl: np.ndarray, xu: np.ndarray, delta: float) -> np.ndarray:
    """Maximizes approximately the absolute value of a quadratic function subject to bound constraints."""
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    step1, q_val1 = _cauchy_geom(const, grad, curv, xl, xu, delta)
    step2, q_val2 = _cauchy_geom(-const, -grad, lambda x: -curv(x), xl, xu, delta)
    
    return step1 if abs(q_val1) >= abs(q_val2) else step2

def spider_geometry(const: float, grad: np.ndarray, curv: Callable[[np.ndarray], float], xpt: np.ndarray, xl: np.ndarray, xu: np.ndarray, delta: float) -> np.ndarray:
    """Maximizes approximately the absolute value of a quadratic function along given straight lines."""
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    step = np.zeros_like(grad)
    q_val = const
    s_norm = np.linalg.norm(xpt, axis=0)

    i_xl_pos = (xl > -np.inf) & (xpt.T > -TINY * xl)
    i_xl_neg = (xl > -np.inf) & (xpt.T < TINY * xl)
    i_xu_pos = (xu < np.inf) & (xpt.T > TINY * xu)
    i_xu_neg = (xu < np.inf) & (xpt.T < -TINY * xu)

    alpha_xl_pos = np.broadcast_to(np.atleast_1d(np.max(np.atleast_2d(np.broadcast_to(xl, i_xl_pos.shape)[i_xl_pos] / xpt.T[i_xl_pos]), axis=1, initial=-np.inf)), xpt.shape[1])
    alpha_xl_neg = np.broadcast_to(np.atleast_1d(np.max(np.atleast_2d(np.broadcast_to(xl, i_xl_neg.shape)[i_xl_neg] / xpt.T[i_xl_neg]), axis=1, initial=np.inf)), xpt.shape[1])
    alpha_xu_neg = np.broadcast_to(np.atleast_1d(np.max(np.atleast_2d(np.broadcast_to(xu, i_xu_neg.shape)[i_xu_neg] / xpt.T[i_xu_neg]), axis=1, initial=-np.inf)), xpt.shape[1])
    alpha_xu_pos = np.broadcast_to(np.atleast_1d(np.max(np.atleast_2d(np.broadcast_to(xu, i_xu_pos.shape)[i_xu_pos] / xpt.T[i_xu_pos]), axis=1, initial=np.inf)), xpt.shape[1])

    for k in range(xpt.shape[1]):
        if s_norm[k] > TINY * delta:
            alpha_tr = max(delta / s_norm[k], 0.0)
        else:
            continue

        alpha_bd_pos = max(min(alpha_xu_pos[k], alpha_xl_neg[k]), 0.0)
        alpha_bd_neg = min(max(alpha_xl_pos[k], alpha_xu_neg[k]), 0.0)

        grad_step = grad @ xpt[:, k]
        curv_step = curv(xpt[:, k])
        
        if (grad_step >= 0.0 and curv_step < -TINY * grad_step) or (grad_step <= 0.0 and curv_step > -TINY * grad_step):
            alpha_quad_pos = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_pos = np.inf
            
        if (grad_step >= 0.0 and curv_step > TINY * grad_step) or (grad_step <= 0.0 and curv_step < TINY * grad_step):
            alpha_quad_neg = min(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_neg = -np.inf

        alpha_pos = min(alpha_tr, alpha_bd_pos)
        alpha_neg = max(-alpha_tr, alpha_bd_neg)
        q_val_pos = const + alpha_pos * grad_step + 0.5 * alpha_pos**2.0 * curv_step
        q_val_neg = const + alpha_neg * grad_step + 0.5 * alpha_neg**2.0 * curv_step
        
        if alpha_quad_pos < alpha_pos:
            q_val_quad_pos = const + alpha_quad_pos * grad_step + 0.5 * alpha_quad_pos**2.0 * curv_step
            if abs(q_val_quad_pos) > abs(q_val_pos):
                alpha_pos = alpha_quad_pos
                q_val_pos = q_val_quad_pos
                
        if alpha_quad_neg > alpha_neg:
            q_val_quad_neg = const + alpha_quad_neg * grad_step + 0.5 * alpha_quad_neg**2.0 * curv_step
            if abs(q_val_quad_neg) > abs(q_val_neg):
                alpha_neg = alpha_quad_neg
                q_val_neg = q_val_quad_neg
                
        if abs(q_val_pos) >= abs(q_val_neg) and abs(q_val_pos) > abs(q_val):
            step = np.clip(alpha_pos * xpt[:, k], xl, xu)
            q_val = q_val_pos
        elif abs(q_val_neg) > abs(q_val_pos) and abs(q_val_neg) > abs(q_val):
            step = np.clip(alpha_neg * xpt[:, k], xl, xu)
            q_val = q_val_neg

    return step

# ==========================================
# QR Factorizations for Active Sets
# ==========================================

def qr_tangential_byrd_omojokun(A_ineq: np.ndarray, A_eq: np.ndarray, free_xl: np.ndarray, free_xu: np.ndarray, free_ub: np.ndarray):
    n = free_xl.size
    identity = np.eye(n)
    # Replaced slow np.block with fast np.vstack
    mat = np.vstack([
        A_eq, 
        A_ineq[~free_ub, :], 
        -identity[~free_xl, :], 
        identity[~free_xu, :]
    ])
    q, r, _ = qr(mat.T, pivoting=True)
    n_act = np.count_nonzero(np.abs(np.diag(r)) >= 10.0 * EPS * n * np.linalg.norm(r[: np.min(r.shape), : np.min(r.shape)], axis=0))
    return n_act, q

def qr_normal_byrd_omojokun(A_ineq: np.ndarray, free_xl: np.ndarray, free_xu: np.ndarray, free_slack: np.ndarray, free_ub: np.ndarray):
    m_ineq, n = A_ineq.shape
    identity_n = np.eye(n)
    identity_m = np.eye(m_ineq)
    
    # Replaced slow np.block with fast hstack/vstack
    row1 = np.hstack([A_ineq[~free_ub, :], -identity_m[~free_ub, :]])
    row2 = np.hstack([np.zeros((m_ineq - np.count_nonzero(free_slack), n)), -identity_m[~free_slack, :]])
    row3 = np.hstack([-identity_n[~free_xl, :], np.zeros((n - np.count_nonzero(free_xl), m_ineq))])
    row4 = np.hstack([identity_n[~free_xu, :], np.zeros((n - np.count_nonzero(free_xu), m_ineq))])
    
    mat = np.vstack([row1, row2, row3, row4])
    q, r, _ = qr(mat.T, pivoting=True)
    
    n_act = np.count_nonzero(np.abs(np.diag(r)) >= 10.0 * EPS * (n + m_ineq) * np.linalg.norm(r[: np.min(r.shape), : np.min(r.shape)], axis=0))
    return n_act, q

# ==========================================
# Trust-Region Subproblem Solvers
# ==========================================

def tangential_byrd_omojokun_unconstrained(grad: np.ndarray, hess_prod: Callable[[np.ndarray], np.ndarray], xl: np.ndarray, xu: np.ndarray, delta: float, improve_tcg: bool = True) -> np.ndarray:
    """Minimizes approximately a quadratic function subject to bound constraints in a trust region."""
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    n = grad.size
    grad = np.copy(grad)
    grad_orig = np.copy(grad)

    free_bd = ((xl < 0.0) | (grad < 0.0)) & ((xu > 0.0) | (grad > 0.0))
    step = np.zeros_like(grad)
    sd = np.zeros_like(step)
    sd[free_bd] = -grad[free_bd]

    k = 0
    reduct = 0.0
    boundary_reached = False
    
    while k < np.count_nonzero(free_bd):
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        try:
            alpha_tr = _alpha_tr(step, sd, delta)
        except ZeroDivisionError:
            break

        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        hess_sd = hess_prod(sd)
        curv_sd = sd @ hess_sd
        alpha_quad = max(-grad_sd / curv_sd, 0.0) if curv_sd > TINY * abs(grad_sd) else np.inf

        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        i_xl = (xl > -np.inf) & (sd < -TINY * np.abs(xl - step))
        i_xu = (xu < np.inf) & (sd > TINY * np.abs(xu - step))
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_xl[i_xl] = np.maximum((xl[i_xl] - step[i_xl]) / sd[i_xl], 0.0)
        all_alpha_xu[i_xu] = np.maximum((xu[i_xu] - step[i_xu]) / sd[i_xu], 0.0)
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_bd = min(alpha_xl, alpha_xu)

        alpha = min(alpha, alpha_bd)
        if alpha > 0.0:
            step[free_bd] = np.clip(step[free_bd] + alpha * sd[free_bd], xl[free_bd], xu[free_bd])
            grad += alpha * hess_sd
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd):
            beta = (grad[free_bd] @ hess_sd[free_bd]) / curv_sd
            sd[free_bd] = beta * sd[free_bd] - grad[free_bd]
            sd[~free_bd] = 0.0
            k += 1
        elif alpha < alpha_tr:
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
            else:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
            free_bd[i_new] = False
            sd[free_bd] = -grad[free_bd]
            sd[~free_bd] = 0.0
            k = 0
        else:
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_bd[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_bd[i_new] = False
            boundary_reached = True
            break

    if improve_tcg and boundary_reached:
        step_base = np.copy(step)
        step_comparator = grad_orig @ step_base + 0.5 * step_base @ hess_prod(step_base)

        while np.count_nonzero(free_bd) > 0:
            step_sq = step[free_bd] @ step[free_bd]
            grad_sq = grad[free_bd] @ grad[free_bd]
            grad_step = grad[free_bd] @ step[free_bd]
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd[free_bd] = grad_step * step[free_bd] - step_sq * grad[free_bd]
            sd[~free_bd] = 0.0
            
            if grad_sd >= -1e-8 * reduct or np.any(grad_sd >= -TINY * np.abs(sd[free_bd])):
                break
            sd[free_bd] /= -grad_sd

            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            temp_xl[free_bd] = step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xl[free_bd] ** 2.0
            temp_xu[free_bd] = step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xu[free_bd] ** 2.0
            temp_xl[temp_xl > 0.0] = np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            temp_xu[temp_xu > 0.0] = np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(all_t_xl[i_xl], dist_xl[i_xl] / temp_xl[i_xl])
            all_t_xu[i_xu] = np.minimum(all_t_xu[i_xu], dist_xu[i_xu] / temp_xu[i_xu])
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            hess_step = hess_prod(step)
            hess_sd = hess_prod(sd)
            curv_step = step @ hess_step
            curv_sd = sd @ hess_sd
            curv_step_sd = step @ hess_sd

            n_samples = int((20 - 3) * t_bd + 3)
            t_samples = np.linspace(t_bd / n_samples, t_bd, n_samples)
            sin_values = 2.0 * t_samples / (1.0 + t_samples**2.0)
            all_reduct = sin_values * (grad_step * t_samples - grad_sd - t_samples * curv_step + sin_values * (t_samples * curv_step_sd - 0.5 * (curv_sd - curv_step)))
            
            if np.all(all_reduct <= 0.0):
                break

            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (1.0 + t_samples[i_max] ** 2.0)
            step[free_bd] = cos_value * step[free_bd] + sin_values[i_max] * sd[free_bd]
            grad += (cos_value - 1.0) * hess_step + sin_values[i_max] * hess_sd
            reduct += all_reduct[i_max]

            if t_bd < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_bd:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_bd[i_new] = False
                if t_xu <= t_bd:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_bd[i_new] = False
            else:
                break

        if grad_orig @ step + 0.5 * step @ hess_prod(step) > step_comparator:
            step = step_base

    return step

def tangential_byrd_omojokun_constrained(grad: np.ndarray, hess_prod: Callable[[np.ndarray], np.ndarray], xl: np.ndarray, xu: np.ndarray, A_ineq: np.ndarray, b_ineq: np.ndarray, A_eq: np.ndarray, delta: float, improve_tcg: bool = True) -> np.ndarray:
    """Minimizes approximately a quadratic function subject to bound and linear constraints in a trust region."""
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)
    b_ineq = np.maximum(b_ineq, 0.0)

    n = grad.size
    grad = np.copy(grad)
    grad_orig = np.copy(grad)

    free_xl = (xl < 0.0) | (grad < 0.0)
    free_xu = (xu > 0.0) | (grad > 0.0)
    free_ub = (b_ineq > 0.0) | (A_ineq @ grad > 0.0)
    n_act, q = qr_tangential_byrd_omojokun(A_ineq, A_eq, free_xl, free_xu, free_ub)

    step = np.zeros_like(grad)
    sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
    resid = np.copy(b_ineq)

    k = 0
    reduct = 0.0
    boundary_reached = False
    
    while k < n - n_act:
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        try:
            alpha_tr = _alpha_tr(step, sd, delta)
        except ZeroDivisionError:
            break

        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        hess_sd = hess_prod(sd)
        curv_sd = sd @ hess_sd
        alpha_quad = max(-grad_sd / curv_sd, 0.0) if curv_sd > TINY * abs(grad_sd) else np.inf

        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        i_xl = free_xl & (xl > -np.inf) & (sd < -TINY * np.abs(xl - step))
        i_xu = free_xu & (xu < np.inf) & (sd > TINY * np.abs(xu - step))
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_xl[i_xl] = np.maximum((xl[i_xl] - step[i_xl]) / sd[i_xl], 0.0)
        all_alpha_xu[i_xu] = np.maximum((xu[i_xu] - step[i_xu]) / sd[i_xu], 0.0)
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_bd = min(alpha_xl, alpha_xu)

        A_ineq_sd = A_ineq @ sd
        i_ub = free_ub & (A_ineq_sd > TINY * np.abs(resid))
        all_alpha_ub = np.full_like(b_ineq, np.inf)
        all_alpha_ub[i_ub] = resid[i_ub] / A_ineq_sd[i_ub]
        alpha_ub = np.min(all_alpha_ub, initial=np.inf)

        alpha = min(alpha, alpha_bd, alpha_ub)
        if alpha > 0.0:
            step = np.clip(step + alpha * sd, xl, xu)
            grad += alpha * hess_sd
            resid = np.maximum(0.0, resid - alpha * A_ineq_sd)
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd, alpha_ub):
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            beta = (grad_proj @ hess_sd) / curv_sd
            sd = beta * sd - grad_proj
            k += 1
        elif alpha < alpha_tr:
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            elif alpha_xu <= alpha:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            else:
                i_new = np.argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_tangential_byrd_omojokun(A_ineq, A_eq, free_xl, free_xu, free_ub)
            sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
            k = 0
        else:
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            if alpha_ub <= alpha:
                i_new = _argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_tangential_byrd_omojokun(A_ineq, A_eq, free_xl, free_xu, free_ub)
            boundary_reached = True
            break

    if improve_tcg and boundary_reached and n_act < n:
        step_base = np.copy(step)
        while n_act < n:
            step_proj = q[:, n_act:] @ (q[:, n_act:].T @ step)
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            step_sq = step_proj @ step_proj
            grad_sq = grad_proj @ grad_proj
            grad_step = grad_proj @ step_proj
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd = q[:, n_act:] @ (q[:, n_act:].T @ (grad_step * step - step_sq * grad))
            
            if grad_sd >= -1e-8 * reduct or np.any(grad_sd >= -TINY * np.abs(sd)):
                break
            sd /= -grad_sd

            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            temp_xl[free_xl] = sd[free_xl] ** 2.0 - dist_xl[free_xl] * (dist_xl[free_xl] - 2.0 * step_proj[free_xl])
            temp_xu[free_xu] = sd[free_xu] ** 2.0 - dist_xu[free_xu] * (dist_xu[free_xu] + 2.0 * step_proj[free_xu])
            temp_xl[temp_xl > 0.0] = np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            temp_xu[temp_xu > 0.0] = np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(all_t_xl[i_xl], dist_xl[i_xl] / temp_xl[i_xl])
            all_t_xu[i_xu] = np.minimum(all_t_xu[i_xu], dist_xu[i_xu] / temp_xu[i_xu])
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            temp_ub = np.zeros_like(resid)
            A_ineq_step = A_ineq @ step_proj
            A_ineq_sd = A_ineq @ sd
            temp_ub[free_ub] = A_ineq_sd[free_ub] ** 2.0 - resid[free_ub] * (resid[free_ub] + 2.0 * A_ineq_step[free_ub])
            temp_ub[temp_ub > 0.0] = np.sqrt(temp_ub[temp_ub > 0.0]) + A_ineq_sd[temp_ub > 0.0]
            
            i_ub = temp_ub > TINY * resid
            all_t_ub = np.ones_like(resid)
            all_t_ub[i_ub] = np.minimum(all_t_ub[i_ub], resid[i_ub] / temp_ub[i_ub])
            t_ub = np.min(all_t_ub, initial=1.0)
            t_min = min(t_bd, t_ub)

            hess_step = hess_prod(step_proj)
            hess_sd = hess_prod(sd)
            curv_step = step_proj @ hess_step
            curv_sd = sd @ hess_sd
            curv_step_sd = step_proj @ hess_sd

            n_samples = int((20 - 3) * t_min + 3)
            t_samples = np.linspace(t_min / n_samples, t_min, n_samples)
            sin_values = 2.0 * t_samples / (1.0 + t_samples**2.0)
            all_reduct = sin_values * (grad_step * t_samples - grad_sd - sin_values * (0.5 * t_samples**2.0 * curv_step - 2.0 * t_samples * curv_step_sd + 0.5 * curv_sd))
            
            if np.all(all_reduct <= 0.0):
                break

            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (1.0 + t_samples[i_max] ** 2.0)
            step = np.clip(step + (cos_value - 1.0) * step_proj + sin_values[i_max] * sd, xl, xu)
            grad += (cos_value - 1.0) * hess_step + sin_values[i_max] * hess_sd
            resid = np.maximum(0.0, resid - (cos_value - 1.0) * A_ineq_step - sin_values[i_max] * A_ineq_sd)
            reduct += all_reduct[i_max]

            if t_min < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_min:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_xl[i_new] = False
                if t_xu <= t_min:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_xl[i_new] = False
                if t_ub <= t_min:
                    i_new = _argmin(all_t_ub)
                    free_ub[i_new] = False
                n_act, q = qr_tangential_byrd_omojokun(A_ineq, A_eq, free_xl, free_xu, free_ub)
            else:
                break

        if grad_orig @ step + 0.5 * step @ hess_prod(step) > grad_orig @ step_base + 0.5 * step_base @ hess_prod(step_base):
            step = step_base

    return step

def normal_byrd_omojokun(A_ineq: np.ndarray, b_ineq: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray, xl: np.ndarray, xu: np.ndarray, delta: float, improve_tcg: bool = True) -> np.ndarray:
    """Minimizes approximately a linear constraint violation subject to bound constraints in a trust region."""
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    m_ineq, n = A_ineq.shape
    grad = np.r_[A_eq.T @ -b_eq, np.maximum(0.0, -b_ineq)]
    free_xl = (xl < 0.0) | (grad[:n] < 0.0)
    free_xu = (xu > 0.0) | (grad[:n] > 0.0)
    free_slack = b_ineq < 0.0
    free_ub = (b_ineq > 0.0) | (A_ineq @ grad[:n] - grad[n:] > 0.0)
    n_act, q = qr_normal_byrd_omojokun(A_ineq, free_xl, free_xu, free_slack, free_ub)

    delta_slack = np.sqrt(b_eq @ b_eq + grad[n:] @ grad[n:])
    step = np.zeros(n)
    sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
    resid = b_ineq + grad[n:]

    k = 0
    reduct = 0.0
    boundary_reached = False
    
    while k < n + m_ineq - n_act:
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        try:
            alpha_tr = _alpha_tr(step, sd[:n], delta)
        except ZeroDivisionError:
            alpha_tr = np.inf

        try:
            alpha_tr = min(alpha_tr, _alpha_tr(grad[n:], sd[n:], delta_slack))
        except ZeroDivisionError:
            pass

        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        hess_sd = np.r_[A_eq.T @ (A_eq @ sd[:n]), sd[n:]]
        curv_sd = sd @ hess_sd
        alpha_quad = max(-grad_sd / curv_sd, 0.0) if curv_sd > TINY * abs(grad_sd) else np.inf

        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        i_xl = free_xl & (xl > -np.inf) & (sd[:n] < -TINY * np.abs(xl - step))
        i_xu = free_xu & (xu < np.inf) & (sd[:n] > TINY * np.abs(xu - step))
        i_slack = free_slack & (sd[n:] < -TINY * np.abs(grad[n:]))
        
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_slack = np.full_like(b_ineq, np.inf)
        
        all_alpha_xl[i_xl] = np.maximum((xl[i_xl] - step[i_xl]) / sd[:n][i_xl], 0.0)
        all_alpha_xu[i_xu] = np.maximum((xu[i_xu] - step[i_xu]) / sd[:n][i_xu], 0.0)
        all_alpha_slack[i_slack] = np.maximum(-grad[n:][i_slack] / sd[n:][i_slack], 0.0)
        
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_slack = np.min(all_alpha_slack, initial=np.inf)
        alpha_bd = min(alpha_xl, alpha_xu, alpha_slack)

        A_ineq_sd = A_ineq @ sd[:n] - sd[n:]
        i_ub = free_ub & (A_ineq_sd > TINY * np.abs(resid))
        all_alpha_ub = np.full_like(b_ineq, np.inf)
        all_alpha_ub[i_ub] = resid[i_ub] / A_ineq_sd[i_ub]
        alpha_ub = np.min(all_alpha_ub, initial=np.inf)

        alpha = min(alpha, alpha_bd, alpha_ub)
        if alpha > 0.0:
            step = np.clip(step + alpha * sd[:n], xl, xu)
            grad += alpha * hess_sd
            resid = np.maximum(0.0, resid - alpha * A_ineq_sd)
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd, alpha_ub):
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            beta = (grad_proj @ hess_sd) / curv_sd
            sd = beta * sd - grad_proj
            k += 1
        elif alpha < alpha_tr:
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            elif alpha_xu <= alpha:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            elif alpha_slack <= alpha:
                i_new = np.argmin(all_alpha_slack)
                free_slack[i_new] = False
            else:
                i_new = np.argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_normal_byrd_omojokun(A_ineq, free_xl, free_xu, free_slack, free_ub)
            sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
            k = 0
        else:
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            boundary_reached = True
            break

    if improve_tcg and boundary_reached:
        step_base = np.copy(step)
        free_bd = free_xl & free_xu
        grad = A_ineq.T @ np.maximum(A_ineq @ step - b_ineq, 0.0) + A_eq.T @ (A_eq @ step - b_eq)
        sd = np.zeros(n)
        
        while np.count_nonzero(free_bd) > 0:
            step_sq = step[free_bd] @ step[free_bd]
            grad_sq = grad[free_bd] @ grad[free_bd]
            grad_step = grad[free_bd] @ step[free_bd]
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd[free_bd] = grad_step * step[free_bd] - step_sq * grad[free_bd]
            sd[~free_bd] = 0.0
            
            if grad_sd >= -1e-8 * reduct or np.any(grad_sd >= -TINY * np.abs(sd[free_bd])):
                break
            sd[free_bd] /= -grad_sd

            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            temp_xl[free_bd] = step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xl[free_bd] ** 2.0
            temp_xu[free_bd] = step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xu[free_bd] ** 2.0
            temp_xl[temp_xl > 0.0] = np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            temp_xu[temp_xu > 0.0] = np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(all_t_xl[i_xl], dist_xl[i_xl] / temp_xl[i_xl])
            all_t_xu[i_xu] = np.minimum(all_t_xu[i_xu], dist_xu[i_xu] / temp_xu[i_xu])
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            n_samples = int((20 - 3) * t_bd + 3)
            t_samples = np.linspace(t_bd / n_samples, t_bd, n_samples)
            resid_ub = np.maximum(A_ineq @ step - b_ineq, 0.0)
            resid_eq = A_eq @ step - b_eq
            step_proj = np.copy(step)
            step_proj[~free_bd] = 0.0
            all_reduct = np.empty(n_samples)
            
            for i in range(n_samples):
                sin_value = 2.0 * t_samples[i] / (1.0 + t_samples[i] ** 2.0)
                step_alt = np.clip(step + sin_value * (sd - t_samples[i] * step_proj), xl, xu)
                resid_ub_alt = np.maximum(A_ineq @ step_alt - b_ineq, 0.0)
                resid_eq_alt = A_eq @ step_alt - b_eq
                all_reduct[i] = 0.5 * (resid_ub @ resid_ub + resid_eq @ resid_eq - resid_ub_alt @ resid_ub_alt - resid_eq_alt @ resid_eq_alt)
                
            if np.all(all_reduct <= 0.0):
                break

            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (1.0 + t_samples[i_max] ** 2.0)
            sin_value = (2.0 * t_samples[i_max] / (1.0 + t_samples[i_max] ** 2.0))
            step[free_bd] = cos_value * step[free_bd] + sin_value * sd[free_bd]
            grad = A_ineq.T @ np.maximum(A_ineq @ step - b_ineq, 0.0) + A_eq.T @ (A_eq @ step - b_eq)
            reduct += all_reduct[i_max]

            if t_bd < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_bd:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_bd[i_new] = False
                if t_xu <= t_bd:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_bd[i_new] = False
            else:
                break

        resid_ub = np.maximum(A_ineq @ step - b_ineq, 0.0)
        resid_ub_base = np.maximum(A_ineq @ step_base - b_ineq, 0.0)
        resid_eq = A_eq @ step - b_eq
        resid_eq_base = A_eq @ step_base - b_eq
        if (resid_ub @ resid_ub + resid_eq @ resid_eq > resid_ub_base @ resid_ub_base + resid_eq_base @ resid_eq_base):
            step = step_base

    return step

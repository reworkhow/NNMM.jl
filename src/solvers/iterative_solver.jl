#=
================================================================================
Gibbs Sampler for Mixed Model Equations
================================================================================
Iterative Gibbs sampling for solving MME in MCMC.

Functions:
- Gibbs(A, x, b, vare): Single-trait sampler (lambda version)
- Gibbs(A, x, b): Multi-trait sampler (general version)

These samplers update the solution vector x in-place by cycling through
each element and sampling from its full conditional distribution.

Reference:
  Sorensen & Gianola (2002) Likelihood, Bayesian, and MCMC Methods 
  in Quantitative Genetics. Springer.

Author: NNMM.jl Team (adapted from JWAS.jl)
================================================================================
=#

"""
    Gibbs(A, x, b, vare::AbstractFloat)

One iteration of Gibbs sampler for single-trait MME (lambda version).

Updates solution vector `x` in-place by sampling each element from its
full conditional distribution.

# Arguments
- `A`: Left-hand side matrix of MME
- `x`: Solution vector (updated in-place)
- `b`: Right-hand side vector
- `vare`: Residual variance (scalar)

# Notes
Handles zero diagonal elements (skip update when A[i,i] = 0).
"""
function Gibbs(A, x, b, vare::AbstractFloat)
    for i = 1:size(x, 1)
        if A[i,i] != 0.0  # Skip zero diagonals (issue #70)
            invlhs = 1.0 / A[i,i]
            μ = invlhs * (b[i] - A[:,i]'x) + x[i]
            x[i] = randn() * sqrt(invlhs * vare) + μ
        end
    end
end

"""
    Gibbs(A, x, b)

One iteration of Gibbs sampler for multi-trait MME (general version).

Updates solution vector `x` in-place. Assumes residual variance is
already absorbed into the MME.

# Arguments
- `A`: Left-hand side matrix of MME (includes Ri)
- `x`: Solution vector (updated in-place)
- `b`: Right-hand side vector

# Notes
Handles zero diagonal elements (skip update when A[i,i] = 0).
"""
function Gibbs(A, x, b)
    for i = 1:size(x, 1)
        if A[i,i] != 0.0  # Skip zero diagonals (issue #70)
            invlhs = 1.0 / A[i,i]
            μ = invlhs * (b[i] - A[:,i]'x) + x[i]
            x[i] = randn() * sqrt(invlhs) + μ
        end
    end
end

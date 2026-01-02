################################################################################
# Gibbs Sampler Functions (used by NNMM MCMC)
################################################################################

#one iteration of Gibbs sampler for lambda version of MME (single-trait)
function Gibbs(A,x,b,vare::AbstractFloat)
    for i = 1:size(x,1)
        if A[i,i] != 0.0 #issue70, zero diagonals in MME
            invlhs  = 1.0/A[i,i]
            μ       = invlhs*(b[i] - A[:,i]'x) + x[i]
            x[i]    = randn()*sqrt(invlhs*vare) + μ
        end
    end
end

#one iteration of Gibbs sampler for general version of MME (multi-trait)
function Gibbs(A,x,b)
    for i = 1:size(x,1)
        if A[i,i] != 0.0 #issue70, zero diagonals in MME
            invlhs  = 1.0/A[i,i]
            μ       = invlhs*(b[i] - A[:,i]'x) + x[i]
            x[i]    = randn()*sqrt(invlhs) + μ
        end
    end
end

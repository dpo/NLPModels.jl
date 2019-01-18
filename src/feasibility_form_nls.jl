export FeasibilityFormNLS,
       reset!,
       obj, grad, grad!,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hess_coord, hess, hprod, hprod!


"""Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`. In other words,
converts

    min ¹/₂‖F(x)‖²
    s.t  cₗ ≤ c(x) ≤ cᵤ
          ℓ ≤   x  ≤ u

to

    min ¹/₂‖r‖²
    s.t   F(x) - r = 0
         cₗ ≤ c(x) ≤ cᵤ
          ℓ ≤   x  ≤ u

If you rather have the first problem, the `nls` model already works as an NLPModel of
that format.
"""
mutable struct FeasibilityFormNLS <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  internal :: AbstractNLSModel
  counters :: NLSCounters
end

"""
    FeasibilityFormNLS(nls)

Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`.
"""
function FeasibilityFormNLS(nls :: AbstractNLSModel)
  nequ = nls.nls_meta.nequ
  meta = nls.meta
  nvar = meta.nvar + nequ
  ncon = meta.ncon + nequ
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=[zeros(nequ); meta.lcon],
                      ucon=[zeros(nequ); meta.ucon],
                      y0=[zeros(nequ); meta.y0],
                      lin=meta.lin,
                      nln=[1:nequ; (meta.nln) .+ nequ]
                     )
  nls_meta = NLSMeta(nequ, nvar, [meta.x0; zeros(nequ)])

  nlp = FeasibilityFormNLS(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function obj(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  return dot(r, r) / 2
end

function grad(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  g = zeros(nlp.meta.nvar)
  return grad!(nlp, x, g)
end

function grad!(nlp :: FeasibilityFormNLS, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return g
end

function objgrad(nlp :: FeasibilityFormNLS, x :: Array{Float64})
  g = zeros(nlp.meta.nvar)
  return objgrad!(nlp, x, g)
end

function objgrad!(nlp :: FeasibilityFormNLS, x :: Array{Float64}, g :: Array{Float64})
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  f = dot(r, r) / 2
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return f, g
end

function cons(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  c = zeros(nlp.meta.ncon)
  return cons!(nlp, x, c)
end

function cons!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  r = @view xr[n+1:end]
  residual!(nlp.internal, x, @view c[1:ne])
  c[1:ne] .-= r
  if m > 0
    cons!(nlp.internal, x, @view c[ne+1:end])
  end
  return c
end

function jac_coord(nlp :: FeasibilityFormNLS, xr :: AbstractVector)
  J = jac(nlp, xr)
  if J isa SparseMatrixCSC
    return findnz(J)
  else
    I = findall(!iszero, J)
    return (getindex.(I, 1), getindex.(I, 2), J[I])
  end
end

function jac(nlp :: FeasibilityFormNLS, xr :: AbstractVector)
  increment!(nlp, :neval_jac)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  JF = jac_residual(nlp.internal, x)
  JC = m > 0 ? jac(nlp.internal, x) : spzeros(m, n)
  return [JF -spdiagm(0 => ones(ne)); JC spzeros(m, ne)]
end

function jprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  jv = zeros(nlp.meta.ncon)
  return jprod!(nlp, x, v, jv)
end

function jprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jprod_residual!(nlp.internal, x, v[1:n], jv[1:ne])
  @views jv[1:ne] .-= v[n+1:end]
  if m > 0
    @views jprod!(nlp.internal, x, v[1:n], jv[ne+1:end])
  end
  return jv
end

function jtprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  jtv = zeros(nlp.meta.nvar)
  return jtprod!(nlp, x, v, jtv)
end

function jtprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jtprod_residual!(nlp.internal, x, v[1:ne], jtv[1:n])
  if m > 0
    @views jtv[1:n] .+= jtprod(nlp.internal, x, v[ne+1:end])
  end
  @views jtv[n+1:end] .= -v[1:ne]
  return jtv
end

function hess_coord(nlp :: FeasibilityFormNLS, xr :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  W = hess(nlp, xr, obj_weight=obj_weight, y=y)
  if W isa SparseMatrixCSC
    return findnz(W)
  else
    I = findall(!iszero, W)
    return (getindex.(I, 1), getindex.(I, 2), W[I])
  end
end

function hess(nlp :: FeasibilityFormNLS, xr :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hess)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views Hx = m > 0 ? hess(nlp.internal, x, obj_weight=0.0, y=y[ne+1:end]) : spzeros(n, n)
  for i = 1:ne
    Hx += hess_residual(nlp.internal, x, i) * y[i]
  end
  return [Hx spzeros(n, ne); spzeros(ne, n) obj_weight * I]
end

function hprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, hv, obj_weight=obj_weight, y=y)
end

function hprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  if m > 0
    @views hprod!(nlp.internal, x, v[1:n], hv[1:n], obj_weight=0.0, y=y[ne+1:end])
  else
    fill!(hv, 0.0)
  end
  for i = 1:ne
    @views hv[1:n] .+= hprod_residual(nlp.internal, x, i, v[1:n]) * y[i]
  end
  @views hv[n+1:end] .= obj_weight * v[n+1:end]
  return hv
end

function residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  n = nlp.internal.meta.nvar
  Fx .= @view x[n+1:end]
  return Fx
end

function jac_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  return [spzeros(ne, n) I]
end

function jprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  n = nlp.internal.meta.nvar
  Jv .= @view v[n+1:end]
  return Jv
end

function jtprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  Jtv[1:n] .= 0.0
  Jtv[n+1:end] .= v
  return Jtv
end

function hess_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int)
  increment!(nlp, :neval_hess_residual)
  n = nlp.meta.nvar
  return spzeros(n, n)
end

function hprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  fill!(Hiv, 0.0)
  return Hiv
end

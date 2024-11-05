export coo_prod!, coo_sym_prod!
export @default_counters
export DimensionError, @lencheck, @rangecheck

"""
    DimensionError <: Exception
    DimensionError(name, dim_expected, dim_found)

Error for unexpected dimension.
Output: "DimensionError: Input `name` should have length `dim_expected` not `dim_found`"
"""
struct DimensionError <: Exception
  name::Union{Symbol, String}
  dim_expected::Int
  dim_found::Int
end

function Base.showerror(io::IO, e::DimensionError)
  print(
    io,
    "DimensionError: Input $(e.name) should have length $(e.dim_expected) not $(e.dim_found)",
  )
end

# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
"""
    @lencheck n x y z …

Check that arrays `x`, `y`, `z`, etc. have a prescribed length `n`.
"""
macro lencheck(l, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs, :(
      if length($(esc(var))) != $(esc(l))
        throw(DimensionError($varname, $(esc(l)), length($(esc(var)))))
      end
    ))
  end
  Expr(:block, exprs...)
end

"""
    @rangecheck ℓ u i j k …

Check that values `i`, `j`, `k`, etc. are in the range `[ℓ,u]`.
"""
macro rangecheck(lo, hi, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(
      exprs,
      :(
        if (
          length($(esc(var))) > 0 && (
            any(broadcast(<, $(esc(var)), $(esc(lo)))) ||
            any(broadcast(>, $(esc(var)), $(esc(hi))))
          )
        )
          error(string($varname, " elements must be between ", $(esc(lo)), " and ", $(esc(hi))))
        end
      ),
    )
  end
  Expr(:block, exprs...)
end

const UnconstrainedErrorMessage = "Trying to evaluate constraints, but the problem is unconstrained."

function check_constrained(nlp)
  if unconstrained(nlp)
    throw(error(UnconstrainedErrorMessage))
  end
end

const NonlinearUnconstrainedErrorMessage = "Trying to evaluate nonlinear constraints, but the problem does not have any."

function check_nonlinearly_constrained(nlp)
  if nlp.meta.nnln == 0
    throw(error(NonlinearUnconstrainedErrorMessage))
  end
end

const LinearUnconstrainedErrorMessage = "Trying to evaluate linear constraints, but the problem does not have any."

function check_linearly_constrained(nlp)
  if nlp.meta.nlin == 0
    throw(error(LinearUnconstrainedErrorMessage))
  end
end

"""
    coo_prod!(rows, cols, vals, v, Av)

Compute the product of a matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
"""
function coo_prod!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  fill!(Av, zero(eltype(v)))
  nnz = length(rows)
  @inbounds for k = 1:nnz
    i, j = rows[k], cols[k]
    Av[i] += vals[k] * v[j]
  end
  return Av
end

"""
    coo_sym_prod!(rows, cols, vals, v, Av)

Compute the product of a symmetric matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
Only one triangle of `A` should be passed.
"""
function coo_sym_prod!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  fill!(Av, zero(eltype(v)))
  nnz = length(rows)
  @inbounds for k = 1:nnz
    i, j, a = rows[k], cols[k], vals[k]
    Av[i] += a * v[j]
    if i != j
      Av[j] += a * v[i]
    end
  end
  return Av
end

"""
    @default_counters Model inner

Define functions relating counters of `Model` to counters of `Model.inner`.
"""
macro default_counters(Model, inner)
  ex = Expr(:block)
  for foo in fieldnames(Counters) ∪ [:sum_counters]
    push!(ex.args, :(NLPModels.$foo(nlp::$(esc(Model))) = $foo(nlp.$inner)))
  end
  push!(ex.args, :(NLPModels.reset!(nlp::$(esc(Model))) = begin
    reset!(nlp.$inner)
    reset_data!(nlp)
  end))
  push!(ex.args, :(NLPModels.increment!(nlp::$(esc(Model)), s::Symbol) = increment!(nlp.$inner, s)))
  push!(ex.args, :(NLPModels.decrement!(nlp::$(esc(Model)), s::Symbol) = decrement!(nlp.$inner, s)))

  push!(
    ex.args,
    :(
      Base.getproperty(nlp::$(esc(Model)), s::Symbol) =
        (s == :counters ? nlp.$inner.counters : getfield(nlp, s))
    ),
  )
  ex
end

"""
    eltype(nlp::AbstractNLPModel{T, S})

Element type of `nlp.meta.x0`.
"""
Base.eltype(nlp::AbstractNLPModel{T, S}) where {T, S} = eltype(nlp.meta.x0)

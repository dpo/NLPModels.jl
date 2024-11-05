export obj, grad, grad!, objgrad, objgrad!, objcons, objcons!
export cons, cons!, cons_lin, cons_lin!, cons_nln, cons_nln!
export jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad
export jac_structure!, jac_structure, jac_coord!, jac_coord
export jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!
export jac_lin_structure!, jac_lin_structure, jac_lin_coord!, jac_lin_coord
export jac_lin, jprod_lin, jprod_lin!, jtprod_lin, jtprod_lin!, jac_lin_op, jac_lin_op!
export jac_nln_structure!, jac_nln_structure, jac_nln_coord!, jac_nln_coord
export jac_nln, jprod_nln, jprod_nln!, jtprod_nln, jtprod_nln!, jac_nln_op, jac_nln_op!
export jth_hess_coord, jth_hess_coord!, jth_hess
export jth_hprod, jth_hprod!, ghjvprod, ghjvprod!
export hess_structure!, hess_structure, hess_coord!, hess_coord
export hess, hprod, hprod!, hess_op, hess_op!
export varscale, lagscale, conscale

"""
    f = obj(nlp, x)

Evaluate ``f(x)``, the objective function of `nlp` at `x`.
"""
function obj end

"""
    g = grad(nlp, x)

Evaluate ``∇f(x)``, the gradient of the objective function at `x`.
"""
function grad(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return grad!(nlp, x, g)
end

"""
    g = grad!(nlp, x, g)

Evaluate ``∇f(x)``, the gradient of the objective function at `x` in place.
"""
function grad! end

"""
    c = cons(nlp, x)

Evaluate ``c(x)``, the constraints at `x`.
"""
function cons(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_constrained(nlp)
  c = S(undef, nlp.meta.ncon)
  return cons!(nlp, x, c)
end

"""
    c = cons!(nlp, x, c)

Evaluate ``c(x)``, the constraints at `x` in place.
"""
function cons!(nlp::AbstractNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon cx
  check_constrained(nlp)
  increment!(nlp, :neval_cons)
  nlp.meta.nlin > 0 && cons_lin!(nlp, x, view(cx, nlp.meta.lin))
  nlp.meta.nnln > 0 && cons_nln!(nlp, x, view(cx, nlp.meta.nln))
  return cx
end

"""
    c = cons_lin(nlp, x)

Evaluate the linear constraints at `x`.
"""
function cons_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_linearly_constrained(nlp)
  c = S(undef, nlp.meta.nlin)
  return cons_lin!(nlp, x, c)
end

"""
    c = cons_lin!(nlp, x, c)

Evaluate the linear constraints at `x` in place.
"""
function cons_lin! end

"""
    c = cons_nln(nlp, x)

Evaluate the nonlinear constraints at `x`.
"""
function cons_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_nonlinearly_constrained(nlp)
  c = S(undef, nlp.meta.nnln)
  return cons_nln!(nlp, x, c)
end

"""
    c = cons_nln!(nlp, x, c)

Evaluate the nonlinear constraints at `x` in place.
"""
function cons_nln! end

function jth_con end

function jth_congrad(nlp::AbstractNLPModel{T, S}, x::AbstractVector, j::Integer) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return jth_congrad!(nlp, x, j, g)
end

function jth_congrad! end

function jth_sparse_congrad end

"""
    f, c = objcons(nlp, x)

Evaluate ``f(x)`` and ``c(x)`` at `x`.
"""
function objcons(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_constrained(nlp)
  c = S(undef, nlp.meta.ncon)
  return objcons!(nlp, x, c)
end

"""
    f, c = objcons!(nlp, x, c)

Evaluate ``f(x)`` and ``c(x)`` at `x`. `c` is overwritten with the value of ``c(x)``.
"""
function objcons!(nlp::AbstractNLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  check_constrained(nlp)
  f = obj(nlp, x)
  cons!(nlp, x, c)
  return f, c
end

"""
    f, g = objgrad(nlp, x)

Evaluate ``f(x)`` and ``∇f(x)`` at `x`.
"""
function objgrad(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return objgrad!(nlp, x, g)
end

"""
    f, g = objgrad!(nlp, x, g)

Evaluate ``f(x)`` and ``∇f(x)`` at `x`. `g` is overwritten with the
value of ``∇f(x)``.
"""
function objgrad!(nlp::AbstractNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  f = obj(nlp, x)
  grad!(nlp, x, g)
  return f, g
end

"""
    (rows,cols) = jac_structure(nlp)

Return the structure of the constraints Jacobian in sparse coordinate format.
"""
function jac_structure(nlp::AbstractNLPModel)
  check_constrained(nlp)
  rows = Vector{Int}(undef, nlp.meta.nnzj)
  cols = Vector{Int}(undef, nlp.meta.nnzj)
  jac_structure!(nlp, rows, cols)
end

"""
    jac_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian in sparse coordinate format in place.
"""
function jac_structure!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T}
  check_constrained(nlp)
  @lencheck nlp.meta.nnzj rows cols
  lin_ind = 1:(nlp.meta.lin_nnzj)
  nlp.meta.nlin > 0 && jac_lin_structure!(nlp, view(rows, lin_ind), view(cols, lin_ind))
  for i in lin_ind
    rows[i] += count(x < nlp.meta.lin[rows[i]] for x in nlp.meta.nln)
  end
  if nlp.meta.nnln > 0
    nln_ind = (nlp.meta.lin_nnzj + 1):(nlp.meta.lin_nnzj + nlp.meta.nln_nnzj)
    jac_nln_structure!(nlp, view(rows, nln_ind), view(cols, nln_ind))
    for i in nln_ind
      rows[i] += count(x < nlp.meta.nln[rows[i]] for x in nlp.meta.lin)
    end
  end
  return rows, cols
end

"""
    (rows,cols) = jac_lin_structure(nlp)

Return the structure of the linear constraints Jacobian in sparse coordinate format.
"""
function jac_lin_structure(nlp::AbstractNLPModel)
  check_linearly_constrained(nlp)
  rows = Vector{Int}(undef, nlp.meta.lin_nnzj)
  cols = Vector{Int}(undef, nlp.meta.lin_nnzj)
  jac_lin_structure!(nlp, rows, cols)
end

"""
    jac_lin_structure!(nlp, rows, cols)

Return the structure of the linear constraints Jacobian in sparse coordinate format in place.
"""
function jac_lin_structure! end

"""
    (rows,cols) = jac_nln_structure(nlp)

Return the structure of the nonlinear constraints Jacobian in sparse coordinate format.
"""
function jac_nln_structure(nlp::AbstractNLPModel)
  check_nonlinearly_constrained(nlp)
  rows = Vector{Int}(undef, nlp.meta.nln_nnzj)
  cols = Vector{Int}(undef, nlp.meta.nln_nnzj)
  jac_nln_structure!(nlp, rows, cols)
end

"""
    jac_nln_structure!(nlp, rows, cols)

Return the structure of the nonlinear constraints Jacobian in sparse coordinate format in place.
"""
function jac_nln_structure! end

"""
    vals = jac_coord!(nlp, x, vals)

Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
function jac_coord!(nlp::AbstractNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  check_constrained(nlp)
  increment!(nlp, :neval_jac)
  lin_ind = 1:(nlp.meta.lin_nnzj)
  nlp.meta.nlin > 0 && jac_lin_coord!(nlp, x, view(vals, lin_ind))
  nln_ind = (nlp.meta.lin_nnzj + 1):(nlp.meta.lin_nnzj + nlp.meta.nln_nnzj)
  nlp.meta.nnln > 0 && jac_nln_coord!(nlp, x, view(vals, nln_ind))
  return vals
end

"""
    vals = jac_coord(nlp, x)

Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format.
"""
function jac_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_constrained(nlp)
  vals = S(undef, nlp.meta.nnzj)
  return jac_coord!(nlp, x, vals)
end

"""
    Jx = jac(nlp, x)

Evaluate ``J(x)``, the constraints Jacobian at `x` as a sparse matrix.
"""
function jac(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  check_constrained(nlp)
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.ncon, nlp.meta.nvar)
end

"""
    vals = jac_lin_coord!(nlp, x, vals)

Evaluate ``J(x)``, the linear constraints Jacobian at `x` in sparse coordinate format,
overwriting `vals`.
"""
function jac_lin_coord! end

"""
    vals = jac_lin_coord(nlp, x)

Evaluate ``J(x)``, the linear constraints Jacobian at `x` in sparse coordinate format.
"""
function jac_lin_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_linearly_constrained(nlp)
  vals = S(undef, nlp.meta.lin_nnzj)
  return jac_lin_coord!(nlp, x, vals)
end

"""
    Jx = jac_lin(nlp, x)

Evaluate ``J(x)``, the linear constraints Jacobian at `x` as a sparse matrix.
"""
function jac_lin(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  check_linearly_constrained(nlp)
  rows, cols = jac_lin_structure(nlp)
  vals = jac_lin_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.nlin, nlp.meta.nvar)
end

"""
    vals = jac_nln_coord!(nlp, x, vals)

Evaluate ``J(x)``, the nonlinear constraints Jacobian at `x` in sparse coordinate format,
overwriting `vals`.
"""
function jac_nln_coord! end

"""
    vals = jac_nln_coord(nlp, x)

Evaluate ``J(x)``, the nonlinear constraints Jacobian at `x` in sparse coordinate format.
"""
function jac_nln_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_nonlinearly_constrained(nlp)
  vals = S(undef, nlp.meta.nln_nnzj)
  return jac_nln_coord!(nlp, x, vals)
end

"""
    Jx = jac_nln(nlp, x)

Evaluate ``J(x)``, the nonlinear constraints Jacobian at `x` as a sparse matrix.
"""
function jac_nln(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  check_nonlinearly_constrained(nlp)
  rows, cols = jac_nln_structure(nlp)
  vals = jac_nln_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.nnln, nlp.meta.nvar)
end

"""
    Jv = jprod(nlp, x, v)

Evaluate ``J(x)v``, the Jacobian-vector product at `x`.
"""
function jprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x v
  check_constrained(nlp)
  Jv = S(undef, nlp.meta.ncon)
  return jprod!(nlp, x, v, Jv)
end

"""
    Jv = jprod!(nlp, x, v, Jv)

Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.
"""
function jprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  check_constrained(nlp)
  increment!(nlp, :neval_jprod)
  nlp.meta.nlin > 0 && jprod_lin!(nlp, x, v, view(Jv, nlp.meta.lin))
  nlp.meta.nnln > 0 && jprod_nln!(nlp, x, v, view(Jv, nlp.meta.nln))
  return Jv
end

"""
    Jv = jprod!(nlp, rows, cols, vals, v, Jv)

Evaluate ``J(x)v``, the Jacobian-vector product, where the Jacobian is given by
`(rows, cols, vals)` in triplet format.
"""
function jprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.ncon Jv
  check_constrained(nlp)
  increment!(nlp, :neval_jprod)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jv = jprod_lin(nlp, x, v)

Evaluate ``J(x)v``, the linear Jacobian-vector product at `x`.
"""
function jprod_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x v
  check_linearly_constrained(nlp)
  Jv = S(undef, nlp.meta.nlin)
  return jprod_lin!(nlp, x, v, Jv)
end

"""
    Jv = jprod_lin!(nlp, x, v, Jv)

Evaluate ``J(x)v``, the linear Jacobian-vector product at `x` in place.
"""
function jprod_lin! end

"""
    Jv = jprod_lin!(nlp, rows, cols, vals, v, Jv)

Evaluate ``J(x)v``, the linear Jacobian-vector product, where the Jacobian is given by
`(rows, cols, vals)` in triplet format.
"""
function jprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nlin Jv
  check_linearly_constrained(nlp)
  increment!(nlp, :neval_jprod_lin)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jv = jprod_nln(nlp, x, v)

Evaluate ``J(x)v``, the nonlinear Jacobian-vector product at `x`.
"""
function jprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x v
  check_nonlinearly_constrained(nlp)
  Jv = S(undef, nlp.meta.nnln)
  return jprod_nln!(nlp, x, v, Jv)
end

"""
    Jv = jprod_nln!(nlp, x, v, Jv)

Evaluate ``J(x)v``, the nonlinear Jacobian-vector product at `x` in place.
"""
function jprod_nln! end

"""
    Jv = jprod_nln!(nlp, rows, cols, vals, v, Jv)

Evaluate ``J(x)v``, the nonlinear Jacobian-vector product, where the Jacobian is given by
`(rows, cols, vals)` in triplet format.
"""
function jprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nnln Jv
  check_nonlinearly_constrained(nlp)
  increment!(nlp, :neval_jprod_nln)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jtv = jtprod(nlp, x, v)

Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jtprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon v
  check_constrained(nlp)
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod!(nlp, x, v, Jtv)

Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
If the problem has linear and nonlinear constraints, this function allocates.
"""
function jtprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  check_constrained(nlp)
  increment!(nlp, :neval_jtprod)
  if nlp.meta.nnln == 0
    jtprod_lin!(nlp, x, v, Jtv)
  elseif nlp.meta.nlin == 0
    jtprod_nln!(nlp, x, v, Jtv)
  elseif nlp.meta.nlin >= nlp.meta.nnln
    jtprod_lin!(nlp, x, view(v, nlp.meta.lin), Jtv)
    if nlp.meta.nnln > 0
      Jtv .+= jtprod_nln(nlp, x, view(v, nlp.meta.nln))
    end
  else
    jtprod_nln!(nlp, x, view(v, nlp.meta.nln), Jtv)
    if nlp.meta.nlin > 0
      Jtv .+= jtprod_lin(nlp, x, view(v, nlp.meta.lin))
    end
  end
  return Jtv
end

"""
    Jtv = jtprod!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product, where the
Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jtprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon v
  @lencheck nlp.meta.nvar Jtv
  check_constrained(nlp)
  increment!(nlp, :neval_jtprod)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    Jtv = jtprod_lin(nlp, x, v)

Evaluate ``J(x)^Tv``, the linear transposed-Jacobian-vector product at `x`.
"""
function jtprod_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin v
  check_linearly_constrained(nlp)
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod_lin!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod_lin!(nlp, x, v, Jtv)

Evaluate ``J(x)^Tv``, the linear transposed-Jacobian-vector product at `x` in place.
"""
function jtprod_lin! end

"""
    Jtv = jtprod_lin!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the linear transposed-Jacobian-vector product, where the
Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jtprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin v
  @lencheck nlp.meta.nvar Jtv
  check_linearly_constrained(nlp)
  increment!(nlp, :neval_jtprod_lin)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    Jtv = jtprod_nln(nlp, x, v)

Evaluate ``J(x)^Tv``, the nonlinear transposed-Jacobian-vector product at `x`.
"""
function jtprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln v
  check_nonlinearly_constrained(nlp)
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod_nln!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod_nln!(nlp, x, v, Jtv)

Evaluate ``J(x)^Tv``, the nonlinear transposed-Jacobian-vector product at `x` in place.
"""
function jtprod_nln! end

"""
    Jtv = jtprod_nln!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the nonlinear transposed-Jacobian-vector product, where the
Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jtprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln v
  @lencheck nlp.meta.nvar Jtv
  check_nonlinearly_constrained(nlp)
  increment!(nlp, :neval_jtprod_nln)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    J = jac_op(nlp, x)

Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_constrained(nlp)
  Jv = S(undef, nlp.meta.ncon)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_op!(nlp, x, Jv, Jtv)

Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv
  check_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod!(nlp, x, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod!(nlp, x, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon Jv
  @lencheck nlp.meta.nvar Jtv
  check_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_lin_op(nlp, x)

Return the linear Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_lin_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_linearly_constrained(nlp)
  Jv = S(undef, nlp.meta.nlin)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_lin_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_lin_op!(nlp, x, Jv, Jtv)

Return the linear Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_lin_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nlin Jv
  check_linearly_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_lin!(nlp, x, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_lin!(nlp, x, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_lin_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the linear Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_lin_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin Jv
  @lencheck nlp.meta.nvar Jtv
  check_linearly_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_lin!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_lin!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_nln_op(nlp, x)

Return the nonlinear Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_nln_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  check_nonlinearly_constrained(nlp)
  Jv = S(undef, nlp.meta.nnln)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_nln_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_nln_op!(nlp, x, Jv, Jtv)

Return the nonlinear Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_nln_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nnln Jv
  check_nonlinearly_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_nln!(nlp, x, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_nln!(nlp, x, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_nln_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the nonlinear Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_nln_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln Jv
  @lencheck nlp.meta.nvar Jtv
  check_nonlinearly_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_nln!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_nln!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

"""
    vals = jth_hess_coord(nlp, x, j)

Evaluate the Hessian of j-th constraint at `x` in sparse coordinate format.
Only the lower triangle is returned.
"""
function jth_hess_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector, j::Integer) where {T, S}
  @lencheck nlp.meta.nvar x
  @rangecheck 1 nlp.meta.ncon j
  check_constrained(nlp)
  vals = S(undef, nlp.meta.nnzh)
  return jth_hess_coord!(nlp, x, j, vals)
end

"""
    vals = jth_hess_coord!(nlp, x, j, vals)

Evaluate the Hessian of j-th constraint at `x` in sparse coordinate format, with `vals` of
length `nlp.meta.nnzh`, in place. Only the lower triangle is returned.
"""
function jth_hess_coord! end

"""
   Hx = jth_hess(nlp, x, j)

Evaluate the Hessian of j-th constraint at `x` as a sparse matrix with
the same sparsity pattern as the Lagrangian Hessian.
A `Symmetric` object wrapping the lower triangle is returned.
"""
function jth_hess(nlp::AbstractNLPModel, x::AbstractVector, j::Integer)
  @lencheck nlp.meta.nvar x
  @rangecheck 1 nlp.meta.ncon j
  check_constrained(nlp)
  rows, cols = hess_structure(nlp)
  vals = jth_hess_coord(nlp, x, j)
  return Symmetric(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar), :L)
end

"""
    Hv = jth_hprod(nlp, x, v, j)

Evaluate the product of the Hessian of j-th constraint at `x` with the vector `v`.
"""
function jth_hprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
) where {T, S}
  @lencheck nlp.meta.nvar x v
  @rangecheck 1 nlp.meta.ncon j
  check_constrained(nlp)
  Hv = S(undef, nlp.meta.nvar)
  return jth_hprod!(nlp, x, v, j, Hv)
end

"""
    Hv = jth_hprod!(nlp, x, v, j, Hv)

Evaluate the product of the Hessian of j-th constraint at `x` with the vector `v`
in place.
"""
function jth_hprod! end

"""
   gHv = ghjvprod(nlp, x, g, v)

Return the vector whose i-th component is gᵀ ∇²cᵢ(x) v.
"""
function ghjvprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x g v
  check_constrained(nlp)
  gHv = S(undef, nlp.meta.ncon)
  return ghjvprod!(nlp, x, g, v, gHv)
end

"""
   ghjvprod!(nlp, x, g, v, gHv)

Return the vector whose i-th component is gᵀ ∇²cᵢ(x) v in place.
"""
function ghjvprod! end

"""
    (rows,cols) = hess_structure(nlp)

Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hess_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  hess_structure!(nlp, rows, cols)
end

"""
    hess_structure!(nlp, rows, cols)

Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function hess_structure! end

"""
    vals = hess_coord!(nlp, x, vals; obj_weight=1.0)

Evaluate the objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(OBJECTIVE_HESSIAN), overwriting `vals`.
Only the lower triangle is returned.
"""
function hess_coord!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  y = fill!(S(undef, nlp.meta.ncon), 0)
  hess_coord!(nlp, x, y, vals, obj_weight = obj_weight)
end

"""
    vals = hess_coord!(nlp, x, y, vals; obj_weight=1.0)

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN), overwriting `vals`.
Only the lower triangle is returned.
"""
function hess_coord! end

"""
    vals = hess_coord(nlp, x; obj_weight=1.0)

Evaluate the objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

$(OBJECTIVE_HESSIAN).
Only the lower triangle is returned.
"""
function hess_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.meta.nnzh)
  return hess_coord!(nlp, x, vals; obj_weight = obj_weight)
end

"""
    vals = hess_coord(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

$(LAGRANGIAN_HESSIAN).
Only the lower triangle is returned.
"""
function hess_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  check_constrained(nlp)
  vals = S(undef, nlp.meta.nnzh)
  return hess_coord!(nlp, x, y, vals; obj_weight = obj_weight)
end

"""
    Hx = hess(nlp, x; obj_weight=1.0)

Evaluate the objective Hessian at `x` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

$(OBJECTIVE_HESSIAN).
A `Symmetric` object wrapping the lower triangle is returned.
"""
function hess(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, obj_weight = obj_weight)
  Symmetric(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar), :L)
end

"""
    Hx = hess(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

$(LAGRANGIAN_HESSIAN).
A `Symmetric` object wrapping the lower triangle is returned.
"""
function hess(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  check_constrained(nlp)
  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, y, obj_weight = obj_weight)
  Symmetric(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar), :L)
end

"""
    Hv = hprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`, where the objective Hessian is
$(OBJECTIVE_HESSIAN).
"""
function hprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x v
  Hv = S(undef, nlp.meta.nvar)
  return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
$(LAGRANGIAN_HESSIAN).
"""
function hprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon y
  check_constrained(nlp)
  Hv = S(undef, nlp.meta.nvar)
  return hprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0)

Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
$(OBJECTIVE_HESSIAN).
"""
function hprod!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x v Hv
  y = fill!(S(undef, nlp.meta.ncon), 0)
  hprod!(nlp, x, y, v, Hv, obj_weight = obj_weight)
end

"""
    Hv = hprod!(nlp, rows, cols, vals, v, Hv)

Evaluate the product of the objective or Lagrangian Hessian given by `(rows, cols, vals)` in
triplet format with the vector `v` in place. Only one triangle of the Hessian should be given.
"""
function hprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.meta.nvar v Hv
  increment!(nlp, :neval_hprod)
  coo_sym_prod!(cols, rows, vals, v, Hv)
end

"""
    Hv = hprod!(nlp, x, y, v, Hv; obj_weight=1.0)

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
$(LAGRANGIAN_HESSIAN).
"""
function hprod! end

"""
    H = hess_op(nlp, x; obj_weight=1.0)

Return the objective Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
$(OBJECTIVE_HESSIAN).
"""
function hess_op(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  Hv = S(undef, nlp.meta.nvar)
  return hess_op!(nlp, x, Hv, obj_weight = obj_weight)
end

"""
    H = hess_op(nlp, x, y; obj_weight=1.0)

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
$(LAGRANGIAN_HESSIAN).
"""
function hess_op(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  check_constrained(nlp)
  Hv = S(undef, nlp.meta.nvar)
  return hess_op!(nlp, x, y, Hv, obj_weight = obj_weight)
end

"""
    H = hess_op!(nlp, x, Hv; obj_weight=1.0)

Return the objective Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
$(OBJECTIVE_HESSIAN).
"""
function hess_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end

"""
    H = hess_op!(nlp, rows, cols, vals, Hv)

Return the Hessian given by `(rows, cols, vals)` as a linear operator,
and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`.
  The vector `Hv` is used as preallocated storage for the operation.  The linear operator H
represents
$(OBJECTIVE_HESSIAN).
"""
function hess_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Hv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.meta.nvar Hv
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, rows, cols, vals, v, Hv)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end

"""
    H = hess_op!(nlp, x, y, Hv; obj_weight=1.0)

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
$(LAGRANGIAN_HESSIAN).
"""
function hess_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  @lencheck nlp.meta.ncon y
  check_constrained(nlp)
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end

function varscale end
function lagscale end
function conscale end

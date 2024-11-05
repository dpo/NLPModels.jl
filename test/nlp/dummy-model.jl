mutable struct DummyModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
end

@testset "Default methods throw MethodError on DummyModel since they're not defined" begin
  model = DummyModel(NLPModelMeta(1))
  @test_throws(MethodError, lagscale(model, 1.0))
  @test_throws(MethodError, obj(model, [0.0]))
  @test_throws(MethodError, varscale(model, [0.0]))
  @test_throws(MethodError, conscale(model, [0.0]))
  @test_throws(MethodError, jac_structure(model, [0], [1]))
  @test_throws(MethodError, hess_structure(model, [0], [1]))
  @test_throws(MethodError, grad!(model, [0.0], [1.0]))
  @test_throws(MethodError, cons_lin!(model, [0.0], [1.0]))
  @test_throws(MethodError, cons_nln!(model, [0.0], [1.0]))
  @test_throws(MethodError, jac_lin_coord!(model, [0.0], [1.0]))
  @test_throws(MethodError, jac_nln_coord!(model, [0.0], [1.0]))
  @test_throws(MethodError, jth_con(model, [0.0], 1))
  @test_throws(MethodError, jth_congrad(model, [0.0], 1))
  @test_throws(MethodError, jth_sparse_congrad(model, [0.0], 1))
  @test_throws(MethodError, jth_congrad!(model, [0.0], 1, [2.0]))
  @test_throws(MethodError, jprod_lin!(model, [0.0], [1.0], [2.0]))
  @test_throws(MethodError, jtprod_lin!(model, [0.0], [1.0], [2.0]))
  @test_throws(MethodError, jprod_nln!(model, [0.0], [1.0], [2.0]))
  @test_throws(MethodError, jtprod_nln!(model, [0.0], [1.0], [2.0]))
  @test_throws(MethodError, jth_hess_coord!(model, [0.0], 1))
  @test_throws(MethodError, jth_hprod!(model, [0.0], [1.0], 2, [3.0]))
  @test_throws(MethodError, ghjvprod!(model, [0.0], [1.0], [2.0], [3.0]))
  @assert isa(hess_op(model, [0.0]), LinearOperator)
  @test_throws ErrorException("Trying to evaluate constraints, but the problem is unconstrained.") jac_op(model, [0.0])
  @test_throws ErrorException("Trying to evaluate linear constraints, but the problem does not have any.") jac_lin_op(model, [0.0])
  @test_throws ErrorException("Trying to evaluate nonlinear constraints, but the problem does not have any.") jac_nln_op(model, [0.0])
end

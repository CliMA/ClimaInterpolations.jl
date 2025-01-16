#=
using Revise; using TestEnv; TestEnv.activate(); include("test/runtests.jl")
=#
using Test
using ClimaInterpolations
using SafeTestsets

#! format: off
@safetestset "Aqua" begin @time include("aqua.jl") end
#! format: on

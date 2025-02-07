using ImportanceSampling
using Test
using Aqua

@testset "ImportanceSampling.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImportanceSampling)
    end
    # Write your tests here.
end

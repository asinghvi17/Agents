using Flux

@testset "FanOut" begin
    m = Agents.FanOut(3)
    @test m([5]) == ([5], [5], [5])
end


@testset "Parallel" begin
    @testset "Basic" begin
        m = Agents.Parallel(x -> x^2, x -> x+1)
        @test m((5, 5)) == (25, 6)
        @test m(5, 5) == (25, 6)
    end

    @testset "Indexing" begin
        d1, d2, d3 = Dense(10, 5), Dense(5, 2), Dense(5, 7)
        m = Agents.Parallel(d1, d2, d3)
        @test m[1] == d1
        @test m[2] == d2
        @test m[3] == d3
        @test m[1:2] == Agents.Parallel(d1, d2)
    end

    @testset "Inputs" begin
        m = Agents.Parallel(Dense(10, 5), Dense(5, 2))
        x, y = randn(10), randn(5)
        @test m((x, y)) == (m[1](x), m[2](y))
        @test m(x, y) == (m[1](x), m[2](y))
    end
end

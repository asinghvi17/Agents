"""
    FanOut(n)

Repeat the argument N times

```julia
m = FanOut(3)
m([5]) == ([5], [5], [5])
```
"""
struct FanOut
  n::Int
end

(f::FanOut)(x) = tuple([x for _ ∈ 1:f.n]...)

Base.show(io::IO, f::FanOut) = print(io, "FanOut($(f.n))")


"""
    Parallel(layers...)

Call N layers on N inputs to produce N outputs

```julia
m = Parallel(x -> x^2, x -> x+1)
m((5, 5)) == [25, 6]
m(5, 5) == [25, 6]

m = Parallel(Dense(10, 5), Dense(5, 2))
x = rand(10)
m((x, x)) == [m[1](x), m[2](x)]
m(x, x) == [m[1](x), m[2](x)]

```
`Parallel` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1](x)` will calculate the output of the first layer.
"""
struct Parallel{T<:Tuple}
  layers::T
  Parallel(xs...) = new{typeof(xs)}(xs)
end

@forward Parallel.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

children(p::Parallel) = p.layers
mapchildren(f, p::Parallel) = Parallel(f.(c.layers)...)

(p::Parallel)(xs::Tuple) = map((layer, x) -> layer(x), p.layers, xs)
(p::Parallel)(xs...) = map((layer, x) -> layer(x), p.layers, xs)

Base.getindex(p::Parallel, i::AbstractArray) = Parallel(p.layers[i]...)

function Base.show(io::IO, p::Parallel)
  print(io, "Parallel(")
  join(io, p.layers, ", ")
  print(io, ")")
end

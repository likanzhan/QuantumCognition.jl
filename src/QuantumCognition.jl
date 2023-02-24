module QuantumCognition

using LinearAlgebra

export BuildInitialState, BuildUnitaryMatrix, BuildCompatible, BuildProjectors

"""
    BuildInitialState(magnitude, phase)
    BuildInitialState(data)

Build a unit length complex vector

# Examples

```julia
BuildInitialState(rand(3), rand(3))
BuildInitialState(rand(6))
```
"""
BuildInitialState(magnitude, phase) = normalize(@. magnitude + phase * im)
function BuildInitialState(data)
    iseven(length(data)) || error("长度应该是偶数")
    return BuildInitialState(data[1:Int(length(data)/2)], data[Int(length(data)/2)+1:end])
end

"""
    BuildUnitaryMatrix(P::AbstractVector)

In linear algebra, an invertible complex square matrix `U`` is unitary if its conjugate transpose is also its inverse.

# Examples
```julia
BuildUnitaryMatrix(rand(9))
```
"""
function BuildUnitaryMatrix(P::AbstractVector)
    dim = sqrt(length(P))
    isinteger(dim) || error("长度 != N^2")
    H = zeros(Complex, Int(dim), Int(dim))
    i = 1
    for idx in CartesianIndices(H)
        if idx[1] == idx[2]
            H[idx] = P[i]
            i += 1
        elseif idx[1] < idx[2]
            H[idx] = P[i] + P[i+1]*im
            i += 2
        end
    end
    H = Hermitian(H, :U)
    U = exp(-im * H)
    return U
end

"""
    BuildCompatible(vs::Vector{Int})

Build projectors for `p` compatible variables, `Y_1, \\cdots, Y_i, \\cdots, Y_p`;
Argument `vs` is a `p` dimensional vector, `[n_1, n_2, \\cdots, n_p]`
where `p` is the counts of compatible variables, 
and `n_i` is the dimension of the `i^{th}` variable.

Return `p` dimension vector of vectors `M`, 
where vector `M[i]` is projectors for the `i^{th}` variable.

`M[i]` is a `n_i` dimension vector of matrixes, 
where `M[i][j]` is the `j^{th}` projector for the variable `Y_i`,
and `M[i][j]` is a `n_1 * n_2 * \\cdots * \\n_p` sized squared matrix.

`M[i][j]` is the serial kron product of the following matrixes 
`I_1, \\cdots, I_{j-1}, MIJ, I_{j+1}, \\cdots, I_p`,
Where `I_i` is the `n_i` dimensional identiry matrix,
and `MIJ` is the `j^{th}` projector of the `i`th variable.

`MIJ` is a `n_i x n_i` squared zero matrix, 
with the `j^{th}` column is replaced by the `n x 1` sized column
`L_j = [0, 0, \\cdots, 1, \\cdots, 0, 0]`,
where the `j^{th}` element is 1, and the remaining elements in `L_j` are zeros.

## Examples
```julia
using LinearAlgebra
n1, n2, n3 = 2, 4, 3
M    = BuildCompatible([n1, n2, n3])
M12  = 1I(n1) .* 1I(n1)[2, :]
M[1][2] == reduce(kron, [M12, 1I(4), 1I(3)]) # The same
```
"""
function BuildCompatible(vs::Vector{Int})
    MtS = Vector{Vector}(undef, length(vs))
    IdS = Vector{Matrix}(undef, length(vs))
    for (idx, val) in enumerate(vs)
        MtS[idx] = Vector{Matrix{Union{Int, Complex}}}(undef, val)
        # MtS[idx] = Vector{Matrix{Union{Int, Complex}}}(undef, prod(vs))
        IdS[idx] = 1I(val)
    end
    for (idx, val) in enumerate(vs)
        for v in 1:val
            IdC         = copy(IdS)
            IdC[idx]    = 1I(val) .* 1I(val)[v, :]
            MtS[idx][v] = reduce(kron, IdC)
        end
        # for v in (val+1):prod(vs)
        #     MtS[idx][v] = zeros(Int, prod(vs), prod(vs))
        # end
    end
    return MtS
end
BuildCompatible(vs...) = BuildCompatible([vs...]) # Splat to vector


"""
    BuildProjectors(VarList, CompIndex, UnitaryList, InCompIndex)

# Example
```julia
VarList     = [1 => 2, 2 => 3, 3 => 2, 4 => 2]
CompIndex   = [1, 3] # index of the compatible variables
UnitaryList = [BuildUnitaryMatrix(rand(16)) for _ in 1:2]
InCompIndex = [2 => 1 => 1, 4 => 3 => 2]

Projectors = BuildProjectors(VarList, CompIndex, UnitaryList, InCompIndex)
Projectors[1][1]
```
"""
function BuildProjectors(VarList, CompIndex, UnitaryList, InCompIndex)
    VarList     = sort(collect(VarList), by = x -> first(x))
    CompIndex   = sort(CompIndex)
    InCompIndex = sort(collect(InCompIndex), by = x -> first(x))

    Projectors  = Vector(undef, length(VarList))
    CompValue   = [last(var) for var in VarList if first(var) in CompIndex]

    Projectors[CompIndex] = BuildCompatible(CompValue)
    
    for val in InCompIndex
        tar, src, uni = first(val), first(last(val)), last(last(val))
        Projectors[tar] = map(X -> UnitaryList[uni] * X * UnitaryList[uni]', Projectors[src])
    end

    return Projectors
end

end ## Module

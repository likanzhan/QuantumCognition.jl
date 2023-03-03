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
BuildInitialState(magnitude, phase) = normalize(magnitude .+ phase * im)
BuildInitialState(magnitude)        = normalize(magnitude)

"""
    BuildUnitaryMatrix(REAL::AbstractVector, IMAG::AbstractVector)

If `X` is skew-Hermitian, then `e^X` is unitary. `X` is skew-Hermitian if and only if `im * X` or `-im * X` is Hermitian.

In mathematics, a Hermitian matrix (or self-adjoint matrix) is a complex square matrix that is equal to its own conjugate transpose - 
that is, the element in the `i`-th row and `j`-th column is equal to the complex conjugate of the element in the `j`-th row and `i`-th column, 
for all indices `i` and `j``. Or in matrix form, ``A = \\overhat{A^T}``. Hermitian matrices can be understood as the complex extension of real symmetric matrices.

A `N x N` Hermitian matrix is determined by `N(N-1)/2` scalars above the main diagonal, and `N` scalars on the main diagonal.
The entries on the main diagonal (top left to bottom right) of any Hermitian matrix are real. And other elements are complex.
So the real part of matrix is determined by `N(N+1)/2 = N(N-1)/2 + N` scalars, and 
the imaginary part of the matrix is determined by `N(N-1)/2` scalars.

And `N^2` scalars in total are needed to create a `N x N` Hermitian matrix.

# Examples
```julia
RL, IM = rand(6), rand(3)
U = BuildUnitaryMatrix(RL, IM)
U'U ≈ U*U' ≈ U*inv(U) ≈ I
```
"""
function BuildUnitaryMatrix(REAL::AbstractVector, IMAG::AbstractVector)
    dimRL = (-1 + sqrt(1+8*length(REAL))) / 2 
    dimIM = ( 1 + sqrt(1+8*length(IMAG))) / 2 
    isinteger(dimRL) || error("REAL 长度 != N(N+1)/2")
    isinteger(dimIM) || error("IMAG 长度 != N(N-1)/2")
    dimRL == dimIM   || error("长度 REAL != IMAG")
    H = zeros(Complex, Int(dimRL), Int(dimRL))
    i, j = 1, 1
    for idx in CartesianIndices(H)
        if idx[1] > idx[2]
            H[idx] = REAL[i] + IMAG[j]*im
            i += 1
            j += 1
        elseif idx[1] == idx[2]
            H[idx] = REAL[i]
            i += 1
        elseif idx[1] < idx[2]
            H[idx] = (H[idx[2], idx[1]])'
        end
    end
    U = exp(-im * H)
    return U
end

"""
    BuildUnitaryMatrix(REAL::AbstractVector)

If `X` is a skew-symmetric matrix (反对称矩阵), then matrix exponential `e^X` is orthogonal. 
A matrix `X` is skew-symmetric if and only if `X' == -X`. 
In terms of the entries of the matrix, if ``a_{ij}`` denotes the entry in the `i`-th row and `j`-th column, 
then the skew-symmetric condition is equivalent to ``a_{ji} = -a_{ij}``. 

The elements on the diagonal of a skew-symmetric matrix are zero, and therefore its trace equals zero. 
A `N x N` skew-symmetric matrix is determined by ``N(N-1)/2`` scalars (the number of entries above the main diagonal)

# Example
```julia
U = BuildUnitaryMatrix(rand(6))
U'U ≈ U*U' ≈ U*inv(U) ≈ I
```
"""
function BuildUnitaryMatrix(REAL::AbstractVector)
    dim = (1 + sqrt(1+8*length(REAL))) / 2 
    isinteger(dim) || error("长度 != N(N-1)/2")
    H = zeros(Int(dim), Int(dim))
    i = 1
    for idx in CartesianIndices(H)
        if idx[1] > idx[2]
            H[idx] = REAL[i]
            i += 1
        elseif idx[1] < idx[2]
            H[idx] = -H[idx[2], idx[1]]
        end
    end
    U = exp(H)
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
            # IdC[idx]    = 1I(val)[v, :] * 1I(val)[v, :]'
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

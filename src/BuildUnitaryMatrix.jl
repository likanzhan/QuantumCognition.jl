"""
    BuildReal(REAL::AbstractVector)

In linear algebra, a symmetric matrix is a square matrix that is equal to its transpose. Formally, `A` is symmetric iff `A=A^T`. 
Because equal matrices have equal dimensions, only square matrices can be symmetric.
The entries of a symmetric matrix are symmetric with respect to the main diagonal. 
So if `a_{ij}` denotes the entry in the `i`th row and `j`th column then `A` is symmetric iff for every `i, j`, `a_{ji} = a_{ij}` for all indices `i` and `j`.
"""
function BuildReal(REAL::AbstractVector)
    dim = (-1 + sqrt(1+8*length(REAL))) / 2 
    isinteger(dim) || error("Build `N X N` matrix, need `N(N+1)/2` values")
    H = zeros(Int(dim), Int(dim))
    i = 1
    for idx in CartesianIndices(H)
        if idx[1] >= idx[2]
            H[idx] = REAL[i]
            i += 1
        # if idx[1] > idx[2]
        #     H[idx] = REAL[i]
        #     i += 1
        # elseif idx[1] == idx[2]
        #     H[idx] = REAL[i]
        #     i += 1
        elseif idx[1] < idx[2]
            H[idx] = H[idx[2], idx[1]]
        end
    end
    return H
end

"""
    BuildImaginary(IMAG::AbstractVector)

A matrix `X` is skew-symmetric if and only if `X' == -X`. 
In terms of the entries of the matrix, 
if ``a_{ij}`` denotes the entry in the `i`-th row and `j`-th column, 
then the skew-symmetric condition is equivalent to ``a_{ji} = -a_{ij}``. 

The elements on the diagonal of a skew-symmetric matrix are zero, and therefore its trace equals zero. 
A `N x N` skew-symmetric matrix is determined by ``N(N-1)/2`` scalars (the number of entries above the main diagonal)
"""
function BuildImaginary(IMAG::AbstractVector)
    dimIM = ( 1 + sqrt(1+8*length(IMAG))) / 2 
    isinteger(dimIM) || error("Build `N x N` matrix, need `N(N-1)/2` values for IMAG part")
    H = zeros(Complex, Int(dimIM), Int(dimIM))
    j = 1
    for idx in CartesianIndices(H)
        if idx[1] > idx[2]
            H[idx] = IMAG[j]*im
            j += 1
        elseif idx[1] < idx[2]
            H[idx] = -(H[idx[2], idx[1]])
        end
    end
    return H
end

"""
    BuildHermitianMatrix(REAL::AbstractVector)

In linear algebra, a symmetric matrix is a square matrix that is equal to its transpose. Formally, `A` is symmetric iff `A=A^T`. 
Because equal matrices have equal dimensions, only square matrices can be symmetric.
The entries of a symmetric matrix are symmetric with respect to the main diagonal. 
So if `a_{ij}` denotes the entry in the `i`th row and `j`th column then `A` is symmetric iff for every `i, j`, `a_{ji} = a_{ij}` for all indices `i` and `j`.
"""
BuildHermitianMatrix(REAL::AbstractVector) = BuildReal(REAL)

"""
    BuildHermitianMatrix(REAL::AbstractVector, IMAG::AbstractVector)

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
function BuildHermitianMatrix(REAL::AbstractVector, IMAG::AbstractVector)
    RealPart, ImaginaryPart = BuildReal(REAL), BuildImaginary(IMAG)
    size(RealPart) == size(ImaginaryPart) || error("Size of Real part and imaginary part differs")
    return RealPart + ImaginaryPart
end

"""
    BuildUnitaryMatrix(REAL::AbstractVector)

If `X` is a skew-symmetric matrix (反对称矩阵), then matrix exponential `e^X` is orthogonal. 

# Example
```julia
U = BuildUnitaryMatrix(rand(6))
U'U ≈ U*U' ≈ U*inv(U) ≈ I
```
"""
function BuildUnitaryMatrix(REAL::AbstractVector)
    H = BuildHermitianMatrix(REAL::AbstractVector)
    U = exp(-im * H)
    return U
end

"""
    BuildUnitaryMatrix(REAL::AbstractVector, IMAG::AbstractVector)

If `X` is skew-Hermitian, then `e^X` is unitary. 
`X` is skew-Hermitian if and only if `im * X` or `-im * X` is Hermitian.

# Examples
```julia
using LinearAlgebra
RL, IM = rand(6), rand(3)
U = BuildUnitaryMatrix(RL, IM)
U'U ≈ U*U' ≈ U*inv(U) ≈ I
```
"""
function BuildUnitaryMatrix(REAL::AbstractVector, IMAG::AbstractVector)
    H = BuildHermitianMatrix(REAL::AbstractVector, IMAG::AbstractVector)
    U = exp(-im * H)
    return U
end
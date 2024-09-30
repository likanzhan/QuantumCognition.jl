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
        IdS[idx] = 1I(val)
    end
    for (idx, val) in enumerate(vs)
        for v in 1:val
            IdC         = copy(IdS)
            IdC[idx]    = 1I(val) .* 1I(val)[v, :]
            MtS[idx][v] = reduce(kron, IdC)
        end
    end
    return MtS
end
BuildCompatible(vs...) = BuildCompatible([vs...]) # Splat to vector

"""
    BuildProjectors(VariableList, CompatibleIndex, UnitaryList, InCompatibleIndex)

- `VariableList = [id_1 => n_1, ..., id_k => n_k,..., id_n => n_n]`: 
    Thre are `n` variables, the `id_k`th variable has `n_k` values.
- `CompatibleIndex = [id_c1, ..., id_ck, ..., id_cn]`: 
    `k` variables are compatible with each other.
- `UnitaryList = [mt_1, ..., mt_k, ..., mt_n]`
    `n` unitary matrix used to rotate the incompatible projectors
- `InCompatibleIndex = [id_ic1 => id_c1 => u_1, ..., id_ick => id_ck => ut_k,..., id_icn => id_cn => ut_n]`: 
    Use the `ut_k`th unitory matrix to rotate the `id_ck`th projector to `id_ick`

# Example
```julia
VariableList     = [1 => 2, 2 => 3, 3 => 2, 4 => 2]
CompatibleIndex   = [1, 3] # index of the compatible variables
UnitaryList = [BuildUnitaryMatrix(rand(6)) for _ in 1:2]
InCompatibleIndex = [2 => 1 => 1, 4 => 3 => 2]

Projectors = BuildProjectors(VariableList, CompatibleIndex, UnitaryList, InCompatibleIndex)
Projectors[1][1]
```
"""
function BuildProjectors(VariableList, CompatibleIndex, UnitaryList, InCompatibleIndex)

    ## Check
    length(VariableList) == (length(CompatibleIndex) + length(InCompatibleIndex)) || error("Lengths of `CompatibleIndex+InCompatibleIndex`` and `VariableList` are Different")
    length(UnitaryList) == length(InCompatibleIndex) || error("Lengths of `UnitaryList`` and `InCompatibleIndex` are Different")
    allequal(map(size, UnitaryList)) || error("Sizes of Unitary matrix are different")

    ## Generate Compatible Projectors
    # VariableList    = sort(collect(VariableList), by = x -> first(x))
    # CompatibleIndex = sort(CompatibleIndex)
    Projectors      = Vector(undef, length(VariableList))
    CompValue       = [last(var) for var in VariableList if first(var) in CompatibleIndex]
    Projectors[CompatibleIndex] = BuildCompatible(CompValue)
    
    ## Generate InCompatible Projectors
    InCompatibleIndex = sort(collect(InCompatibleIndex), by = x -> first(x))
    first(map(size, UnitaryList)) == (reduce(*, CompValue), reduce(*, CompValue)) ||  error("Unitary matrix and Projector size not match")

    for val in InCompatibleIndex
        tar, src, uni = first(val), first(last(val)), last(last(val))
        Projectors[tar] = map(X -> (UnitaryList[uni]) * X * (UnitaryList[uni])', Projectors[src])
    end

    ## Return the values
    return Projectors
end


"""
    BuildCompatible(CompatibleVariableList::Tuple)

- Input: `(x_1 = [v1_1, ... v1_n1], x_k = [vk_1,..., vk_nk], ..., x_n = [vn_1, ..., vn_nn])` 
- Build projectors for `n` compatible variables.
    For the variable named `x_k`, it has `[vk_1, ..., vk_nk` possible responses.

"""
function BuildCompatible(CompatibleVariableList::Tuple)
    ## Create basis for each compatible variable
    DIdS = Dict{String, Matrix}()
    for vv in CompatibleVariableList
        idx, res = first(vv), last(vv)
        setindex!(DIdS, 1I(length(res)), idx)
    end

    ### Create projectors for compatible variables
    DMtS = Dict{String, Dict}()
    for vv in CompatibleVariableList
        idx, res = first(vv), last(vv)
        MT = Dict{String, Matrix}()
        for (rid, rval) in enumerate(res)
            DIdC = copy(DIdS)
            dim = length(res)
            setindex!(DIdC, (1I(dim) .* (1I(dim)[rid, :])), idx)
            setindex!(MT, reduce(kron, [last(d) for d in DIdC]), rval)
        end
        setindex!(DMtS, MT, idx)
    end

    ### Return
    return DMtS
end

"""
    BuildProjectors(CompatibleIndex, UnitaryList, InCompatibleIndex)

- `CompatibleIndex = (id_c1 => n_1, ..., id_ck => n_k, ..., id_cn => n_n)`: 
    `n` variables are compatible with each other, with the `id_ck`th variable has `n_k` possible values.
- `InCompatibleIndex = [id_ic1 => id_c1, ..., id_ick => id_ck,..., id_icn => id_cn]`:
    `n` variables incompatible with the remaing variables.
    Projector for variabel `id_ick` formed by rotate the `id_ck`th projector using unitary matrix names `id_ick`
- `UnitaryList = (id_ic1 => mt_1, ..., id_ick => mt_k, ..., id_icn => mt_n)`
    `n` unitary matrix corresponding to the `n` incompatible variables, used to rotate the specific projector

# Example
```julia
CompatibleIndex   = ("A" => ["Y", "N"], "I" => ["Y", "N"])
InCompatibleIndex = ("H" => "A", "U" => "I")
UnitaryList       = ("HA" => BuildUnitaryMatrix(rand(6)), "UI" => BuildUnitaryMatrix(rand(6)))

Projectors = BuildProjectors(CompatibleIndex, InCompatibleIndex, UnitaryList)
```
"""
function BuildProjectors(CompatibleIndex::Tuple, InCompatibleIndex::Tuple, UnitaryList::Tuple)
    Projectors = BuildCompatible(CompatibleIndex)
    for val in InCompatibleIndex
        TargetLabel, SourceLabel = first(val), last(val)
        TargetUnitary   = [last(x) for x in UnitaryList if (first(x) == TargetLabel*SourceLabel)][]
        TargetProjector = Dict(a=>TargetUnitary*b*TargetUnitary' for (a, b) in Projectors[SourceLabel])
        setindex!(Projectors, TargetProjector,  TargetLabel)
    end
    return Projectors
end
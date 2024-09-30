using DataStructures: OrderedDict
using LinearAlgebra: I
"""
    BuildCompatibleBasis(CompatibleList)

Create a dictory of bases for each compatible variable

# Example
```{julia}
vals = ["Y", "N"]
CompatibleList=("P" => vals, "B" => vals, "I" => vals, "L" => vals)
BuildCompatibleBasis(CompatibleList)
```
"""
function BuildCompatibleBasis(CompatibleList)
    CompatibleBasis = OrderedDict{String, Matrix}()
    for vv in CompatibleList
        idx, res = first(vv), last(vv)
        setindex!(CompatibleBasis, 1I(length(res)), idx)
    end
    return CompatibleBasis
end

"""
    BuildCompatibleProjectors(CompatibleList)

Create projectors for compatible variables

# Example
```{julia}
vals = ["Y", "N"]
CompatibleList=("P" => vals, "B" => vals)
BuildCompatibleProjectors(CompatibleList)
```
"""
function BuildCompatibleProjectors(CompatibleList)
    CompatibleProjectors = OrderedDict{String, OrderedDict}()
    for vv in CompatibleList
        idx, res = first(vv), last(vv)
        MT = OrderedDict{String, Matrix}()
        for (rid, rval) in enumerate(res)
            CompatibleBasis = BuildCompatibleBasis(CompatibleList)
            dim = length(res)
            setindex!(CompatibleBasis, (1I(dim) .* (1I(dim)[rid, :])), idx)
            setindex!(MT, reduce(kron, [last(d) for d in CompatibleBasis]), rval)
        end
        setindex!(CompatibleProjectors, MT, idx)
    end
    return CompatibleProjectors
end

"""
    BuildInCompatibleProjectors(CompatibleList, InCompatibleList)

Create projectors for compatible variables

# Example
```{julia}
parameters = rand(16)
res = ["Y", "N"]
CompatibleList   = ("A" => res, "I" => res)
InCompatibleList = (
    "H" => "A" => BuildUnitaryMatrix(parameters[9:11],  parameters[[12]]), 
    "U" => "I" => BuildUnitaryMatrix(parameters[13:15], parameters[[16]])
    )
BuildInCompatibleProjectors(CompatibleList, InCompatibleList)
```
"""
function BuildInCompatibleProjectors(CompatibleList, InCompatibleList)
    InCompatibleProjectors = OrderedDict{String, OrderedDict}()
    for val in InCompatibleList
        TargetLabel, SourceLabel, UnitaryMatrix = first(val), first(last(val)), last(last(val))
        vv = CompatibleList[findfirst(x -> first(x) == SourceLabel, CompatibleList)]
        idx, res = first(vv), last(vv)
        MT = OrderedDict{String, Matrix}()
        for (rid, rval) in enumerate(res)
            CompatibleBasis = BuildCompatibleBasis(CompatibleList)
            dim = length(res)
            SourceProjector = (1I(dim) .* (1I(dim)[rid, :]))
            TargetProjector = UnitaryMatrix * SourceProjector * UnitaryMatrix'
            setindex!(CompatibleBasis, TargetProjector, idx)
            setindex!(MT, reduce(kron, [last(d) for d in CompatibleBasis]), rval)
        end
        setindex!(InCompatibleProjectors, MT, TargetLabel)
    end
    return InCompatibleProjectors
end
# function BuildInCompatibleProjectors(CompatibleList, InCompatibleList)
#     InCompatibleProjectors = OrderedDict{String, OrderedDict}()
#     for val in InCompatibleList
#         TargetLabel, SourceLabel, UnitaryMatrix = first(val), first(last(val)), last(last(val))
#         vv = CompatibleList[findfirst(x -> first(x) == SourceLabel, CompatibleList)]
#         idx, res = first(vv), last(vv)
#         OriginalProjector = BuildCompatibleProjectors(CompatibleList)[idx]
#         MT = OrderedDict{String, Matrix}()
#         for (rid, rval) in enumerate(res)
#             CompatibleBasis = BuildCompatibleBasis(CompatibleList)
#             setindex!(CompatibleBasis, UnitaryMatrix, idx)
#             RotationMatrix = reduce(kron, [last(d) for d in CompatibleBasis])
#             setindex!(MT, RotationMatrix * OriginalProjector[rval] * RotationMatrix', rval)
#         end
#         setindex!(InCompatibleProjectors, MT, TargetLabel)
#     end
#     return InCompatibleProjectors
# end

"""
    BuildProjectors(CompatibleList)
    BuildProjectors(CompatibleList, InCompatibleList)

- `CompatibleList = (id_c1 => n_1, ..., id_ck => n_k, ..., id_cn => n_n)`: 
    `n` variables are compatible with each other, with the `id_ck`th variable has `n_k` possible values.
- `InCompatibleList = [id_ic1 => id_c1, ..., id_ick => id_ck,..., id_icn => id_cn]`:
    `n` variables incompatible with the remaing variables.
    Projector for variabel `id_ick` formed by rotate the `id_ck`th projector using unitary matrix names `id_ick`
- `UnitaryList = (id_ic1 => mt_1, ..., id_ick => mt_k, ..., id_icn => mt_n)`
    `n` unitary matrix corresponding to the `n` incompatible variables, used to rotate the specific projector

# Example
```julia
parameters = rand(12)
InitialState = BuildInitialState([0; parameters[1:3]], [0; parameters[4:6]])
U1 = BuildUnitaryMatrix([0; parameters[7:8]], parameters[[9]])
U2 = BuildUnitaryMatrix([0; parameters[10:11]], parameters[[12]])
res = ["Y", "N"]
CompatibleList   = ("A" => res, "I" => res)
InCompatibleList = ("H" => "A" => U1, "U" => "I" => U2)
Projectors = BuildProjectors(CompatibleList, InCompatibleList)
```
"""
BuildProjectors(CompatibleList) = BuildCompatibleProjectors(CompatibleList)
BuildProjectors(CompatibleList, InCompatibleList) = merge(BuildCompatibleProjectors(CompatibleList), BuildInCompatibleProjectors(CompatibleList, InCompatibleList))

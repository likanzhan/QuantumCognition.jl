using QuantumCognition
using Test

@testset "QuantumCognition.jl" begin

    parameters = rand(12)
    InitialState = BuildInitialState([0; parameters[1:3]], [0; parameters[4:6]])
    UH = BuildUnitaryMatrix([0; parameters[7:8]], parameters[[9]])
    UU = BuildUnitaryMatrix([0; parameters[10:11]], parameters[[12]])
    res = ["Y", "N"]
    CompatibleList   = ("A" => res, "I" => res)
    InCompatibleList = ("H" => "A" => UH, "U" => "I" => UU)

    P = BuildProjectors(CompatibleList, InCompatibleList)

    kron((1I(2) .* (1I(2)[1, :])), [1 0; 0 1]) ==  P["A"]["Y"]
    kron((1I(2) .* (1I(2)[2, :])), [1 0; 0 1]) ==  P["A"]["N"]
    kron([1 0; 0 1], (1I(2) .* (1I(2)[1, :]))) ==  P["I"]["Y"]
    kron([1 0; 0 1], (1I(2) .* (1I(2)[2, :]))) ==  P["I"]["N"]
    kron(UH *(1I(2) .* (1I(2)[1, :])) * UH', [1 0; 0 1]) ==  P["H"]["Y"]
    kron(UH *(1I(2) .* (1I(2)[2, :])) * UH', [1 0; 0 1]) ==  P["H"]["N"]
    kron([1 0; 0 1], UU *(1I(2) .* (1I(2)[1, :])) * UU') ==  P["U"]["Y"]
    kron([1 0; 0 1], UU *(1I(2) .* (1I(2)[2, :])) * UU') ==  P["U"]["N"]


    # Write your tests here.
end

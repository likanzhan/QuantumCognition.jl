module QuantumCognition

include("BuildInitialState.jl")
export BuildInitialState

include("BuildUnitaryMatrix.jl")
export BuildUnitaryMatrix

include("BuildProjectors.jl")
export BuildProjectors

include("BuildPredictions.jl")
export BuildPredictions

end # Module

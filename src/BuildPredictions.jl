"""
    BuildPredictions(Projectors, Contexts, InitialState)

# example
```{julia}
parameters = rand(16)
InitialState = BuildInitialState(parameters[1:4], parameters[5:8])
res = ["Y", "N"]
CompatibleList   = ("A" => res, "I" => res)
InCompatibleList = (
    "H" => "A" => BuildUnitaryMatrix(parameters[9:11],  parameters[[12]]), 
    "U" => "I" => BuildUnitaryMatrix(parameters[13:15], parameters[[16]])
    )
Contexts = ("A" => "H", "A" => "I", "A" => "U", "H" => "I", "H" => "U", "I" => "U", "H" => "A", "U" => "I")
Projectors = BuildProjectors(CompatibleList, InCompatibleList)
BuildPredictions(Projectors, Contexts, InitialState)
```
"""
function BuildPredictions(Projectors, Contexts, InitialState)
    Context = String[]; Responses = String[]; Probability = Float64[]
    for context in Contexts
        Projector1, Projector2 = Projectors[first(context)], Projectors[last(context)]
        Responses1, Responses2 = [first(p) for p in Projector1], [first(p) for p in Projector2]
        for res1 in Responses1, res2 in Responses2
            prj2prj1 = Projector2[res2] * Projector1[res1] * InitialState
            prob  = real(prj2prj1'prj2prj1)
            push!(Context, first(context)*last(context))
            push!(Responses, res1*res2)
            push!(Probability, prob)
        end
    end
    return (Context = Context, Responses = Responses, Probability = Probability)
end
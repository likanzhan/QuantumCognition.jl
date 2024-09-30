using LinearAlgebra: normalize

"""
    BuildInitialState(magnitude, phase)

Build a unit length complex vector

# Examples
```julia
BuildInitialState(rand(3), rand(3))
```
"""
BuildInitialState(magnitude, phase) = normalize(magnitude .+ phase * im)
BuildInitialState(magnitude) = normalize(magnitude)

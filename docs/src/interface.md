# Generic interface

```@meta
CurrentModule = FittedItemBanks
```

This page details the operations which should be supported by different
`ItemResponse` elements, as well as traits for categorisation which can be used
to dispatch to different operations.

## Basic types

```@docs
AbstractItemBank
ItemResponse
```

## AbstractItemBank traits

### Domain

```@docs
DomainType
DiscreteDomain
ContinuousDomain
VectorContinuousDomain
OneDimContinuousDomain
DiscreteIndexableDomain
DiscreteIterableDomain
```

### Response

```@docs
ResponseType
BooleanResponse
MultinomialResponse
```

## AbstractItemBank methods

```@docs
Base.length(::_DocsItemBank)
subset
subset_view
item_bank_domain
Base.eachindex(::AbstractItemBank)
item_params(::AbstractItemBank, ::Any)
```

## ItemResponse methods

```@docs
resp
resp_vec
log_resp
log_resp_vec
responses
```

## Logarithmic probabilities

```@docs
LogItemBank
```

## Testing an item bank

When implementing an `AbstractItemBank`, either in another package or for inclusion
here, use the `Test` extension to check the response interface:

```julia
using Test, FittedItemBanks

bank_tests = Base.get_extension(FittedItemBanks, :TestExt)
@testset "MyItemBank" begin
    # bank is an instance of your implementation.
    bank_tests.test_item_bank(bank, [-40.0, -0.7, 0.0, 0.9, 40.0];
        strictly_positive=true)
end
```

The helper checks metadata, category ordering, probability bounds and normalization,
and agreement between scalar, vector, and log responses. Supply valid abilities
for your bank: scalars for scalar domains, or vectors of length `domdims(bank)` for
vector domains. Include ordinary and extreme finite abilities to exercise numerical
stability.

Ordinary probability comparisons allow small absolute errors near zero, such as
those from computing a complement as `1 - p`; adjust `rtol` and `atol` if needed.
Log normalization and scalar/vector log agreement are checked separately.

Set `strictly_positive=true` only if every category has mathematically positive
probability at every supplied ability. This requires finite log probabilities even
when ordinary probabilities underflow. Otherwise, leave it at its default `false`,
which allows `-Inf` for exact zero probabilities.

For a raw `PointsItemBank`, use `bank_tests.test_points_item_bank(bank)` to check
its `item_xs`/`item_ys` interface; use `test_item_bank` for continuous smoothed
wrappers. Add model-specific tests for numerical accuracy, derivatives, and other
operations beyond the response contract. If contributing a bank here, also add a
fixture to `test/invariants.jl`.

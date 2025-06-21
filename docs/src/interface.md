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
responses
```
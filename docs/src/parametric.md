# Parametric item banks

Parametric IRT models

## High-level item banks

The high-level item banks provide shortcuts for common IRT parameterisations.

```@autodocs
Modules = [FittedItemBanks]
Pages   = ["porcelain.jl"]
```

## Composable item banks

The composable item banks allow flexible specification of item banks by
combining different blocks to build a variety of model parameterisations.

```@autodocs
Modules = [FittedItemBanks]
Pages   = ["cdf_items.jl", "cdf_mirt_items.jl", "guess_slip_items.jl", "nominal_items.jl", "monopoly.jl", "bspline.jl"]
```

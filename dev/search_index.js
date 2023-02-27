var documenterSearchIndex = {"docs":
[{"location":"parametric/#Parametric-item-banks","page":"Parametric item banks","title":"Parametric item banks","text":"","category":"section"},{"location":"parametric/","page":"Parametric item banks","title":"Parametric item banks","text":"Parametric IRT models","category":"page"},{"location":"parametric/#High-level-item-banks","page":"Parametric item banks","title":"High-level item banks","text":"","category":"section"},{"location":"parametric/","page":"Parametric item banks","title":"Parametric item banks","text":"The high-level item banks provide shortcuts for common IRT parameterisations.","category":"page"},{"location":"parametric/","page":"Parametric item banks","title":"Parametric item banks","text":"Modules = [FittedItemBanks]\nPages   = [\"porcelain.jl\"]","category":"page"},{"location":"parametric/#FittedItemBanks.ItemBank2PL-Tuple{Any, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBank2PL","text":"Convenience function to construct an item bank of the standard 2-parameter logistic single-dimensional IRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#FittedItemBanks.ItemBank3PL-Tuple{Any, Any, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBank3PL","text":"Convenience function to construct an item bank of the standard 3-parameter logistic single-dimensional IRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#FittedItemBanks.ItemBank4PL-NTuple{4, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBank4PL","text":"Convenience function to construct an item bank of the standard 4-parameter logistic single-dimensional IRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#FittedItemBanks.ItemBankMirt2PL-Tuple{Any, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBankMirt2PL","text":"Convenience function to construct an item bank of the standard 2-parameter logistic MIRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#FittedItemBanks.ItemBankMirt3PL-Tuple{Any, Any, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBankMirt3PL","text":"Convenience function to construct an item bank of the standard 3-parameter logistic MIRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#FittedItemBanks.ItemBankMirt4PL-NTuple{4, Any}","page":"Parametric item banks","title":"FittedItemBanks.ItemBankMirt4PL","text":"Convenience function to construct an item bank of the standard 4-parameter logistic MIRT model.\n\n\n\n\n\n","category":"method"},{"location":"parametric/#Composable-item-banks","page":"Parametric item banks","title":"Composable item banks","text":"","category":"section"},{"location":"parametric/","page":"Parametric item banks","title":"Parametric item banks","text":"The composable item banks allow flexible specification of item banks by combining different blocks to build a variety of model parameterisations.","category":"page"},{"location":"parametric/","page":"Parametric item banks","title":"Parametric item banks","text":"Modules = [FittedItemBanks]\nPages   = [\"cdf_items.jl\", \"cdf_mirt_items.jl\", \"guess_slip_items.jl\", \"nominal_items.jl\"]","category":"page"},{"location":"parametric/#FittedItemBanks.CdfMirtItemBank","page":"Parametric item banks","title":"FittedItemBanks.CdfMirtItemBank","text":"This item bank corresponds to the most commonly found version of MIRT in the literature. Its items feature multidimensional discriminations and its learners multidimensional abilities, but item difficulties are single-dimensional.\n\n\n\n\n\n","category":"type"},{"location":"parametric/#FittedItemBanks.PerCategoryFloat","page":"Parametric item banks","title":"FittedItemBanks.PerCategoryFloat","text":"This item bank implements the nominal model. The Graded Partial Credit Model (GPCM) is implemented in terms of this one. See:\n\nA Generalized Partial Credit Model: Application of an EM Algorithm Muraki, E., (1992). Applied Psychological Measurement 10.1177/014662169201600206\n\nAnd/or\n\nA Generalized Partial Credit Model Muraki, E. (1997).  In Handbook of Modern Item Response Theory. Springer, New York, NY. https://doi.org/10.1007/978-1-4757-2691-6_9\n\nCurrently, this item bank only supports the normal scaled logistic as the characteristic/transfer function.\n\n\n\n\n\n","category":"type"},{"location":"interface/#Generic-interface","page":"Generic interface","title":"Generic interface","text":"","category":"section"},{"location":"interface/","page":"Generic interface","title":"Generic interface","text":"This page details the operations which should be supported by different ItemResponse elements, as well as traits for categorisation which can be used to dispatch to different operations.","category":"page"},{"location":"interface/","page":"Generic interface","title":"Generic interface","text":"Modules = [FittedItemBanks]\nPages   = [\"FittedItemBanks.jl\"]","category":"page"},{"location":"interface/#FittedItemBanks.FittedItemBanks","page":"Generic interface","title":"FittedItemBanks.FittedItemBanks","text":"This module provides abstract and concrete item banks, which store information about items and their parameters such as difficulty, most typically resulting from fitting an Item-Response Theory (IRT) model.\n\n\n\n\n\n","category":"module"},{"location":"interface/#FittedItemBanks.BooleanResponse","page":"Generic interface","title":"FittedItemBanks.BooleanResponse","text":"struct BooleanResponse <: FittedItemBanks.ResponseType\n\nA boolean/dichotomous response.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.ContinuousDomain","page":"Generic interface","title":"FittedItemBanks.ContinuousDomain","text":"abstract type ContinuousDomain <: DomainType\n\nA continuous domain.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.DiscreteDomain","page":"Generic interface","title":"FittedItemBanks.DiscreteDomain","text":"abstract type DiscreteDomain <: DomainType\n\nA discrete domain. Typically this is a sampled version of a continuous domain item bank.\n\nItem response functions with discrete domains tend to support less operations than those with continuous domains.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.DiscreteIndexableDomain","page":"Generic interface","title":"FittedItemBanks.DiscreteIndexableDomain","text":"struct DiscreteIndexableDomain <: DiscreteDomain\n\nAn discrete domain which is efficiently indexable and iterable.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.DiscreteIterableDomain","page":"Generic interface","title":"FittedItemBanks.DiscreteIterableDomain","text":"struct DiscreteIterableDomain <: DiscreteDomain\n\nAn discrete domain which is only efficiently iterable.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.DomainType","page":"Generic interface","title":"FittedItemBanks.DomainType","text":"abstract type DomainType\n\nDomain type for a item banks' item response function.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.ItemResponse","page":"Generic interface","title":"FittedItemBanks.ItemResponse","text":"struct ItemResponse{ItemBankT<:AbstractItemBank}\n\nitem_bank::AbstractItemBank\nindex::Int64\n\nAn item response.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.MultinomialResponse","page":"Generic interface","title":"FittedItemBanks.MultinomialResponse","text":"struct MultinomialResponse <: FittedItemBanks.ResponseType\n\nA multinomial response, including ordinal responses.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.OneDimContinuousDomain","page":"Generic interface","title":"FittedItemBanks.OneDimContinuousDomain","text":"struct OneDimContinuousDomain <: ContinuousDomain\n\nA continuous domain that is scalar valued.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.ResponseType","page":"Generic interface","title":"FittedItemBanks.ResponseType","text":"abstract type ResponseType\n\nA response type for an item bank.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks.VectorContinuousDomain","page":"Generic interface","title":"FittedItemBanks.VectorContinuousDomain","text":"struct VectorContinuousDomain <: ContinuousDomain\n\nA continuous domain that is vector valued.\n\n\n\n\n\n","category":"type"},{"location":"interface/#FittedItemBanks._search-Union{Tuple{F}, Tuple{F, Vararg{Any, 4}}} where F","page":"Generic interface","title":"FittedItemBanks._search","text":"Binary search for the point x where f(x) = target += precis given f is assumed as monotonically increasing.\n\n\n\n\n\n","category":"method"},{"location":"interface/#FittedItemBanks.item_bank_domain-Tuple{AbstractItemBank}","page":"Generic interface","title":"FittedItemBanks.item_bank_domain","text":"Given an item bank, this function returns the domain of the item bank, i.e. the range (lo, hi) which includes for each item the range in which the the item response function is changing.\n\n\n\n\n\n","category":"method"},{"location":"non_parametric/#Non-parametric-item-banks","page":"Non-parametric item banks","title":"Non-parametric item banks","text":"","category":"section"},{"location":"non_parametric/","page":"Non-parametric item banks","title":"Non-parametric item banks","text":"Non-parametric IRT models ","category":"page"},{"location":"non_parametric/#Sampled-and-smoothed-item-banks","page":"Non-parametric item banks","title":"Sampled and smoothed item banks","text":"","category":"section"},{"location":"non_parametric/","page":"Non-parametric item banks","title":"Non-parametric item banks","text":"Modules = [FittedItemBanks]\nPages   = [\"sampled_items.jl\"]","category":"page"},{"location":"non_parametric/#FittedItemBanks.DichotomousPointsItemBank","page":"Non-parametric item banks","title":"FittedItemBanks.DichotomousPointsItemBank","text":"struct DichotomousPointsItemBank <: AbstractItemBank\n\nxs::Vector{Float64}\nys::Matrix{Float64}\n\nAn item bank where all items have IRFs computed at a fixed grid across the latent/ability dimension specified as xs. The responses are stored in ys. In most cases this item banks will be coupled with a Smoother and wrapped in a DichotomousSmoothedItemBank.\n\n\n\n\n\n","category":"type"},{"location":"non_parametric/#FittedItemBanks.KernelSmoother","page":"Non-parametric item banks","title":"FittedItemBanks.KernelSmoother","text":"struct KernelSmoother <: Smoother\n\nkernel::Function\nbandwidths::Vector{Float64}\n\nA smoother that uses a kernel to smooth the IRF. The bandwidths field stores the kernel bandwidth for each item.\n\n\n\n\n\n","category":"type"},{"location":"non_parametric/#FittedItemBanks.NearestNeighborSmoother","page":"Non-parametric item banks","title":"FittedItemBanks.NearestNeighborSmoother","text":"struct NearestNeighborSmoother <: Smoother\n\nNearest neighbor/staircase smoother.\n\n\n\n\n\n","category":"type"},{"location":"non_parametric/#FittedItemBanks.PointsItemBank","page":"Non-parametric item banks","title":"FittedItemBanks.PointsItemBank","text":"struct PointsItemBank <: AbstractItemBank\n\nxs::Vector{Float64}\nys::ArraysOfArrays.VectorOfArrays{Float64, 2, M, VT} where {M, VT<:AbstractVector{Float64}}\n\nAn item bank where all items have IRFs computed at a fixed grid across the latent/ability dimension specified as xs. The responses per-category are stored in ys. In most cases this item banks will be coupled with a Smoother and wrapped in a SmoothedItemBank.\n\n\n\n\n\n","category":"type"},{"location":"non_parametric/#FittedItemBanks.Smoother","page":"Non-parametric item banks","title":"FittedItemBanks.Smoother","text":"abstract type Smoother\n\n\n\n\n\n","category":"type"},{"location":"non_parametric/#FittedItemBanks.gauss_kern-Tuple{Any}","page":"Non-parametric item banks","title":"FittedItemBanks.gauss_kern","text":"gauss_kern(u)\n\n\nA guassian kernel for use with KernelSmoother\n\n\n\n\n\n","category":"method"},{"location":"non_parametric/#FittedItemBanks.quad_kern-Tuple{Any}","page":"Non-parametric item banks","title":"FittedItemBanks.quad_kern","text":"quad_kern(u)\n\n\nA quadratic kernel for use with KernelSmoother\n\n\n\n\n\n","category":"method"},{"location":"non_parametric/#FittedItemBanks.uni_kern-Tuple{Any}","page":"Non-parametric item banks","title":"FittedItemBanks.uni_kern","text":"uni_kern(u)\n\n\nA uniform kernel for use with KernelSmoother\n\n\n\n\n\n","category":"method"},{"location":"#FittedItemBanks.jl","page":"Home","title":"FittedItemBanks.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This module provides abstract and concrete item banks, which store information about items and their parameters such as difficulty, most typically resulting from fitting an Item-Response Theory (IRT) model.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FittedItemBanks","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"interface.md\", \"parametric.md\", \"non_parametric.md\"]\nDepth = 1","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}

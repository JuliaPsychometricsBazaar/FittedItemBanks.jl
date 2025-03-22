using StaticArrays

PerRankReal = AbstractArray{<:AbstractArray{<:Real}, 1}
PerCategoryFloat = AbstractArray{<:AbstractArray{Float64}, 1}

"""
$(TYPEDEF)

This item bank implements the nominal model. The Graded Partial Credit Model
(GPCM) is implemented in terms of this one.

Currently, this item bank only supports the normal scaled logistic as the
characteristic/transfer function.

### References:

 * [*A Generalized Partial Credit Model: Application of an EM Algorithm*,
    Muraki, E., (1992).
    Applied Psychological Measurement.
   ](https://doi.org/10.1177/014662169201600206)
 * [*A Generalized Partial Credit Model*,
    Muraki, E. (1997).
    In Handbook of Modern Item Response Theory.
   ](https://doi.org/10.1007/978-1-4757-2691-6_9)
"""
struct NominalItemBank{RankStorageT <: PerRankReal, CategoryStorageT <: PerCategoryFloat} <:
       AbstractItemBank
    ranks::RankStorageT # ak_1 ... ak_k
    discriminations::Matrix{Float64} # a_1 ... a_n
    cut_points::CategoryStorageT # d_1 ... d_k

    function NominalItemBank(ranks, discriminations, cut_points)
        if length(ranks) != length(cut_points)
            error(
                "Number of ranks should match number of cut points"
            )
        end
        if size(discriminations, 2) != length(cut_points)
            error(
                "Number of cut points " *
                "should match number of item in 2nd dimension of discriminations"
            )
        end
        new{typeof(ranks), typeof(cut_points)}(ranks, discriminations, cut_points)
    end
end

function NominalItemBank(ranks::Matrix{Float64}, discriminations::Matrix{Float64},
        cut_points::Matrix{Float64})
    NominalItemBank(nestedview(ranks), discriminations, nestedview(cut_points))
end

function NominalItemBank(ranks, discriminations::Vector{Float64}, cut_points)
    NominalItemBank(ranks, reshape(discriminations, 1, :), cut_points)
end

function GPCMItemBank(discriminations, cut_points::PerCategoryFloat)
    NominalItemBank(
        # XXX: Could probably be more efficient by making this lazy somehow
        [1:length(item_cut_points) for item_cut_points in cut_points],
        discriminations,
        cut_points
    )
end

function GPCMItemBank(discriminations, cut_points::Matrix{Float64})
    GPCMItemBank(discriminations, nestedview(cut_points))
end

DomainType(::NominalItemBank) = VectorContinuousDomain()
ResponseType(::NominalItemBank) = MultinomialResponse()

Base.length(item_bank::NominalItemBank) = size(item_bank.discriminations, 2)
domdims(item_bank::NominalItemBank) = size(item_bank.discriminations, 1)

function resp_logdensity_vec(ir::ItemResponse{<:NominalItemBank}, θ)
    aks = ir.item_bank.ranks[ir.index]
    as = @view ir.item_bank.discriminations[:, ir.index]
    ds = ir.item_bank.cut_points[ir.index]
    StaticArrays.sacollect(SVector{num_ranks(ir), Float64}, aks .* (dot(as, θ) .+ ds))
end

function num_response_categories(ir::ItemResponse{<:NominalItemBank})
    length(ir.item_bank.cut_points[ir.index])
end

function resp(ir::ItemResponse{<:NominalItemBank}, resp, θ)
    outs = exp.(resp_logdensity_vec(ir, θ))
    outs[resp] ./ sum(outs)
end

function resp_vec(ir::ItemResponse{<:NominalItemBank}, θ)
    outs = exp.(resp_logdensity_vec(ir, θ))
    outs ./ sum(outs)
end

# TODO
function item_domain(ir::ItemResponse{<:NominalItemBank}; reference_point,
        mass = default_mass, left_mass = mass, right_mass = mass)
    error("Not implemented")
    #=
    TODO:
    z = (k(a \dot θ + d)
    z_i - logsumexp z = log 0.99
    Hopefully a hyperplan in \theta
    Find closest point to reference_point
    Try for each category i
    =#
end

function num_ranks(ir::ItemResponse{<:NominalItemBank})
    length(ir.item_bank.ranks[ir.index])
end

function log_resp(ir::ItemResponse{<:NominalItemBank}, resp, θ)
    outs = resp_logdensity_vec(ir, θ)
    outs[resp] .- logsumexp(outs)
end

function log_resp_vec(ir::ItemResponse{<:NominalItemBank}, θ)
    outs = resp_logdensity_vec(ir, θ)
    outs .- logsumexp(outs)
end

function item_params(item_bank::NominalItemBank, idx)
    (; discrimination = @view item_bank.discriminations[:, idx])
end

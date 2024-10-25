# TODO: Could probably refactor to be more generic w.r.t. cdf_items.jl

using LinearAlgebra: dot

"""
This item bank corresponds to the most commonly found version of MIRT in the
literature. Its items feature multidimensional discriminations and its learners
multidimensional abilities, but item difficulties are single-dimensional.
"""
struct CdfMirtItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    difficulties::Vector{Float64}
    discriminations::Matrix{Float64}

    function CdfMirtItemBank(
            distribution::DistT,
            difficulties::Vector{Float64},
            discriminations::Matrix{Float64}
    ) where {DistT <: ContinuousUnivariateDistribution}
        if size(discriminations, 2) != length(difficulties)
            error(
                "Number of items in first (only) dimension of difficulties " *
                "should match number of item in 2nd dimension of discriminations"
            )
        end
        new{typeof(distribution)}(distribution, difficulties, discriminations)
    end
end

DomainType(::CdfMirtItemBank) = VectorContinuousDomain()
ResponseType(::CdfMirtItemBank) = BooleanResponse()

function Base.length(item_bank::CdfMirtItemBank)
    length(item_bank.difficulties)
end

function domdims(item_bank::CdfMirtItemBank)
    size(item_bank.discriminations, 1)
end

function _mirt_norm_abil(θ, difficulty, discrimination)
    dot((θ .- difficulty), discrimination)
end

function norm_abil(ir::ItemResponse{<:CdfMirtItemBank}, θ)
    _mirt_norm_abil(θ, ir.item_bank.difficulties[ir.index],
        @view ir.item_bank.discriminations[:, ir.index])
end

function resp_vec(ir::ItemResponse{<:CdfMirtItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

function item_domain(ir::ItemResponse{<:CdfMirtItemBank}, reference_point, mass = 1e-3)
    item_domain(ir, reference_point, mass, mass)
end

#=
function item_domain(ir::ItemResponse{<:CdfMirtItemBank}, reference_point, left_mass, right_mass)
    ndims = domdims(ir.item_bank)
    z_lo = quantile(ir.item_bank.distribution, left_mass)
    z_hi = quantile(ir.item_bank.distribution, 1.0 - right_mass)
    lo = fill(Inf, ndims)
    hi = fill(-Inf, ndims)
    difficulty = ir.item_bank.difficulties[ir.index]
    discrimination = @view ir.item_bank.discriminations[:, ir.index]
    diff_disc = sum(difficulty .* discrimination)
    function add_unnormed(z, i)
        # The dot of discrimination and reference point excluding the i-th element
        ref_disc_rest = sum((rp * d for (j, rp, d) in zip(1:length(reference_point), reference_point, discrimination) if j != i))
        @info "add_unnormed" z i diff_disc ref_disc_rest discrimination[i]
        unnormed = (z + diff_disc - ref_disc_rest) / discrimination[i]
        if unnormed < lo[i]
            lo[i] = unnormed
        end
        if unnormed > hi[i]
            hi[i] = unnormed
        end
    end
    for i in 1:ndims
        add_unnormed(z_lo, i)
        add_unnormed(z_hi, i)
    end
    return (lo, hi)
end
=#

function item_domain(
        ir::ItemResponse{<:CdfMirtItemBank}, reference_point, left_mass, right_mass)
    ndims = domdims(ir.item_bank)
    z_lo = quantile(ir.item_bank.distribution, left_mass)
    z_hi = quantile(ir.item_bank.distribution, 1.0 - right_mass)
    lo = fill(Inf, ndims)
    hi = fill(-Inf, ndims)
    difficulty = ir.item_bank.difficulties[ir.index]
    discrimination = @view ir.item_bank.discriminations[:, ir.index]
    diff_disc = sum(difficulty .* discrimination)
    function nearest_point(z)
        c = z + diff_disc
        t = (c - dot(reference_point, discrimination)) / sum(discrimination .^ 2)
        return reference_point .+ t .* discrimination
    end
    function update_bounds!(point)
        for i in 1:ndims
            if point[i] < lo[i]
                lo[i] = point[i]
            end
            if point[i] > hi[i]
                hi[i] = point[i]
            end
        end
    end
    update_bounds!(nearest_point(z_lo))
    update_bounds!(nearest_point(z_hi))
    return (lo, hi)
end

function resp(ir::ItemResponse{<:CdfMirtItemBank}, outcome::Bool, θ)
    if outcome
        resp(ir, θ)
    else
        cresp(ir, θ)
    end
end

function resp(ir::ItemResponse{<:CdfMirtItemBank}, θ)
    cdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function cresp(ir::ItemResponse{<:CdfMirtItemBank}, θ)
    ccdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_params(item_bank::CdfMirtItemBank, idx)
    (; difficulty = item_bank.difficulties[idx],
        discrimination = @view item_bank.discriminations[:, idx])
end

function spec_description(item_bank::CdfMirtItemBank, level)
    dim = length(item_bank.difficulties)
    if item_bank.distribution == normal_scaled_logistic
        if level == :long
            return "Two parameter $(dim)-dimensional multidimensional item bank with normal scaled logistic distribution"
        elseif level == :short
            return "2PL MIRT $(dim)d"
        else
            return "2pl_mirt_$(dim)d"
        end
    else
        if level == :long
            return "Two parameter $(dim)-dimensional multidimensional item bank with unknown transfer function"
        elseif level == :short
            return "2P MIRT $(dim)d"
        else
            return "2p_mirt_$(dim)d"
        end
    end
end

# TODO: Could probably refactor to be more generic w.r.t. cdf_items.jl

using LinearAlgebra: dot

"""
```julia
struct $(FUNCTIONNAME) <: AbstractItemBank
$(FUNCTIONNAME)(distribution, difficulties, discriminations) -> $(FUNCTIONNAME)
DomainType(::CdfMirtItemBank) = VectorContinuousDomain()
ResponseType(::CdfMirtItemBank) = BooleanResponse()
```

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

function subset(item_bank::CdfMirtItemBank, idxs)
    CdfMirtItemBank(
        item_bank.distribution,
        item_bank.difficulties[idxs],
        item_bank.discriminations[:, idxs]
    )
end
@views function subset_view(item_bank::CdfMirtItemBank, idxs)
    CdfMirtItemBank(
        item_bank.distribution,
        item_bank.difficulties[idxs],
        item_bank.discriminations[:, idxs]
    )
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

#=function item_domain(ir::ItemResponse{<:CdfMirtItemBank}; reference_point, mass = 1e-3)
    item_domain(ir, reference_point, mass, mass)
end=#

#=
function item_domain(ir::ItemResponse{<:CdfMirtItemBank}; reference_point, left_mass, right_mass)
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
        ir::ItemResponse{<:CdfMirtItemBank}; reference_point, mass = default_mass, left_mass = mass, right_mass = mass)
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

"""
```julia
struct $(FUNCTIONNAME) <: AbstractItemBank
$(FUNCTIONNAME)(distribution, difficulties, discriminations) -> $(FUNCTIONNAME)
DomainType(::SlopeInterceptMirtItemBank) = VectorContinuousDomain()
ResponseType(::SlopeInterceptMirtItemBank) = BooleanResponse()
```

This item bank corresponds to the slope-intercept version of MIRT in the
literature. Its items feature multidimensional discriminations and its learners
multidimensional abilities, but item difficulties are single-dimensional.
"""
struct SlopeInterceptMirtItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    intercepts::Vector{Float64}
    slopes::Matrix{Float64}

    function SlopeInterceptMirtItemBank(
            distribution::DistT,
            intercepts::Vector{Float64},
            slopes::Matrix{Float64}
    ) where {DistT <: ContinuousUnivariateDistribution}
        if size(slopes, 2) != length(intercepts)
            error(
                "Number of items in first (only) dimension of difficulties " *
                "should match number of item in 2nd dimension of discriminations"
            )
        end
        new{typeof(distribution)}(distribution, intercepts, slopes)
    end
end

function SlopeInterceptMirtItemBank(item_bank::CdfMirtItemBank)
    SlopeInterceptMirtItemBank(
        item_bank.distribution,
        [diff * sum(disc) for (diff, disc) in zip(item_bank.difficulties, eachcol(item_bank.discriminations))],
        item_bank.discriminations
    )
end

function CdfMirtItemBank(item_bank::SlopeInterceptMirtItemBank)
    CdfMirtItemBank(
        item_bank.distribution,
        [intercept / sum(slope) for (intercept, slope) in zip(item_bank.intercepts, eachcol(item_bank.slopes))],
        item_bank.slopes
    )
end

DomainType(::SlopeInterceptMirtItemBank) = VectorContinuousDomain()
ResponseType(::SlopeInterceptMirtItemBank) = BooleanResponse()

function Base.length(item_bank::SlopeInterceptMirtItemBank)
    length(item_bank.intercepts)
end

function subset(item_bank::SlopeInterceptMirtItemBank, idxs)
    SlopeInterceptMirtItemBank(
        item_bank.distribution,
        item_bank.intercepts[idxs],
        item_bank.slopes[:, idxs]
    )
end

@views function subset_view(item_bank::SlopeInterceptMirtItemBank, idxs)
    SlopeInterceptMirtItemBank(
        item_bank.distribution,
        item_bank.intercepts[idxs],
        item_bank.slopes[:, idxs]
    )
end

function domdims(item_bank::SlopeInterceptMirtItemBank)
    size(item_bank.slopes, 1)
end

function _mirt_norm_abil_si(θ, intercept, slope)
    dot(θ, slope)  .- intercept
end

function norm_abil(ir::ItemResponse{<:SlopeInterceptMirtItemBank}, θ)
    _mirt_norm_abil_si(θ, ir.item_bank.intercepts[ir.index],
        @view ir.item_bank.slopes[:, ir.index])
end

function resp_vec(ir::ItemResponse{<:SlopeInterceptMirtItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

#=
function item_domain(
        ir::ItemResponse{<:SlopeInterceptMirtItemBank}; reference_point, mass = default_mass, left_mass = mass, right_mass = mass)
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
=#

function resp(ir::ItemResponse{<:SlopeInterceptMirtItemBank}, outcome::Bool, θ)
    if outcome
        resp(ir, θ)
    else
        cresp(ir, θ)
    end
end

function resp(ir::ItemResponse{<:SlopeInterceptMirtItemBank}, θ)
    cdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function cresp(ir::ItemResponse{<:SlopeInterceptMirtItemBank}, θ)
    ccdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_params(item_bank::SlopeInterceptMirtItemBank, idx)
    (; intercept = item_bank.intercepts[idx],
       slop = @view item_bank.slopes[:, idx])
end

function spec_description(item_bank::Union{SlopeInterceptMirtItemBank, CdfMirtItemBank}; ppi=2, level=:long)
    dim = domdims(item_bank)
    if level != :long
        result = [
            short_spec_descriptions(ppi, item_bank.distribution),
            "mirt",
            "$(dim)d"
        ]
        return level == :short ? uppercase(join(" ", result)) : join("_", result)
    end
    params = uppercasefirst(spelled_out(ppi))
    variant = variant_description(item_bank)
    tf_desc = transfer_function_description(item_bank.distribution)
    return "$params parameter $(dim)-dimensional multidimensional item bank, $variant, with $tf_desc"
end

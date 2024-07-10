"""
$(SIGNATURES)

A guassian kernel for use with `KernelSmoother`
"""
gauss_kern(u) = exp(-u^2.0/2)

"""
$(SIGNATURES)

A uniform kernel for use with `KernelSmoother`
"""
uni_kern(u) = u >= -1 && u <= 1 ? 1.0 : 0.0

"""
$(SIGNATURES)

A quadratic kernel for use with `KernelSmoother`
"""
quad_kern(u) = u >= -1 && u <= 1 ? 1.0 - u^2 : 0.0

abstract type PointsItemBank <: AbstractItemBank end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

An item bank where all items have IRFs computed at a fixed grid across the
latent/ability dimension specified as `xs`. The responses are stored in `ys`.
In most cases this item banks will be coupled with a `Smoother` and wrapped in
a `DichotomousSmoothedItemBank`.
"""
struct DichotomousPointsItemBank{DomainT} <: PointsItemBank
    xs::DomainT
    ys::Array{Float64, 2}
end

DomainType(::DichotomousPointsItemBank) = DiscreteIndexableDomain()
domdims(::DichotomousPointsItemBank) = 0
ResponseType(::DichotomousPointsItemBank) = BooleanResponse()
function Base.length(item_bank::DichotomousPointsItemBank)
    size(item_bank.ys, 2)
end

function item_bank_xs(item_bank::DichotomousPointsItemBank)
    item_bank.xs
end

function item_domain(ir::ItemResponse{<:DichotomousPointsItemBank})
   (ir.item_bank.xs[1], ir.item_bank.xs[end])
end

function item_xs(ir::ItemResponse{<:DichotomousPointsItemBank})
    ir.item_bank.xs
end

function item_ys(ir::ItemResponse{<:DichotomousPointsItemBank})
    @view ir.item_bank.ys[:, ir.index]
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

An item bank where all items each IRF has been computed on a potentially distrinct
grid across the latent/ability dimension specified as `xs`. The responses are stored
in `ys`. In most cases this item banks will be coupled with a `Smoother` and wrapped in
a `DichotomousSmoothedItemBank`.
"""
struct MultiGridDichotomousPointsItemBank <: PointsItemBank
    xs::VectorOfVectors{Float64}
    ys::VectorOfVectors{Float64}
end

DomainType(::MultiGridDichotomousPointsItemBank) = DiscreteIndexableDomain()
domdims(::MultiGridDichotomousPointsItemBank) = 0
ResponseType(::MultiGridDichotomousPointsItemBank) = BooleanResponse()
function Base.length(item_bank::MultiGridDichotomousPointsItemBank)
    length(item_bank.ys)
end

function item_domain(ir::ItemResponse{<:MultiGridDichotomousPointsItemBank})
   (ir.item_bank.xs[ir.index][1], ir.item_bank.xs[ir.index][end])
end

function item_xs(ir::ItemResponse{<:MultiGridDichotomousPointsItemBank})
    ir.item_bank.xs[ir.index]
end

function item_ys(ir::ItemResponse{<:MultiGridDichotomousPointsItemBank})
    ir.item_bank.ys[ir.index]
end

function item_ys(ir::ItemResponse{<:PointsItemBank}, outcome::Bool)
    if outcome
        return item_ys(ir)
    else
        return 1.0 .- item_ys(ir)
    end
end

"""
$(TYPEDEF)
"""
abstract type Smoother end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

A smoother that uses a kernel to smooth the IRF. The `bandwidths` field stores
the kernel bandwidth for each item.
"""
struct KernelSmoother <: Smoother
    kernel::Function
    bandwidths::Vector{Float64}
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Nearest neighbor/staircase smoother.
"""
struct NearestNeighborSmoother <: Smoother end

struct DichotomousSmoothedItemBank{P <: PointsItemBank, S <: Smoother} <: AbstractItemBank
    inner_bank::P
    smoother::S
end

DomainType(::DichotomousSmoothedItemBank) = OneDimContinuousDomain()
ResponseType(::DichotomousSmoothedItemBank) = BooleanResponse()
inner_item_response(ir::ItemResponse{<: DichotomousSmoothedItemBank}) = ItemResponse(ir.item_bank.inner_bank, ir.index)

function Base.length(item_bank::DichotomousSmoothedItemBank)
    length(item_bank.inner_bank)
end

function item_domain(ir::ItemResponse{<:DichotomousSmoothedItemBank})
   item_domain(ItemResponse(ir.item_bank.inner_bank, ir.index))
end

function resp_vec(ir::ItemResponse{<:DichotomousSmoothedItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank}, outcome::Bool, θ)
    r = resp(ir, θ)
    if outcome
        r
    else
        1.0 - r
    end
end

function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank{<:PointsItemBank, <:KernelSmoother}}, θ)
    # XXX: Avoid allocating here? @turbo here?
    inner_ir = inner_item_response(ir)
    kernel_comb = ir.item_bank.smoother.kernel.((item_xs(inner_ir) .- θ) ./ ir.item_bank.smoother.bandwidths[ir.index])
    sum(kernel_comb .* (item_ys(inner_ir))) / sum(kernel_comb)
end

function nearest_index(xs, ys, θ)
    neighbor_idx = searchsortedfirst(xs, θ)
    if (
        neighbor_idx != 1 &&
        (
            (neighbor_idx == length(xs) + 1) ||
            ((θ - xs[neighbor_idx - 1]) < (xs[neighbor_idx] - θ))
        )
    )
        neighbor_idx = neighbor_idx - 1
    end
    return neighbor_idx
end

function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank{<:PointsItemBank, <:NearestNeighborSmoother}}, θ)
    inner_ir = inner_item_response(ir)
    xs = item_xs(inner_ir)
    ys = item_ys(inner_ir)
    ys[nearest_index(xs, ys, θ)]
end

function nearest_indices(xs, ys, thetas)
    pivot_theta_idx = length(thetas) ÷ 2
    pivot_theta = thetas[pivot_theta_idx]
    pivot_xs_idx = nearest_index(xs, ys, pivot_theta)
    nearest_indices((@view xs[1:pivot_xs_idx]), (@view ys[1:pivot_xs_idx]), (@view thetas[1:(pivot_theta - 1)]))
end

#=
TODO: Implement nested binary search for multiple theta points
function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank{<:PointsItemBank, <:NearestNeighborSmoother}}, thetas::AbstractVector)
    inner_ir = inner_item_response(ir)
    xs = item_xs(inner_ir)
    ys = item_ys(inner_ir)
    return nearest_index(xs, ys, thetas)
end
=#

"""
$(SIGNATURES)

Converts a dichotomous item bank `item_bank` into a gridded item bank by evaluating the items at points `xs`.
"""
function gridify(item_bank, xs)::DichotomousPointsItemBank
    ys = Array{Float64}(undef, (length(xs), length(item_bank)))
    for item_idx in 1:length(item_bank)
        ys[:, item_idx] .= interp(xs, resp.(Ref(ItemResponse(item_bank, item_idx)), xs))
    end
    return DichotomousPointsItemBank(xs, ys)
end
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

"""
$(TYPEDEF)
$(TYPEDFIELDS)

An item bank where all items have IRFs computed at a fixed grid across the
latent/ability dimension specified as `xs`. The responses are stored in `ys`.
In most cases this item banks will be coupled with a `Smoother` and wrapped in
a `DichotomousSmoothedItemBank`.
"""
struct DichotomousPointsItemBank <: AbstractItemBank
    xs::Vector{Float64}
    ys::Array{Float64, 2}
end

ResponseType(::DichotomousPointsItemBank) = BooleanResponse()
function Base.length(item_bank::DichotomousPointsItemBank)
    size(item_bank.ys, 1)
end

function item_domain(ir::ItemResponse{<:DichotomousPointsItemBank})
   (ir.item_bank.xs[1], ir.item_bank.xs[end])
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

struct DichotomousSmoothedItemBank{S <: Smoother} <: AbstractItemBank
    inner_bank::DichotomousPointsItemBank
    smoother::S
end

DomainType(::DichotomousSmoothedItemBank) = OneDimContinuousDomain()
ResponseType(::DichotomousSmoothedItemBank) = BooleanResponse()

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

function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank{<:KernelSmoother}}, θ)
    # XXX: Avoid allocating here? @turbo here?
    kernel_comb = ir.item_bank.smoother.kernel.((ir.item_bank.inner_bank.xs .- θ) ./ ir.item_bank.smoother.bandwidths[ir.index])
    sum(kernel_comb .* (@view ir.item_bank.inner_bank.ys[ir.index, :])) / sum(kernel_comb)
end

function resp(ir::ItemResponse{<:DichotomousSmoothedItemBank{<:NearestNeighborSmoother}}, θ)
    neighbor_idx = searchsortedfirst(ir.item_bank.inner_bank.xs, θ)
    if (
        neighbor_idx != 1 &&
        (
            neighbor_idx == length(ir.item_bank.inner_bank.xs) ||
            (θ - ir.item_bank.inner_bank.xs[neighbor_idx - 1]) < (ir.item_bank.inner_bank.ys[found] - θ)
        )
    )
        neighbor_idx = neighbor_idx - 1
    end
    ir.inner_bank.inner_bank.ys[ir.index, neighbor_idx]
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

An item bank where all items have IRFs computed at a fixed grid across the
latent/ability dimension specified as `xs`. The responses per-category are
stored in `ys`. In most cases this item banks will be coupled with a `Smoother`
and wrapped in a `SmoothedItemBank`.
"""
struct PointsItemBank <: AbstractItemBank
    xs::Vector{Float64}
    ys::VectorOfArrays{Float64, 2}
end
gauss_kern(u) = exp(-u^2.0/2)
uni_kern(u) = u >= -1 && u <= 1 ? 1.0 : 0.0
quad_kern(u) = u >= -1 && u <= 1 ? 1.0 - u^2 : 0.0

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

abstract type Smoother end

struct KernelSmoother <: Smoother
    kernel::Function
    bandwidths::Vector{Float64}
end

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

struct PointsItemBank <: AbstractItemBank
    xs::Vector{Float64}
    ys::VectorOfArrays{Float64, 2}
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

end
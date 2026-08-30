module DummyData

using Distributions: Normal, MvNormal, Zeros, ScalMat
using Random
using ArraysOfArrays: VectorOfArrays, VectorOfVectors
using BSplines

import ..SimpleItemBankSpec, ..StdModelForm, ..StdModel2PL, ..StdModel3PL, ..StdModel4PL
import ..OneDimContinuousDomain, ..VectorContinuousDomain, ..BooleanResponse,
       ..MultinomialResponse, ..ItemBank
import ..GuessItemBank, ..GuessAndSlipItemBank
import ..ItemResponse, ..resp
import ..MonopolyItemBank, ..BSplineItemBank
import ..GPCMItemBank, ..NominalItemBank
import ..DichotomousPointsItemBank, ..DichotomousSmoothedItemBank, ..KernelSmoother
import ..gauss_kern, ..gridify
import ..FittedItemBanks: FittedItemBanks

export dummy_item_bank, convert_dummy_item_bank, dummy_full

const default_num_questions = 8000
const default_num_testees = 30
const std_normal = Normal()

std_mv_normal(dim) = MvNormal(Zeros(dim), ScalMat(dim, 1.0))

abs_rand(rng, dist, dims...) = abs.(rand(rng, dist, dims...) .- 0.1) .+ 0.1
clamp_rand(rng, dist, dims...) = clamp.(rand(rng, dist, dims...), 0.0, 0.4)

function dummy_item_bank(type::Type, args...; kwargs...)
    dummy_item_bank(Random.default_rng(), type, args...; kwargs...)
end

include("./transfer_based.jl")
include("./flexible.jl")
include("./nominal.jl")

end

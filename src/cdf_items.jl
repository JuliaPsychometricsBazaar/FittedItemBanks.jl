struct TransferItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    difficulties::Vector{Float64}
    discriminations::Vector{Float64}
end

DomainType(::TransferItemBank) = OneDimContinuousDomain()
ResponseType(::TransferItemBank) = BooleanResponse()

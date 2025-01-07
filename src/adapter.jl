struct OneDimensionItemBankAdapter{ItemBankT <: AbstractItemBank} <: AbstractItemBank
    inner_bank::ItemBankT
    
    function OneDimensionItemBankAdapter(inner)
        if DomainType(inner) != VectorContinuousDomain()
            error("Inner item bank must have a vector continuous domain")
        end
        new{typeof(inner)}(inner)
    end
end

function inner_item_response(ir::ItemResponse{<:OneDimensionItemBankAdapter})
    ItemResponse(ir.item_bank.inner_bank, ir.index)
end

DomainType(::OneDimensionItemBankAdapter) = OneDimContinuousDomain()

@forward OneDimensionItemBankAdapter.inner_bank Base.length, subset, ResponseType
function subset(item_bank::OneDimensionItemBankAdapter, idxs)
    OneDimensionItemBankAdapter(subset(item_bank.inner_bank, idxs))
end

domdims(item_bank::OneDimensionItemBankAdapter) = 0

function resp(ir::ItemResponse{<:OneDimensionItemBankAdapter}, r, θ)
    resp(inner_item_response(ir), r, SA[θ])
end

function resp(ir::ItemResponse{<:OneDimensionItemBankAdapter}, θ)
    resp(inner_item_response(ir), SA[θ])
end

function resp_vec(ir::ItemResponse{<:OneDimensionItemBankAdapter}, θ)
    resp_vec(inner_item_response(ir), SA[θ])
end

function num_response_categories(ir::ItemResponse{<:OneDimensionItemBankAdapter})
    num_response_categories(inner_item_response(ir))
end
function power_summary(io::IO, obj::AbstractItemBank; kwargs...)
    if applicable(spec_description, obj)
        spec = spec_description(obj)
        println(io, spec)
        indent_io = indent(io, 2)
        nitems = length(obj)
        println(indent_io, "Number of items: $nitems")
        dt = DomainType(obj)
        if dt isa OneDimContinuousDomain
            domtype = "One-dimensional continuous"
        elseif dt isa VectorContinuousDomain
            domtype = "Multi-dimensional continuous"
        elseif dt isa DiscreteIndexableDomain
            domtype = "Discrete, indexable"
        elseif dt isa DiscreteIterableDomain
            domtype = "Discrete, iterable"
        else
            domtype = "Unknown domain type"
        end
        println(indent_io, "Domain type: $domtype")
        if dt isa VectorContinuousDomain
            ndims_desc = "$ndims"
            println(indent_io, "Domain dimensions: $ndims_desc")
        end
        rt = ResponseType(obj)
        if rt isa BooleanResponse
            restype = "Boolean/dichotomous"
        elseif rt isa MultinomialResponse
            restype = "Multinomial"
        else
            restype = "Unknown response type"
        end
        println(indent_io, "Response type: $restype")
        if rt isa MultinomialResponse
            nresp = num_response_categories(obj)
            println(indent_io, "Number of response categories: $nresp")
        end
    else
        # Fallback to show(...)
        invoke(show, Tuple{IO, MIME"text/plain", Any}, io, MIME("text/plain"), obj)
    end
end

function transfer_function_description(distribution::Distribution)
    if distribution == std_logistic
        return "standard logistic transfer function"
    elseif distribution == normal_scaled_logistic
        return "normal scaled logistic transfer function"
    else
        return "transfer function based on " * power_summary(distribution)
    end
end

function short_spec_descriptions(ppi, distribution)
    result = "$(ppi)p"
    if item_bank.distribution == std_logistic
        result *= "l"
    elseif item_bank.distribution == normal_scaled_logistic
        result *= "ln"
    end
    return result
end

variant_description(::Union{TransferItemBank, CdfMirtItemBank}) = "classical parameterization"
variant_description(::Union{SlopeInterceptTransferItemBank, SlopeInterceptMirtItemBank}) = "slope-intercept parameterization"

function spec_description(item_bank::Union{TransferItemBank, SlopeInterceptTransferItemBank}; level=:long, ppi=2)
    if level != :long
        result = short_spec_descriptions(ppi, item_bank.distribution)
        return level == :short ? uppercase(result) : result
    end
    params = uppercasefirst(spelled_out(ppi))
    variant = variant_description(item_bank)
    tf_desc = transfer_function_description(item_bank.distribution)
    return "$params parameter unidimensional item bank, $variant, with $tf_desc"
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

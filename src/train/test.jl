function loadweights(statstore::SNNJulia.Datasource, epoch::Int64, update::Function, params::SNNJulia.params)
    weights = SNNJulia.datasrcread(statstore, "weights$epoch")
    #weights = clampweights(weights, .4, .2)
    numspikes = SNNJulia.datasrcread(statstore, "numspikes$epoch")
    homeostasis = SNNJulia.datasrcread(statstore, "homeostasis$epoch")
    numneuron, numinp = size(weights)
    neurons = [SNNJulia.Neuron(params.prest, params.potdep, numspikes[i], homeostasis[i]) for i in 1:numneuron]
    layer = SNNJulia.SNNLayer(params, weights, neurons, update)
    return layer
end

function clampweights(w, thresh, val)
    hi = maximum(w)
    for i in 1:size(w)[1] * size(w)[2]
        w[i] = w[i] > thresh ? hi : val
    end
    w
end

function testsingle(layer::SNNJulia.SNNLayer, dataset::SNNJulia.Dataset, assignment::Array{Int64, 1})
    numclass = length(dataset.classes)
    con = zeros(Int64, numclass, numclass)
    for cls in dataset.classes
        for i in 1:length(dataset.data[cls])
            inp = SNNJulia.getitem(dataset, cls, i)
            SNNJulia.layerforward(layer, inp, false)
            win = getwinningneuron(layer)
            guess = assignment[win]
            con[clsind(cls, dataset.classes), guess] += 1
        end
    end
    return con
end

function assignneurons(layer::SNNJulia.SNNLayer, dataset::SNNJulia.Dataset)
    numclass = length(dataset.classes)
    count = zeros(Int64, numclass, layer.numneuron)
    for cls in dataset.classes
        for i in 1:length(dataset.data[cls])
            inp = SNNJulia.getitem(dataset, cls, i)
            SNNJulia.layerforward(layer, inp, false)
            win = getwinningneuron(layer)
            count[clsind(cls, dataset.classes), win] += 1
        end
    end

    assign = zeros(Int64, layer.numneuron)
    for i in 1:layer.numneuron
        _, assign[i] = findmax(count[:, i])
    end
    return assign
end

function getwinningneuron(layer::SNNJulia.SNNLayer)
    spikes = layer.trainvars.spikeVSt
    spikecount = sum(spikes, dims=2)
    _, win = findmax(spikecount)
    return win
end

function clsind(cls::String, classes::Array{String, 1})
    findfirst(c -> c == cls, classes)
end


struct ClassAssignment
    activityfunc::Function
    activity::Dict{String, Vector}
    popid::Int
    numneurons::Int
    numclasses::Int

    function ClassAssignment(activityfunc, activity, popid)
        numneurons = first(length.(values(activity)))
        numclasses = length(keys(activity))
        new(activityfunc, activity, popid, numneurons, numclasses)
    end
end

"""
    classify(
        cortex::Cortex,
        popid::Int,
        trainloader::Dataloader,
        testloader::Dataloader,
        act::Function)

This function has not been exported.
Produces a `confusion` matrix (2D `Array`) and an accuracy `Vector` based on the classes specified in `trainloader` and `testloader`. `popid` is the ID of the population being used to classify the samples.
`act` is an activity function, used to determine the most active neuron. Its function signature should be like that of `mostspikes!`.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inp_neuron_types = [(PoissonInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, LIFSpike);

julia> traindset = Dataset("data/mnist.jld2", "train"; activecls=["0", "1"]); trainloader = Dataloader(traindset, true);

julia> valdset = Dataset("data/mnist.jld2", "val"; activecls=["0", "1"]); valloader = Dataloader(valdset, false);

julia> resizeprop!(traindset, 2); resizeprop!(valdset, 2);

julia> for (cls, sample) in trainloader
           process_sample!(cortex, sample)
       end

julia> conf, acc = Thorn.classify(cortex, 2, trainloader, valloader, Thorn.mostspikes!);

julia> println(conf); println(acc)
[11.0 13.0; 4.0 23.0]
0.6666666666666666
```
"""
function classify(
        cortex::Cortex,
        popid::Int,
        trainloader::Dataloader,
        testloader::Dataloader,
        act::Function)
    activity = probeactivity(act, cortex, trainloader, popid)
    assign = ClassAssignment(act, activity, popid)
    result = classify(cortex, assign, testloader)
    # Build confusion matrix and compute accuracy
    classes = trainloader.dataset.classes
    confusion = zeros(assign.numclasses, assign.numclasses)
    for (expected, actual) in result
        ex = findfirst(isequal(expected), classes)
        ac = findfirst(isequal(actual), classes)
        confusion[ex, ac] += 1
    end
    accuracy = sum(isequal.(first.(result), last.(result))) / length(result)
    confusion, accuracy
end

# loader must contain test data
function classify(cortex::Cortex, assign::ClassAssignment, loader::Dataloader)
    result = Tuple{String, String}[]
    for (cls, data) in loader
        # pass into a cortex
        process_sample!(cortex, data; train=false)
        # extract needed spikes
        spikes = copy(cortex.populations[assign.popid].out_spikes.items)
        # pass into classify
        # put the result into an array
        push!(result, (cls, classify(spikes, assign)))
    end
    result
end

function classify(spikes::Array{T, 1}, assign::ClassAssignment) where T<:Spike
    # This function will take in an array of spikes and a ClassAssignment
    # output the name of the class
    activity = zeros(assign.numneurons)
    assign.activityfunc(activity, spikes, assign.numneurons)
    decide(activity, assign.activity)
end

# I assume the assign function for clustering is very similar to this API
# loader must contain train data
function probeactivity(
        activityfunc::Function, cortex::Cortex, loader::Dataloader, popid::Int)
    numneurons = cortex.populations[popid].length
    # For each class, we keep track of how "active" each neuron is
    # (can be number of spikes, number of spikes within a window, time of first spike)
    activity = Dict(cls=>zeros(numneurons) for cls in loader.dataset.classes)
    # Iterate over examples by class
    lens = Dict(cls=>0 for (cls, d) in loader)
    for (cls, _) in loader
        lens[cls] += 1
    end

    for (cls, len) in lens
        for i in 1:len
            inp = loader.dataset[cls, i]
            process_sample!(cortex, inp)
            spikes = copy(cortex.populations[popid].out_spikes.items)
            # A function determines how active each neuron was for the example
            # In this case it only affects the neuron that had the most spikes
            # activityfunc must modify the activity[cls] array given to it
            activityfunc(activity[cls], spikes, numneurons)
        end
    end
    activity
end

# example of activityfunc
"""
    mostspikes!(
        activity::Array{T, 1},
        spikes::Array{S, 1},
        numneurons::Int) where {T, S<:Spike}

An activity function used to determine the most active neuron. `activity` is an array of the same size as the number of neurons in the population.
`spikes` is the train of spikes produced by the population of interest.
This function modifies `activity` by adding 1 to the i-th element where i is the index of the neuron that produced the most number of spikes in `spikes`.
Use this as input to `classify`.
"""
function mostspikes!(
        activity::Array{T, 1},
        spikes::Array{S, 1},
        numneurons::Int) where {T, S<:Spike}
    i = last(findmax(countspikes(spikes, numneurons)))
    activity[i] += 1
end

countspikes(spikes, numneurons) = [count(s->s.neuron_id == i, spikes) for i in 1:numneurons]

function decide(
        activity::Vector{S}, probedactivity::Dict{String, Vector{T} where T}) where S
    # Each neuron is mapped to the class that it was most active for
    # using a decision function
    ks, vs = keys(probedactivity), values(probedactivity)
    # neuronactivities is a matrix of n x cls
    neuronactivities = hcat(vs...)
    inds = [argmax(row) for row in eachrow(neuronactivities)]
    assignmentvec = collect(ks)[inds]
    assignmentvec[argmax(activity)]
end


function test(cortex::Cortex, trainloader::Dataloader, testloader::Dataloader, act, dec, popid::Int)
    activity = probeactivity(act, cortex, trainloader, popid)
    assign = ClassAssignment(funcs..., activity, popid)
    result = classifyall(cortex, assign, testloader)
    # Build confusion matrix and compute accuracy
    classes = trainloader.dataset.classes
    confusion = zeros(assign.numclasses, assign.numclasses)
    for (expected, actual) in result
        ex = classes[findfirst(expected)]
        ac = classes[findfirst(actual)]
        confusion[ex, ac] += 1
    end
    accuracy = sum(map.(isequal.(result), last.(result))) / length(result)
    confusion, accuracy
end

# loader must contain test data
function classifyall(cortex::Cortex, assign::ClassAssignment, loader::Dataloader)
    result = Tuple{String, String}[]
    for (cls, data) in loader
        # pass into a cortex
        pop_spikes, _ = process_sample!(cortex, data)
        # extract needed spikes
        spikes = filter(x->first(x) == assign.popid, pop_spikes)
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
    assign.activityfunc(activity, spikes)
    assign.decisionfunc(activity, assign.activity)
end

struct ClassAssignment
    activityfunc::Function
    decisionfunc::Function
    activity::Dict{String, Vector}
    popid::Int
    numneurons::Int
    numclasses::Int

    function ClassAssignment(activityfunc, decisionfunc, activity, popid)
        numneurons = first(length.(values(activity)))
        numclasses = length(keys(activity))
        new(activityfunc, decisionfunc, activity, popid, numneurons, numclasses)
    end
end

# I assume the assign function for clustering is very similar to this API
# loader must contain train data
function probeactivity(activityfunc::Function, cortex::Cortex, loader::Dataloader, popid::Int)
    numneurons = length(cortex.populations[popid].neurons)
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
            inp = loader[cls, i]
            pop_spikes, _ = process_sample!(cortex, inp)
            spikes = last.(filter(x->first(x) == popid, pop_spikes))
            # A function determines how active each neuron was for the example
            # In this case it only affects the neuron that had the most spikes
            # activityfunc must modify the activity[cls] array given to it
            activityfunc(activity[cls], spikes)
            #mostspike = neuronmostspikes(spikes, numneurons)
            #activity[cls][mostspike] += 1
        end
    end
    activity
end

# example of activityfunc
function mostspikes(activity::Array{Any, 1}, spikes::Array{T, 1}) where T<:Spike
    i = last(findmax(countspikes(spikes, numneurons)))
    activity[i] += 1
end

countspikes(spikes, numneurons) = [count(s->s.neuron_index == i, spikes) for i in numneurons]

# example of decisionfunc
function decidemostspikes(activity::Array{Any, 1}, probedactivity::Dict{String, Array{Any, 1}})
# Each neuron is mapped to the class that it was most active for using a decision function
    ks, vs = keys(probedactivity), values(probedactivity)
    #neuronactivities is a matrix of n x cls
    neuronactivities = hcat(vs...)
    assignmentvec = ks[argmax(neuronactivities[i, :]) for i in 1:length(activity)]
    assignmentvec[argmax(activity)]
end


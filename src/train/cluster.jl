function cluster(cortex::Cortex, popid::Int, loader::Dataloader, numclusters::Int, act, clusteralg, dec)
    numclusters < 2 && error("Unable to cluster into fewer than 2 clusters.")
    #TODO measure activity of each neuron in response to each sample
    #This becomes an n-dimensional clustering problem. Just run K-means or whatever on the result of the activity
    #Then run classification on the neurons with each cluster as the classes
end


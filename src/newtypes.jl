macro newneuron()
    ex = :($(Expr(:toplevel, :(import SpikingNN.state_update!), :(import SpikingNN.output_spike!))))
end

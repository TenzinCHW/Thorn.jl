macro newprocessingneuron()
    ex = :($(Expr(:toplevel, :(import SpikingNN.state_update!), :(import SpikingNN.output_spike!), :(import SpikingNN.reset!))))
end

macro newinputneuron()
    ex = :($(Expr(:toplevel, :(import SpikingNN.generate_input))))
end

macro newprocpop()
    ex = :($(Expr(:toplevel, :(import SpikingNN.process_spike!))))
end

macro newinppop()
    ex = :($(Expr(:toplevel, :(import SpikingNN.generate_input_spikes!))))
end

macro newcortex()
    ex = :($(Expr(:toplevel, :(import SpikingNN.process_sample!))))
end


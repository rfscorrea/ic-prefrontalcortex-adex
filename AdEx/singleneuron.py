from brian2 import *

class SingleNeuron():

    def __init__(self, params, I=0, V0=False, w0=0, method='rk4', time_step=0.05):

        self.net = Network()
        self.time_step= time_step

        self.params = params
        self.I = I

        membrane_eq = """
        
        
        dV/dt = ( -(V-V_r + delta_L exp((V-V_L)/delta_L)) - R*u + R*I )/tau_m : amp/second
        du/dt = (a*(V-V_r)-u)/tau_u : amp/second
        
        V : volt
        V_r : volt
        V_th : volt
        delta_L : volt
        u : amp
        I : amp
        b : amp
        a : siemens
        R : ohm
        
        """

        membrane_threshold = 'V > V_th'
        membrane_reset = 'V = V_r; u = u + b'

        self.group = NeuronGroup(1, model=membrane_eq, threshold=membrane_threshold, reset=membrane_reset, method=method, refractory=5*ms, dt=self.time_step*ms)
        self.spikem = SpikeMonitor(self.group)
        self.neurom = StateMonitor(self.group, True, True)
        self.net.add(self.group, self.spikem, self.neurom)

    def run(self, t):

        self.net.run(t, report='text', report_period=10*second)

    def plot(self):

        plot(self.neurom.t/ms. self.neurom.V[0]/mV)
        tspikes = self.spikem.t/ms
        for t in tspikes:
            vlines(t, self.params[4], 20)
        show()



params = [0 for i in range(9)]
I = 0

params[0] = 1
params[1] = 1
params[2] = 1
params[3] = 1
params[4] = 1
params[5] = 1
params[6] = 1
params[7] = 1
params[8] = 1

neuron = SingleNeuron(params, I)
neuron.run(5000*ms)
neuron.plot()
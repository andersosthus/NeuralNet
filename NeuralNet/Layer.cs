using System.Collections.Generic;

namespace NeuralNet
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public string Name { get; set; }

        public Layer()
        {
            Neurons = new List<Neuron>();
        }

        public void AddNeuron(Neuron neuron)
        {
            Neurons.Add(neuron);
        }
    }
}
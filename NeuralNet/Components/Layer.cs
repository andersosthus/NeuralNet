using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet.Components
{
    [Serializable]
    public class Layer
    {
        public Layer()
        {
            Neurons = new List<Neuron>();
        }

        public List<Neuron> Neurons { get; }
        public string Name { get; set; }

        public void AddNeuron(Neuron neuron)
        {
            Neurons.Add(neuron);
        }

        public List<double> GetWeights()
        {
            return Neurons.SelectMany(x => x.GetWeights()).ToList();
        }
    }
}
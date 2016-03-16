using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural
{
    [Serializable]
    public class Net
    {
        private double _error;

        public Net(IReadOnlyList<int> topology)
        {
            var layers = new List<Layer>();
            var numberOfLayers = topology.Count;
            for (var layerNumber = 0; layerNumber < numberOfLayers; layerNumber++)
            {
                var layer = new Layer();

                if (layerNumber == 0)
                    layer.Name = "Input";
                else if (layerNumber == numberOfLayers - 1)
                    layer.Name = "Output";
                else
                    layer.Name = "Hidden" + (layerNumber + 1);

                var numberOfOutputs = (layerNumber == numberOfLayers - 1) ? 0 : topology[layerNumber + 1];

                for (var neuronNumber = 0; neuronNumber <= topology[layerNumber]; neuronNumber++)
                {
                    layer.AddNeuron(new Neuron(numberOfOutputs, neuronNumber));
                }

                // Set the output of the bias neutron to 1.0
                layer.Neurons.Last().OutputValue = 1.0;

                layers.Add(layer);
            }

            Layers = layers;
        }

        public double RecentAverageError { get; private set; }
        public List<Layer> Layers { get; }

        public void FeedForward(List<double> inputValues)
        {
            if (inputValues.Count != Layers[0].Neurons.Count - 1)
                throw new ArgumentOutOfRangeException(nameof(inputValues));

            for (var i = 0; i < inputValues.Count; i++)
            {
                Layers[0].Neurons[i].OutputValue = inputValues[i];
            }

            for (var layerNumber = 1; layerNumber < Layers.Count; layerNumber++)
            {
                var previousLayer = Layers[layerNumber - 1];
                for (var neuron = 0; neuron < Layers[layerNumber].Neurons.Count - 1; neuron++)
                {
                    Layers[layerNumber].Neurons[neuron].FeedForward(previousLayer);
                }
            }
        }

        public List<double> GetWeights()
        {
            return Layers.SelectMany(x => x.GetWeights()).ToList();
        }

        public void BackProp(List<double> targetValues)
        {
            // Calculate overall net error (RMS of output neuron errors)
            var outputLayer = Layers.Last();
            _error = 0.0;

            for (var n = 0; n < outputLayer.Neurons.Count - 1; n++)
            {
                var delta = targetValues[n] - outputLayer.Neurons[n].OutputValue;
                _error += delta*delta;
            }
            _error /= outputLayer.Neurons.Count - 1; // Get average error squared
            _error = Math.Sqrt(_error); // RMS

            RecentAverageError = (RecentAverageError*Constants.RecentAverageSmoothingFactor + _error)/
                                 (Constants.RecentAverageSmoothingFactor + 1.0);

            // Calculate output layer gradients
            for (var n = 0; n < outputLayer.Neurons.Count - 1; n++)
            {
                outputLayer.Neurons[n].CalculateOutputGradients(targetValues[n]);
            }

            // Calculate gradients on hidden layers
            for (var layerNumber = Layers.Count - 2; layerNumber > 0; layerNumber--)
            {
                var hiddenLayer = Layers[layerNumber];
                var nextLayer = Layers[layerNumber + 1];

                for (var n = 0; n < hiddenLayer.Neurons.Count; ++n)
                {
                    hiddenLayer.Neurons[n].CalculateHiddenGradients(nextLayer);
                }
            }

            // For all layers from outputs to first hidden layer
            // update connection weights
            for (var layerNumber = Layers.Count - 1; layerNumber > 0; layerNumber--)
            {
                var layer = Layers[layerNumber];
                var previousLayer = Layers[layerNumber - 1];

                for (var n = 0; n < layer.Neurons.Count - 1; n++)
                {
                    layer.Neurons[n].UpdateInputWeights(previousLayer);
                }
            }
        }

        public List<double> GetResults()
        {
            var results = new List<double>();

            var lastLayer = Layers.Last();
            for (var n = 0; n < lastLayer.Neurons.Count - 1; ++n)
            {
                results.Add(lastLayer.Neurons[n].OutputValue);
            }

            return results;
        }
    }
}
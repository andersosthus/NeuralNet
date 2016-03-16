using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    public class Net
    {
        private readonly List<double> cheat = new List<double>
        {
            0.628224,
            0.464278,
            0.826289,
            0.865627,
            0.664144,
            0.141606,
            0.722373,
            0.543504,
            0.0809351,
            0.893155,
            0.401318,
            0.887814,
            0.359722,
        };

        private double _error;
        private double _recentAverageError;

        public Net(IReadOnlyList<int> topology)
        {
            var taken = 0;

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
                    var a = cheat.Skip(taken).Take(numberOfOutputs).ToList();
                    taken += numberOfOutputs;
                    layer.AddNeuron(new Neuron(numberOfOutputs, neuronNumber, a));
                }

                // Set the output of the bias neutron to 1.0
                layer.Neurons.Last().OutputValue = 1.0;

                layers.Add(layer);
            }

            Layers = layers;
        }

        public List<Layer> Layers { get; }

        public double GetRecentAverageError()
        {
            return _recentAverageError;
        }

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

            // Implement a recent average measurement
            //var left = RecentAverageError* (Constants.RecentAverageSmoothingFactor + _error);
            //const double right = Constants.RecentAverageSmoothingFactor + 1.0;
            //RecentAverageError = left/right;

            _recentAverageError = (_recentAverageError*Constants.RecentAverageSmoothingFactor + _error)/
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
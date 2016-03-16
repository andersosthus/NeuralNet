using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural
{
    [Serializable]
    public class Neuron
    {
        private readonly int _myIndex;

        public Neuron(int numberOfOutputs, int myIndex)
        {
            _myIndex = myIndex;
            Connections = new List<Connection>();

            for (var c = 0; c < numberOfOutputs; c++)
            {
                var weight = Helpers.GetRandomWeight(0, 1); // w[c];
                var connection = new Connection
                {
                    Weight = weight
                };

                Connections.Add(connection);
            }
        }

        public List<Connection> Connections { get; set; }
        public double OutputValue { get; set; }
        public double Gradient { get; private set; }

        public List<double> GetWeights()
        {
            return Connections.Select(x => x.Weight).ToList();
        }

        public void FeedForward(Layer previousLayer)
        {
            var sum = 0.0;

            for (var n = 0; n < previousLayer.Neurons.Count; n++)
            {
                sum += previousLayer.Neurons[n].OutputValue*previousLayer.Neurons[n].Connections[_myIndex].Weight;
            }

            OutputValue = TransferFunction(sum);
        }

        private static double TransferFunction(double input)
        {
            return Math.Tanh(input);
        }

        private static double TransferFunctionDerivative(double input)
        {
            return 1.0 - (Math.Tanh(input)*Math.Tanh(input));
        }

        public void CalculateOutputGradients(double targetValue)
        {
            var delta = targetValue - OutputValue;
            Gradient = delta*TransferFunctionDerivative(OutputValue);
        }

        public void CalculateHiddenGradients(Layer nextLayer)
        {
            var dow = SumDow(nextLayer);
            Gradient = dow*TransferFunctionDerivative(OutputValue);
        }

        private double SumDow(Layer nextLayer)
        {
            var sum = 0.0;

            // Sum our contributions of the errors at the nodes we feed
            for (var n = 0; n < nextLayer.Neurons.Count - 1; n++)
            {
                sum += Connections[n].Weight*nextLayer.Neurons[n].Gradient;
            }

            return sum;
        }

        public void UpdateInputWeights(Layer previousLayer)
        {
            // The weights to be updated are in the Connection container
            // in the neurons in the preceding layer

            foreach (var neuron in previousLayer.Neurons)
            {
                var oldDeltaWeight = neuron.Connections[_myIndex].DeltaWeight;

                var newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                    Constants.Eta
                    *neuron.OutputValue
                    *Gradient
                    + Constants.Alpha // Also add momentum = a fraction of the previous delta weight
                    *oldDeltaWeight;

                neuron.Connections[_myIndex].DeltaWeight = newDeltaWeight;
                neuron.Connections[_myIndex].Weight += newDeltaWeight;
            }
        }
    }
}
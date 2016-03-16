using System;

namespace NeuralNet.Components
{
    [Serializable]
    public class Connection
    {
        public double Weight { get; set; }
        public double DeltaWeight { get; set; }
    }
}
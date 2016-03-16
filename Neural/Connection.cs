using System;

namespace Neural
{
    [Serializable]
    public class Connection
    {
        public double Weight { get; set; }
        public double DeltaWeight { get; set; }
    }
}
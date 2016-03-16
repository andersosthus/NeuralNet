using System;

namespace NeuralNet
{
    public static class Helpers
    {
        private static readonly Random Rand;

        static Helpers()
        {
            Rand = new Random();
        }

        public static double GetRandomWeight(double minimum, double maximum)
        {
            return Rand.NextDouble()*(maximum - minimum) + minimum;
        }
    }
}
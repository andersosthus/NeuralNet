namespace NeuralNet
{
    public static class Constants
    {
        public const double Eta = 0.15; // (ETA) 0.0 = Slow learner, 0.2 = Medium learner, 1.0 = Reckless learner
        public const double Alpha = 0.5; // (Alpha) 0.0 = No momentum, 0.5 = Moderate momentum
        public const double RecentAverageSmoothingFactor = 100.0;
    }
}
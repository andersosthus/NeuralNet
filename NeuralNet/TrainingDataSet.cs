using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class TrainingDataSet
    {
        private TrainingDataSet(List<int> topology, List<XorTrainingSample> data)
        {
            Topology = topology;
            Data = data;
        }

        public List<int> Topology { get; set; }
        public List<XorTrainingSample> Data { get; set; }

        public static TrainingDataSet CreateXorTraining(int count)
        {
            var random = new Random();

            var topology = new List<int> {2, 5, 3, 1};
            var data = new List<XorTrainingSample>();

            for (var i = count; i >= 0; --i)
            {
                var val1 = random.Next(0, 2);
                var val2 = random.Next(0, 2);
                var target = val1 ^ val2;

                data.Add(new XorTrainingSample { Left = val1, Right = val2, Output = target });
            }

            return new TrainingDataSet(topology, data);
        }
    }

    public struct XorTrainingSample
    {
        public double Left { get; set; }
        public double Right { get; set; }
        public double Output { get; set; }
    }
}
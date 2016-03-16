using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Generating training data...");
            var trainingData = TrainingDataSet.CreateXorTraining(4000);

            Console.WriteLine("Training network...");
            var myNet = new Net(trainingData.Topology);
            var trainingPass = 0;

            var runs = 1;

            for (var i = 0; i < runs; i++)
            {
                foreach (var data in trainingData.Data)
                {
                    trainingPass++;

                    Console.WriteLine($"Pass {trainingPass}");
                    Console.WriteLine($"Input left: {data.Left} - Input right: {data.Right}");

                    myNet.FeedForward(new List<double> { data.Left, data.Right });
                    var results = myNet.GetResults();

                    Console.WriteLine($"Outputs: {results.Last()}");
                    Console.WriteLine($"Targets: {data.Output}");

                    myNet.BackProp(new List<double> { data.Output });
                    Console.WriteLine($"Net recent average error: {myNet.GetRecentAverageError()}");
                }
            }

            for (var i = 0; i < myNet.Layers.Count; i++)
            {
                for (var n = 0; n < myNet.Layers[i].Neurons.Count; n++)
                {
                    Console.WriteLine($"Layer {i} - Neuron {n} has {myNet.Layers[i].Neurons[n].Connections.Count} connections");
                    for (var c = 0; c < myNet.Layers[i].Neurons[n].Connections.Count; c++)
                    {
                        Console.WriteLine($"Layer {i} - Neuron {n} - Connection {c}: {myNet.Layers[i].Neurons[n].Connections[c].Weight}");
                    }
                }
            }

            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using Neural;

namespace NeuralNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            // Normalize ranges 1-4
            for (var i = 1; i <= 4; i++)
            {
                var input = Convert.ToDouble(i);
                double min = 1;
                double max = 4;

                var top = input - min;
                var bottom = max - min;
                var act = top / bottom;
            }


            const string netFile = @"net.bin";
            //if (File.Exists(netFile))
            //{
            //    var fromDisk = File.ReadAllText(netFile);
            //    var newNet = DeserializeFromString<Net>(fromDisk);

            //    newNet.FeedForward(new List<double> { 1, 1 });
            //    var res = newNet.GetResults();
            //    Console.WriteLine($"Got {res[0]}");

            //    Console.WriteLine("Done");
            //    Console.ReadLine();

            //    return;
            //}

            Console.WriteLine("Generating training data...");
            var trainingData = TrainingDataSet.CreateXorTraining(100000);

            Console.WriteLine("Training network...");
            var myNet = new Net(trainingData.Topology);

            var pass = 0;
            foreach (var data in trainingData.Data)
            {
                pass++;

                //Console.WriteLine($"Pass {pass}");
                //Console.WriteLine($"Input left: {data.Left} - Input right: {data.Right}");

                myNet.FeedForward(new List<double> { data.Left, data.Right });
                //var results = myNet.GetResults();

                //Console.WriteLine($"Outputs: {results.Last()}");
                //Console.WriteLine($"Targets: {data.Output}");

                myNet.BackProp(new List<double> { data.Output });
                //Console.WriteLine($"Net recent average error: {myNet.RecentAverageError}");
            }

            //myNet.GetWeights().ForEach(x => Console.WriteLine($"Weight: {x}"));

            myNet.FeedForward(new List<double> { 0, 0 });
            var res = myNet.GetResults();
            Console.WriteLine($"Got {res[0]}");

            //var serializedNet = SerializeToString(myNet);
            //File.WriteAllText(netFile, serializedNet);

            Console.WriteLine("Done");
            Console.ReadLine();
        }

        private static TData DeserializeFromString<TData>(string settings)
        {
            var b = Convert.FromBase64String(settings);
            using (var stream = new MemoryStream(b))
            {
                var formatter = new BinaryFormatter();
                stream.Seek(0, SeekOrigin.Begin);
                return (TData)formatter.Deserialize(stream);
            }
        }

        private static string SerializeToString<TData>(TData settings)
        {
            using (var stream = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(stream, settings);
                stream.Flush();
                stream.Position = 0;
                return Convert.ToBase64String(stream.ToArray());
            }
        }
    }
}
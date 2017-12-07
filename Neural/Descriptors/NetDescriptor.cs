using System;
using System.Collections.Generic;
using System.Linq;
using MoreLinq;

namespace Neural.Descriptors
{
    public class NetDescriptor
    {
        public NetDescriptor()
        {
            Layers = new List<LayerDescriptor>();
        }

        public List<LayerDescriptor> Layers { get; }
        public double Eta { get; set; } = 0.15;
        public double Alpha { get; set; } = 0.5;
        public double RecentAverageSmoothingFactor { get; set; } = 100.0;

        public void AddLayer(LayerType layerType, int neutronCount, bool hasBias)
        {
            if(Layers.Any(x => x.LayerType == LayerType.Input) && layerType == LayerType.Input)
                throw new ArgumentException("You can only have one input layer");
            if (Layers.Any(x => x.LayerType == LayerType.Output) && layerType == LayerType.Output)
                throw new ArgumentException("You can only have one output layer");

            var layer = new LayerDescriptor
            {
                LayerType = layerType,
                NeutronCount = neutronCount,
                HasBias = hasBias,
                Order = GetOrder(layerType)
            };

            Layers.Add(layer);

            ReOrder();
            SetOutputs();
        }

        private void SetOutputs()
        {
            Layers
                .Where(x => x.LayerType != LayerType.Output)
                .OrderBy(x => x.Order)
                .ForEach((item, i) =>
                {
                    var nextI = i + 1;
                    if (nextI < Layers.Count)
                        item.OutputCount = Layers[nextI].NeutronCount;
                });
        }

        private void ReOrder()
        {
            var hasInput = false;

            if (Layers.Any(x => x.LayerType == LayerType.Input))
            {
                Layers.Single(x => x.LayerType == LayerType.Input).Order = 0;
                hasInput = true;
            }

            if(Layers.Any(x =>x.LayerType == LayerType.Output))
                Layers.Single(x => x.LayerType == LayerType.Output).Order = Layers.Count - 1;

            Layers
                .Where(x => x.LayerType == LayerType.Hidden)
                .OrderBy(x => x.Order)
                .ForEach((item, i) =>
                {
                    item.Order = (hasInput) ? i + 1 : i;
                });
        }

        private int GetOrder(LayerType layerType)
        {
            if (layerType == LayerType.Input)
                return 0;

            if (layerType == LayerType.Output)
                return Layers.Count;

            return Layers.Count(x => x.LayerType == LayerType.Hidden) + 1;
        }
    }
}
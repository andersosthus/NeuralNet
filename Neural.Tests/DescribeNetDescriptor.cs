using System;
using System.Linq;
using FluentAssertions;
using MoreLinq;
using Neural.Descriptors;
using Xunit;

namespace Neural.Tests
{
    public class DescribeNetDescriptor
    {
        [Fact]
        public void ItShouldNotHaveAnyLayers()
        {
            // g
            var descriptor = new NetDescriptor();

            // w

            // t
            descriptor.Layers.Count.Should().Be(0);
        }

        [Fact]
        public void ItShouldBeAbleToAddInputLayer()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Input, 1, false);

            // t
            descriptor.Layers.Count.Should().Be(1);
            descriptor.Layers.First().LayerType.Should().Be(LayerType.Input);
        }

        [Fact]
        public void ItShouldNotBeAbleToAddTwoInputLayers()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Input, 1, false);

            // t
            Assert.Throws<ArgumentException>(() => { descriptor.AddLayer(LayerType.Input, 1, false); });
        }

        [Fact]
        public void ItShouldBeAbleToAddOutputLayer()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Output, 1, false);

            // t
            descriptor.Layers.Count.Should().Be(1);
            descriptor.Layers.First().LayerType.Should().Be(LayerType.Output);
        }

        [Fact]
        public void ItShouldNotBeAbleToAddTwoOutputLayers()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Output, 1, false);

            // t
            Assert.Throws<ArgumentException>(() => { descriptor.AddLayer(LayerType.Output, 1, false); });
        }

        [Fact]
        public void WithOneLayerItShouldHaveOrder0()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Output, 1, false);

            // t
            descriptor.Layers.First().Order.Should().Be(0);
        }

        [Fact]
        public void WithTwoLayersItShouldHaveOrder01()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Output, 1, false);
            descriptor.AddLayer(LayerType.Input, 1, false);

            // t
            var actual = descriptor.Layers.OrderBy(x => x.Order).ToList();
            actual.First().Order.Should().Be(0);
            actual.Last().Order.Should().Be(1);
        }

        [Fact]
        public void WithThreeLayersItShouldHaveOrder012()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Output, 1, false);
            descriptor.AddLayer(LayerType.Input, 1, false);
            descriptor.AddLayer(LayerType.Hidden, 1, false);

            // t
            var actual = descriptor.Layers.OrderBy(x => x.Order).ToList();
            actual.ForEach((item, i) =>
            {
                item.Order.Should().Be(i);
            });

            actual.First().LayerType.Should().Be(LayerType.Input);
            actual.Last().LayerType.Should().Be(LayerType.Output);
        }

        [Fact]
        public void WithFiveLayersItShouldHaveOrder01234()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            descriptor.AddLayer(LayerType.Hidden, 1, false);
            descriptor.AddLayer(LayerType.Output, 1, false);
            descriptor.AddLayer(LayerType.Input, 1, false);
            descriptor.AddLayer(LayerType.Hidden, 1, false);
            descriptor.AddLayer(LayerType.Hidden, 1, false);

            // t
            var actual = descriptor.Layers.OrderBy(x => x.Order).ToList();
            actual.ForEach((item, i) =>
            {
                item.Order.Should().Be(i);
            });

            actual.First().LayerType.Should().Be(LayerType.Input);
            actual.Last().LayerType.Should().Be(LayerType.Output);
        }

        [Fact]
        public void ItShouldSetOutputWithTwoLayers()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            var outputNeutrons = 1;
            descriptor.AddLayer(LayerType.Input, 1, false);
            descriptor.AddLayer(LayerType.Output, outputNeutrons, false);

            // t
            var actual = descriptor.Layers.Single(x => x.LayerType == LayerType.Input);
            actual.OutputCount.Should().Be(outputNeutrons);
        }

        [Fact]
        public void ItShouldSetOutputWithThreeLayers()
        {
            // g
            var descriptor = new NetDescriptor();

            // w
            var outputNeutrons = 1;
            var hiddenNeutrons = 1;
            descriptor.AddLayer(LayerType.Input, 1, false);
            descriptor.AddLayer(LayerType.Hidden, hiddenNeutrons, false);
            descriptor.AddLayer(LayerType.Output, outputNeutrons, false);

            // t
            descriptor.Layers.Single(x => x.LayerType == LayerType.Input).OutputCount.Should().Be(hiddenNeutrons);
            descriptor.Layers.Single(x => x.LayerType == LayerType.Hidden).OutputCount.Should().Be(outputNeutrons);
        }
    }
}
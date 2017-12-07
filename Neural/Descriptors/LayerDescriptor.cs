namespace Neural.Descriptors
{
    public class LayerDescriptor
    {
        public LayerType LayerType { get; set; }
        public int NeutronCount { get; set; }
        public int OutputCount { private get; set; }
        public bool HasBias { get; set; } = true;
        public int Order { get; set; }

        public int GetOutputCount()
        {
            if (HasBias)
                return OutputCount + 1;

            return OutputCount;
        }
    }
}

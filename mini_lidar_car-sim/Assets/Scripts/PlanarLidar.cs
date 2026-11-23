using System;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public sealed class PlanarLidar : MonoBehaviour
{
    private const int BeamCount = 360;
    [SerializeField, Min(0f)] private float heightAboveChassisCenter = 0.08f;
    [SerializeField, Min(0.1f)] private float maxRange = 8f;

    [SerializeField] private bool showGizmos = true;
    [SerializeField, Range(1, 36)] private int gizmoBeamStride = 6;

    [SerializeField, Min(0.15f)] private float trainingMaxRange = 6f;
    [SerializeField, Min(0f)] private float noiseBaseMeters = 0.004f;
    [SerializeField, Min(0f)] private float noisePerMeter = 0.003f;
    [SerializeField, Range(0f, 0.2f)] private float beamDropout = 0.015f;

    internal sealed class NoiseRealization
    {
        public readonly System.Random Random;
        public readonly float Bias;

        public NoiseRealization(int seed)
        {
            Random = new System.Random(seed);
            Bias = ((float)Random.NextDouble() * 2f - 1f) * 0.008f;
        }
    }

    private NoiseRealization noise;
    private Vector3 scanOrigin;
    private Vector3 scanForward;
    private float scanMaxRange;
    private readonly float[] ranges = new float[BeamCount];
    private readonly float[] rawRanges = new float[BeamCount];
    private double nextScanTime;
    private Rigidbody body;

    public event Action ScanReady;

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
    }

    // Index zero is forward. Angles increase toward the left, one degree per beam.
    // Positive infinity indicates no return within maxRange.
    public ReadOnlySpan<float> Ranges => ranges;
    public double ScanTime { get; private set; } = double.NegativeInfinity;

    public void SetNoiseSeed(int? seed)
    {
        noise = seed.HasValue ? new NoiseRealization(seed.Value) : null;
    }

    internal void ScanOnNextFixedUpdate() => nextScanTime = double.NegativeInfinity;

    internal void SampleNoise(NoiseRealization realization, Span<float> destination)
    {
        for (int i = 0; i < rawRanges.Length; i++)
            destination[i] = ModelReturn(rawRanges[i], scanMaxRange, realization);
    }

    private float ModelReturn(float measuredRange, float activeRange, NoiseRealization realization)
    {
        if (float.IsPositiveInfinity(measuredRange) || realization == null)
            return measuredRange;
        if (measuredRange < 0.15f || realization.Random.NextDouble() < beamDropout)
            return float.PositiveInfinity;

        double u = 1.0 - realization.Random.NextDouble();
        double v = realization.Random.NextDouble();
        float normal = (float)(Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v));
        float noisy = measuredRange + realization.Bias + normal * (noiseBaseMeters + noisePerMeter * measuredRange);
        return Mathf.Clamp(noisy, 0.15f, activeRange);
    }

    private void FixedUpdate()
    {
        double now = Time.fixedTimeAsDouble;
        if (now + 1e-6 < nextScanTime)
            return;

        scanOrigin = body.position + Vector3.up * heightAboveChassisCenter;
        scanForward = body.rotation * Vector3.forward;
        scanMaxRange = noise == null ? maxRange : Mathf.Min(maxRange, trainingMaxRange);
        for (int i = 0; i < ranges.Length; i++)
        {
            Vector3 direction = BeamDirection(i);
            float measuredRange = Physics.Raycast(scanOrigin, direction, out RaycastHit hit, scanMaxRange,
                                        Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore)
                      ? hit.distance : float.PositiveInfinity;
            rawRanges[i] = measuredRange;
            ranges[i] = ModelReturn(measuredRange, scanMaxRange, noise);
        }

        ScanTime = now;
        nextScanTime = now + 0.1;
        ScanReady?.Invoke();
    }

    private Vector3 BeamDirection(int index)
    {
        return Quaternion.AngleAxis(-index, Vector3.up) * scanForward;
    }

    private void OnDrawGizmos()
    {
        if (!showGizmos || !Application.isPlaying || double.IsNegativeInfinity(ScanTime))
            return;

        Color previousColor = Gizmos.color;
        int stride = Mathf.Max(1, gizmoBeamStride);
        for (int i = 0; i < ranges.Length; i += stride)
        {
            float range = ranges[i];
            bool hit = !float.IsPositiveInfinity(range);
            Vector3 end = scanOrigin + BeamDirection(i) * (hit ? range : scanMaxRange);
            Gizmos.color = hit ? new Color(0.1f, 0.8f, 1f, 0.7f) : new Color(0.5f, 0.5f, 0.5f, 0.22f);
            Gizmos.DrawLine(scanOrigin, end);
            if (hit)
                Gizmos.DrawSphere(end, 0.014f);
        }
        Gizmos.color = previousColor;
    }
}

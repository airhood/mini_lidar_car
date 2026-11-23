using System;
using System.IO;
using UnityEngine;

internal sealed class LowerPolicyModel
{
    private const int Beams = 360;
    private const int History = 4;
    private const float ScanLimit = 8f;
    private readonly float[] history = new float[History * Beams];
    private readonly float[] conv1 = new float[16 * 120];
    private readonly float[] conv2 = new float[32 * 40];
    private const int Sectors = 12;
    private const int TemporalWidth = 16;
    private readonly float[] pooled;
    private readonly bool temporal;
    private readonly float[] sectorRanges;
    private readonly float[] temporalState;
    private readonly float[] nextTemporalState;
    private readonly float[] hidden = new float[64];
    private readonly float[] output = new float[2];
    private readonly float[][] parameters;
    private readonly int historyStride;
    private readonly int capacity;
    private readonly float[] bufferedScans;
    private int newestScan = -1;
    private int bufferedCount;

    public bool UsesRelativeGoal { get; }

    public LowerPolicyModel(byte[] weights)
    {
        using var reader = new BinaryReader(new MemoryStream(weights, writable: false));
        string header = new string(reader.ReadChars(8));
        temporal = header == "LIDART01" || header == "LIDART05";
        historyStride = header == "LIDARH05" || header == "LIDART05" ? 5 : 1;
        UsesRelativeGoal = header == "LIDARG01" || temporal || header == "LIDARH05";
        if (header != "LIDARP01" && !UsesRelativeGoal)
            throw new InvalidDataException("Invalid lower policy header");
        capacity = (History - 1) * historyStride + 1;
        bufferedScans = new float[capacity * Beams];
        pooled = new float[774 + (temporal ? TemporalWidth : 0)];
        if (temporal)
        {
            sectorRanges = new float[Sectors];
            temporalState = new float[TemporalWidth];
            nextTemporalState = new float[TemporalWidth];
        }
        int[] expected = temporal
            ? new[] { 16 * 4 * 9, 16, 32 * 16 * 5, 32, 64 * pooled.Length, 64, 2 * 64, 2,
                      3 * TemporalWidth * Sectors, 3 * TemporalWidth * TemporalWidth, 3 * TemporalWidth, 3 * TemporalWidth }
            : new[] { 16 * 4 * 9, 16, 32 * 16 * 5, 32, 64 * pooled.Length, 64, 2 * 64, 2 };
        parameters = new float[expected.Length][];
        for (int i = 0; i < expected.Length; i++)
        {
            if (reader.ReadInt32() != expected[i])
                throw new InvalidDataException($"Invalid parameter shape at {i}");
            parameters[i] = new float[expected[i]];
            for (int j = 0; j < expected[i]; j++)
                parameters[i][j] = reader.ReadSingle();
        }
        if (reader.BaseStream.Position != reader.BaseStream.Length)
            throw new InvalidDataException("Unexpected trailing lower policy bytes");
    }

    public void Reset()
    {
        newestScan = -1;
        bufferedCount = 0;
    }

    public Vector2 Predict(ReadOnlySpan<float> ranges, NavigationIntent intent, Vector2 previous)
    {
        if (ranges.Length != Beams || (int)intent < 0 || (int)intent >= 4)
            throw new ArgumentException("Expected 360 ranges and a valid navigation intent");
        newestScan = (newestScan + 1) % capacity;
        bufferedCount = Math.Min(bufferedCount + 1, capacity);
        for (int i = 0; i < Beams; i++)
        {
            float distance = ranges[i];
            bufferedScans[newestScan * Beams + i] = float.IsNaN(distance) || distance < 0f
                ? 1f : Mathf.Min(distance, ScanLimit) / ScanLimit;
        }
        for (int frame = 0; frame < History; frame++)
        {
            int offset = Math.Min((History - 1 - frame) * historyStride, bufferedCount - 1);
            int index = (newestScan - offset + capacity) % capacity;
            Array.Copy(bufferedScans, index * Beams, history, frame * Beams, Beams);
        }

        Convolve(history, 4, 360, conv1, 16, 120, 9, parameters[0], parameters[1]);
        Convolve(conv1, 16, 120, conv2, 32, 40, 5, parameters[2], parameters[3]);
        for (int channel = 0; channel < 32; channel++)
        for (int bin = 0; bin < 24; bin++)
        {
            int start = bin * 40 / 24;
            int end = ((bin + 1) * 40 + 23) / 24;
            float sum = 0f;
            for (int position = start; position < end; position++)
                sum += conv2[channel * 40 + position];
            pooled[channel * 24 + bin] = sum / (end - start);
        }

        for (int i = 0; i < 4; i++)
            pooled[32 * 24 + i] = i == (int)intent ? 1f : 0f;
        pooled[772] = previous.x;
        pooled[773] = previous.y;
        if (temporal)
            EncodeTemporal();
        Dense(pooled, hidden, parameters[4], parameters[5], relu: true);
        Dense(hidden, output, parameters[6], parameters[7], relu: false);
        return new Vector2((float)Math.Tanh(output[0]), (float)Math.Tanh(output[1]));
    }

    private void EncodeTemporal()
    {
        Array.Clear(temporalState, 0, TemporalWidth);
        for (int frame = 0; frame < History; frame++)
        {
            const int beamsPerSector = Beams / Sectors;
            for (int sector = 0; sector < Sectors; sector++)
            {
                float closest = 1f;
                int start = frame * Beams + sector * beamsPerSector;
                for (int offset = 0; offset < beamsPerSector; offset++)
                    closest = Mathf.Min(closest, history[start + offset]);
                sectorRanges[sector] = closest;
            }

            for (int unit = 0; unit < TemporalWidth; unit++)
            {
                int updateGate = unit + TemporalWidth;
                int candidateGate = unit + 2 * TemporalWidth;
                float reset = Sigmoid(InputGate(unit) + HiddenGate(unit));
                float update = Sigmoid(InputGate(updateGate) + HiddenGate(updateGate));
                float candidate = (float)Math.Tanh(InputGate(candidateGate) + reset * HiddenGate(candidateGate));
                nextTemporalState[unit] = (1f - update) * candidate + update * temporalState[unit];
            }
            Array.Copy(nextTemporalState, temporalState, TemporalWidth);
        }
        Array.Copy(temporalState, 0, pooled, 774, TemporalWidth);
    }

    private float InputGate(int gate) =>
        AffineGate(sectorRanges, parameters[8], parameters[10], gate);

    private float HiddenGate(int gate) =>
        AffineGate(temporalState, parameters[9], parameters[11], gate);

    private static float AffineGate(float[] features, float[] weights, float[] bias, int gate)
    {
        float value = bias[gate];
        for (int i = 0; i < features.Length; i++)
            value += features[i] * weights[gate * features.Length + i];
        return value;
    }

    private static float Sigmoid(float value) => 1f / (1f + (float)Math.Exp(-value));

    private static void Convolve(float[] x, int channels, int length, float[] y, int filters,
                                 int steps, int kernel, float[] weights, float[] bias)
    {
        for (int filter = 0; filter < filters; filter++)
        for (int step = 0; step < steps; step++)
        {
            float sum = bias[filter];
            for (int channel = 0; channel < channels; channel++)
            for (int offset = 0; offset < kernel; offset++)
            {
                int position = (step * 3 + offset - kernel / 2 + length) % length;
                sum += x[channel * length + position] * weights[(filter * channels + channel) * kernel + offset];
            }
            y[filter * steps + step] = Mathf.Max(0f, sum);
        }
    }

    private static void Dense(float[] x, float[] y, float[] weights, float[] bias, bool relu)
    {
        for (int i = 0; i < y.Length; i++)
        {
            float sum = bias[i];
            for (int j = 0; j < x.Length; j++)
                sum += x[j] * weights[i * x.Length + j];
            y[i] = relu ? Mathf.Max(0f, sum) : sum;
        }
    }
}

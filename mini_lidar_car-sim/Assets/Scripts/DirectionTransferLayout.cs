using System;
using System.Collections.Generic;
using UnityEngine;

internal sealed class DirectionTransferLayout
{
    public const int ScenarioCount = 7;
    public const int StressScenarioCount = 6;
    public readonly List<Rect> Walls = new List<Rect>();
    public readonly List<Rect> Boxes = new List<Rect>();
    public readonly string Family;
    public readonly Rect Area;
    public readonly Vector2 Start;
    public readonly float StartHeading;
    public readonly float ExitZ;
    public readonly float ExitHalfWidth;

    public const int LargeScenarioCount = 3;

    public Vector2 DoorBearing(Vector2 position, bool continueToExit)
    {
        // The four outer walls come first; each subsequent pair defines one doorway.
        for (int i = 4; i + 1 < Walls.Count; i += 2)
        {
            Rect left = Walls[i];
            Rect right = Walls[i + 1];
            if (position.y <= left.yMax + 0.28f)
            {
                var target = new Vector2((left.xMax + right.xMin) * 0.5f, left.yMax + 0.60f);
                return (target - position).normalized;
            }
        }
        if (!continueToExit)
            return Vector2.up;
        Vector2 exitOffset = new Vector2(0f, ExitZ + 0.60f) - position;
        return exitOffset.sqrMagnitude > 1e-10f ? exitOffset.normalized : Vector2.up;
    }

    public DirectionTransferLayout(int seed, int scenario, bool stress = false, bool large = false, bool doors = false)
    {
        if (large)
        {
            var rng = new System.Random(unchecked(seed * 104729 + scenario * 8191 + (doors ? 761 : 0)));
            float Draw(float min, float max) => Mathf.Lerp(min, max, (float)rng.NextDouble());
            float half = Draw(8.5f, 13f);
            float endZ = Draw(19f, 25f);
            Start = new Vector2(Draw(-2.9f, 2.9f), Draw(0.0f, 1.2f));
            ExitZ = Draw(12.5f, 16.0f);
            ExitHalfWidth = half - 0.40f;
            Area = Rect.MinMaxRect(-half, -2f, half, endZ);
            Walls.Add(Rect.MinMaxRect(-half - 0.15f, -2.15f, -half, endZ + 0.15f));
            Walls.Add(Rect.MinMaxRect(half, -2.15f, half + 0.15f, endZ + 0.15f));
            Walls.Add(Rect.MinMaxRect(-half, -2.15f, half, -2f));
            Walls.Add(Rect.MinMaxRect(-half, endZ, half, endZ + 0.15f));
            Family = doors ? scenario switch
            {
                0 => "variable-width-doors",
                1 => "offset-doors",
                2 => "post-door-clutter",
                _ => throw new ArgumentOutOfRangeException(nameof(scenario)),
            } : scenario switch
            {
                0 => "open-furniture",
                1 => "scattered-islands",
                2 => "offset-partitions",
                _ => throw new ArgumentOutOfRangeException(nameof(scenario)),
            };

            if (doors)
            {
                float doorSide = seed % 2 == 0 ? 1f : -1f;
                float firstGate = doorSide * Draw(0.7f, 2.7f);
                Start = new Vector2(firstGate - doorSide * Draw(1.05f, 2.25f), Draw(0f, 0.9f));
                for (int row = 0; row < 2; row++)
                {
                    float z = 4.3f + row * 4.2f + Draw(-0.45f, 0.45f);
                    float center = row == 0 ? firstGate : -doorSide * Draw(0.5f, scenario == 0 ? 1.8f : 3.4f);
                    float opening = Draw(scenario == 0 ? 1.10f : 0.80f, scenario == 0 ? 1.65f : 1.27f);
                    Walls.Add(Rect.MinMaxRect(-half, z, center - opening, z + 0.23f));
                    Walls.Add(Rect.MinMaxRect(center + opening, z, half, z + 0.23f));
                    if (row == 0)
                    {
                        float approach = Mathf.Atan2(center - Start.x, z - Start.y) * Mathf.Rad2Deg;
                        StartHeading = Mathf.Clamp(approach + Draw(-9f, 9f), -30f, 30f);
                    }
                    if (scenario == 2)
                    {
                        float obstacleX = center + (row == 0 ? doorSide : -doorSide) * Draw(0.35f, 0.83f);
                        Boxes.Add(new Rect(obstacleX - 0.36f, z + Draw(0.90f, 1.34f), 0.72f, Draw(0.52f, 0.86f)));
                    }
                }
            }
            else if (scenario == 2)
            {
                for (int row = 0; row < 2; row++)
                {
                    float z = 4.7f + row * 4.2f + Draw(-0.45f, 0.45f);
                    float center = (row == 0 ? 1f : -1f) * Draw(0.8f, 2.4f);
                    if (seed % 2 == 0)
                        center = -center;
                    float opening = Draw(2.0f, 3.3f);
                    Walls.Add(Rect.MinMaxRect(-half, z, center - opening, z + 0.22f));
                    Walls.Add(Rect.MinMaxRect(center + opening, z, half, z + 0.22f));
                }
            }

            int count = doors ? 12 : (scenario == 1 ? 25 : 17);
            for (int i = 0; i < 180 && Boxes.Count < count; i++)
            {
                float x = Draw(-half + 0.9f, half - 0.9f);
                float z = Draw(2.35f, ExitZ - 0.9f);
                if (i < 5)
                    x = Draw(-3.4f, 3.4f);
                float width = Draw(0.48f, scenario == 1 ? 2.15f : 1.30f);
                float length = Draw(0.48f, scenario == 1 ? 2.65f : 1.33f);
                var box = new Rect(x - width / 2f, z - length / 2f, width, length);
                var padded = Rect.MinMaxRect(box.xMin - 0.42f, box.yMin - 0.42f,
                                             box.xMax + 0.42f, box.yMax + 0.42f);
                if (padded.Contains(Start) || padded.yMin < 1.6f || padded.yMax > ExitZ - 0.25f ||
                    padded.xMin <= -half || padded.xMax >= half ||
                    Walls.Exists(wall => wall.Overlaps(padded)) || Boxes.Exists(other => other.Overlaps(padded)))
                    continue;
                Boxes.Add(box);
            }
            Boxes.Sort((a, b) => a.yMax.CompareTo(b.yMax));
            return;
        }

        int family = scenario / 2;
        var random = new System.Random(unchecked(seed * 65537 + family * 8191));
        float side = scenario % 2 == 0 ? -1f : 1f;
        float halfWidth = Sample(1.60f, 2.14f);
        Start = new Vector2(Sample(-0.12f, 0.12f), 0f);
        if (stress)
        {
            switch (family)
            {
                case 0:
                    Family = "short-gap";
                    halfWidth = Sample(1.56f, 1.92f);
                    float front = Sample(1.48f, 1.84f);
                    for (int i = 0; i < 4; i++)
                    {
                        float length = Sample(0.43f, 0.64f);
                        float direction = i % 2 == 0 ? side : -side;
                        AddBox(direction * Sample(0.28f, 0.44f), front,
                               Sample(0.70f, 0.92f), length);
                        front += length + Sample(0.79f, 1.00f);
                    }
                    break;
                case 1:
                    Family = "angled-entry";
                    halfWidth = Sample(1.96f, 2.44f);
                    float entryLength = Sample(2.02f, 2.36f);
                    float entryHalfWidth = Sample(0.66f, 0.78f);
                    float shift = side * Sample(0.53f, 0.76f);
                    const int slices = 12;
                    float sliceLength = entryLength / slices;
                    for (int i = 0; i < slices; i++)
                    {
                        float z = i * sliceLength;
                        float center = shift * (i + 0.5f) / slices;
                        float back = i == 0 ? -0.65f : z;
                        Walls.Add(Rect.MinMaxRect(-halfWidth, back, center - entryHalfWidth, z + sliceLength));
                        Walls.Add(Rect.MinMaxRect(center + entryHalfWidth, back, halfWidth, z + sliceLength));
                    }
                    AddBox(shift + side * Sample(0.00f, 0.20f), entryLength + Sample(0.77f, 1.07f),
                           Sample(0.68f, 0.88f), Sample(0.47f, 0.65f));
                    AddBox(-side * Sample(0.22f, 0.56f), entryLength + Sample(2.40f, 2.82f),
                           Sample(0.67f, 0.86f), Sample(0.47f, 0.72f));
                    break;
                case 2:
                    Family = "wall-side";
                    halfWidth = Sample(1.52f, 1.74f);
                    AddBox(side * Sample(0.46f, 0.58f), Sample(1.25f, 1.58f),
                           Sample(1.24f, 1.43f), Sample(0.54f, 0.74f));
                    AddBox(-side * Sample(0.59f, 0.67f), Sample(2.99f, 3.22f),
                           Sample(1.43f, 1.62f), Sample(0.56f, 0.79f));
                    AddBox(side * Sample(0.37f, 0.52f), Sample(4.66f, 4.98f),
                           Sample(0.86f, 1.04f), Sample(0.56f, 0.75f));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(scenario));
            }
        }
        else switch (family)
        {
            case 0:
                Family = "freestanding";
                AddBox(side * Sample(0.16f, 0.24f), Sample(1.50f, 2.38f),
                       Sample(0.67f, 0.95f), Sample(0.55f, 0.95f));
                break;
            case 1:
                Family = "staggered";
                float offset = Sample(0.40f, 0.60f);
                AddBox(side * offset, Sample(1.5f, 1.90f), Sample(0.84f, 1.04f),
                       Sample(0.56f, 0.74f));
                AddBox(-side * (offset + 0.08f), Sample(3.18f, 3.58f),
                       Sample(1.00f, 1.25f), Sample(0.52f, 0.76f));
                AddBox(side * (offset + 0.29f), Sample(4.83f, 5.24f),
                       Sample(0.78f, 0.98f), Sample(0.52f, 0.76f));
                break;
            case 2:
                Family = "corridor-hall";
                halfWidth = Sample(1.90f, 2.44f);
                float mouth = Sample(1.80f, 2.38f);
                float corridorHalfWidth = Sample(0.62f, 0.82f);
                Walls.Add(Rect.MinMaxRect(-halfWidth, -0.65f, -corridorHalfWidth, mouth));
                Walls.Add(Rect.MinMaxRect(corridorHalfWidth, -0.65f, halfWidth, mouth));
                AddBox(side * Sample(0.30f, 0.44f), mouth + Sample(0.78f, 1.25f),
                       Sample(0.72f, 0.94f), Sample(0.55f, 0.74f));
                AddBox(-side * Sample(0.70f, 1.08f), mouth + Sample(2.55f, 2.96f),
                       Sample(0.55f, 0.83f), Sample(0.52f, 0.78f));
                break;
            case 3:
                Family = "clear";
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(scenario));
        }

        ExitZ = Boxes.Count == 0 ? Sample(3.6f, 4.8f) : Boxes[Boxes.Count - 1].yMax + Sample(1.10f, 1.42f);
        ExitHalfWidth = halfWidth - 0.35f;
        float end = ExitZ + 0.72f;
        Area = Rect.MinMaxRect(-halfWidth, -0.65f, halfWidth, end);
        Walls.Add(Rect.MinMaxRect(-halfWidth - 0.15f, -0.8f, -halfWidth, end + 0.15f));
        Walls.Add(Rect.MinMaxRect(halfWidth, -0.8f, halfWidth + 0.15f, end + 0.15f));
        Walls.Add(Rect.MinMaxRect(-halfWidth, -0.8f, halfWidth, -0.65f));
        Walls.Add(Rect.MinMaxRect(-halfWidth, end, halfWidth, end + 0.15f));

        float Sample(float min, float max) => Mathf.Lerp(min, max, (float)random.NextDouble());

        void AddBox(float x, float z, float width, float length) =>
            Boxes.Add(new Rect(x - width * 0.5f, z, width, length));
    }
}

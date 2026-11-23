using System;
using System.Collections.Generic;
using UnityEngine;

internal static class AvoidancePlacement
{
    public static bool TryPlace(GraphFloorPlan map, GraphFloorPlan.Task task, int seed,
                                bool placeAfterTurn, float clearance, out Rect obstacle,
                                out List<Vector2> reference, out List<Vector2> detour)
    {
        obstacle = default;
        reference = new List<Vector2>();
        detour = new List<Vector2>();
        var walls = new List<Bounds>();
        foreach (GraphFloorPlan.Wall wall in map.Walls)
            walls.Add(ToBounds(wall.Area));

        if (!Plan(map, task, clearance, walls, reference, out int junctionIndex))
            return false;

        var random = new System.Random(seed);
        int first = placeAfterTurn ? junctionIndex : 0;
        int last = placeAfterTurn ? reference.Count - 1 : junctionIndex;
        if (last <= first)
            return false;

        float desiredSide = random.Next(2) == 0 ? -1 : 1;
        for (int attempt = 0; attempt < 30; attempt++)
        {
            int segment = first + attempt % (last - first);
            Vector2 a = reference[segment], b = reference[segment + 1];
            float fraction = 0.32f + (float)random.NextDouble() * 0.36f;
            Vector2 center = Vector2.Lerp(a, b, fraction);
            if (Vector2.Distance(center, task.Start) < 1.25f ||
                Vector2.Distance(center, task.Via) < 0.65f ||
                Vector2.Distance(center, task.Goal) < 1.15f)
                continue;

            float w = 0.26f + (float)random.NextDouble() * 0.52f;
            float h = 0.26f + (float)random.NextDouble() * 0.52f;
            var box = new Rect(center.x - w / 2f, center.y - h / 2f, w, h);
            if (!map.Area.Contains(box.min) || !map.Area.Contains(box.max) ||
                walls.Exists(wall => Overlaps(wall, box)))
                continue;

            Vector2 tangent = (b - a).normalized;
            Vector2 normal = new Vector2(-tangent.y, tangent.x);
            float extent = (Mathf.Abs(tangent.x) * w + Mathf.Abs(tangent.y) * h) / 2f + clearance;
            float offset = (Mathf.Abs(normal.x) * w + Mathf.Abs(normal.y) * h) / 2f + clearance + 0.04f;
            float ramp = extent + 0.52f;
            if (Vector2.Distance(center, a) < ramp || Vector2.Distance(center, b) < ramp)
                continue;

            Vector2 enter = center - tangent * ramp;
            Vector2 exit = center + tangent * ramp;
            float side = attempt % 2 == 0 ? desiredSide : -desiredSide;
            Vector2 aroundA = center - tangent * (extent + 0.08f) + normal * offset * side;
            Vector2 aroundB = center + tangent * (extent + 0.08f) + normal * offset * side;
            walls.Add(ToBounds(box));
            bool possible = Clear(enter, aroundA) && Clear(aroundA, aroundB) && Clear(aroundB, exit);
            if (!possible)
            {
                walls.RemoveAt(walls.Count - 1);
                continue;
            }

            detour.Clear();
            detour.AddRange(reference.GetRange(0, segment + 1));
            detour.Add(enter);
            detour.Add(aroundA);
            detour.Add(aroundB);
            detour.Add(exit);
            detour.AddRange(reference.GetRange(segment + 1, reference.Count - segment - 1));
            for (int i = 1; i < detour.Count && possible; i++)
                possible = Clear(detour[i - 1], detour[i]);
            walls.RemoveAt(walls.Count - 1);
            if (!possible)
                continue;
            obstacle = box;
            return true;
        }

        detour.Clear();
        return false;

        bool Clear(Vector2 a, Vector2 b)
        {
            if (!map.Area.Contains(a) || !map.Area.Contains(b))
                return false;
            float minX = Mathf.Min(a.x, b.x) - clearance;
            float minZ = Mathf.Min(a.y, b.y) - clearance;
            float maxX = Mathf.Max(a.x, b.x) + clearance;
            float maxZ = Mathf.Max(a.y, b.y) + clearance;
            int samples = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / 0.025f));
            foreach (Bounds wall in walls)
            {
                Vector3 min = wall.min, max = wall.max;
                if (min.x > maxX || max.x < minX || min.z > maxZ || max.z < minZ)
                    continue;
                for (int i = 0; i <= samples; i++)
                {
                    Vector2 point = Vector2.Lerp(a, b, (float)i / samples);
                    float dx = Mathf.Max(0f, Mathf.Max(min.x - point.x, point.x - max.x));
                    float dz = Mathf.Max(0f, Mathf.Max(min.z - point.y, point.y - max.z));
                    if (dx * dx + dz * dz < clearance * clearance)
                        return false;
                }
            }
            return true;
        }
    }

    public static bool Plan(GraphFloorPlan map, GraphFloorPlan.Task task, float clearance,
                            List<Bounds> bounds, List<Vector2> route, out int junction)
    {
        junction = 0;
        if (!TeacherRoutePlanner.TryPlan(task.Start, task.Via, map.Area, clearance, bounds,
                                         task.Intent == NavigationIntent.Left, route))
            return false;
        junction = route.Count - 1;
        if (task.Blocked)
            return true;

        var tail = new List<Vector2>();
        if (!TeacherRoutePlanner.TryPlan(task.Via, task.Goal, map.Area, clearance, bounds,
                                         task.Intent == NavigationIntent.Left, tail))
            return false;
        for (int i = 1; i < tail.Count; i++)
            route.Add(tail[i]);
        return true;
    }

    private static bool Overlaps(Bounds wall, Rect box) =>
        box.xMin < wall.max.x && box.xMax > wall.min.x &&
        box.yMin < wall.max.z && box.yMax > wall.min.z;

    public static Bounds ToBounds(Rect rect) =>
        new Bounds(new Vector3(rect.center.x, 0.2f, rect.center.y),
                   new Vector3(rect.width, 0.4f, rect.height));
}

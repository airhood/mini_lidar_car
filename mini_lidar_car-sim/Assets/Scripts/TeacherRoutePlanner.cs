using System.Collections.Generic;
using UnityEngine;

internal static class TeacherRoutePlanner
{
    private const float GridStep = 0.05f;

    public static bool TryPlan(Vector2 start, Vector2 goal, Rect area, float clearance,
                               List<Bounds> obstacles, bool preferLeft, List<Vector2> route)
    {
        var open = new List<Vector2Int>();
        var costs = new Dictionary<Vector2Int, float>();
        var parents = new Dictionary<Vector2Int, Vector2Int>();
        var closed = new HashSet<Vector2Int>();
        var from = Cell(start);
        var to = Cell(goal);
        var buckets = IndexObstacles(obstacles, clearance);
        route.Clear();
        if (!ClearPoint(start) || !ClearPoint(goal))
            return false;
        if (ClearLine(start, goal))
        {
            route.Add(start);
            route.Add(goal);
            return true;
        }

        open.Add(from);
        costs[from] = 0f;

        while (open.Count > 0)
        {
            int best = 0;
            for (int i = 1; i < open.Count; i++)
                if (Score(open[i]) < Score(open[best]))
                    best = i;

            Vector2Int current = open[best];
            open.RemoveAt(best);
            if (current == to)
            {
                var cells = new List<Vector2> { goal };
                while (current != from)
                {
                    cells.Add(Point(current));
                    current = parents[current];
                }
                cells.Add(start);
                cells.Reverse();

                int anchor = 0;
                route.Add(start);
                while (anchor < cells.Count - 1)
                {
                    int next = cells.Count - 1;
                    while (next > anchor + 1 && !ClearLine(cells[anchor], cells[next]))
                        next--;
                    if (!ClearLine(cells[anchor], cells[next]))
                    {
                        route.Clear();
                        return false;
                    }
                    route.Add(cells[next]);
                    anchor = next;
                }
                return true;
            }

            closed.Add(current);
            for (int dx = -1; dx <= 1; dx++)
            for (int dz = -1; dz <= 1; dz++)
            {
                if (dx == 0 && dz == 0)
                    continue;
                var neighbor = current + new Vector2Int(preferLeft ? dx : -dx, dz);
                if (closed.Contains(neighbor) || !ClearLine(Point(current), Point(neighbor)))
                    continue;
                float cost = costs[current] + (dx == 0 || dz == 0 ? 1f : 1.414214f);
                if (costs.TryGetValue(neighbor, out float known) && cost >= known)
                    continue;
                costs[neighbor] = cost;
                parents[neighbor] = current;
                if (!open.Contains(neighbor))
                    open.Add(neighbor);
            }
        }
        return false;

        float Score(Vector2Int cell)
        {
            return costs[cell] + Vector2.Distance(cell, to);
        }

        bool ClearLine(Vector2 a, Vector2 b)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / (GridStep / 4f)));
            for (int i = 0; i <= steps; i++)
            {
                if (!ClearPoint(Vector2.Lerp(a, b, (float)i / steps)))
                    return false;
            }
            return true;
        }
    
        bool ClearPoint(Vector2 point)
        {
            if (!area.Contains(point))
                return false;
            var key = new Vector2Int(Mathf.FloorToInt(point.x), Mathf.FloorToInt(point.y));
            if (!buckets.TryGetValue(key, out List<Bounds> candidates))
                return true;
            foreach (Bounds box in candidates)
            {
                float dx = Mathf.Max(0f, Mathf.Max(box.min.x - point.x, point.x - box.max.x));
                float dz = Mathf.Max(0f, Mathf.Max(box.min.z - point.y, point.y - box.max.z));
                if (dx * dx + dz * dz < clearance * clearance)
                    return false;
            }
            return true;
        }
    }

    private static Dictionary<Vector2Int, List<Bounds>> IndexObstacles(List<Bounds> obstacles, float clearance)
    {
        var buckets = new Dictionary<Vector2Int, List<Bounds>>();
        foreach (Bounds box in obstacles)
        {
            int minX = Mathf.FloorToInt(box.min.x - clearance);
            int maxX = Mathf.FloorToInt(box.max.x + clearance);
            int minZ = Mathf.FloorToInt(box.min.z - clearance);
            int maxZ = Mathf.FloorToInt(box.max.z + clearance);
            for (int z = minZ; z <= maxZ; z++)
            for (int x = minX; x <= maxX; x++)
            {
                var key = new Vector2Int(x, z);
                if (!buckets.TryGetValue(key, out List<Bounds> candidates))
                {
                    candidates = new List<Bounds>();
                    buckets.Add(key, candidates);
                }
                candidates.Add(box);
            }
        }
        return buckets;
    }

    private static Vector2Int Cell(Vector2 point)
    {
        return new Vector2Int(Mathf.RoundToInt(point.x / GridStep),
                              Mathf.RoundToInt(point.y / GridStep));
    }

    private static Vector2 Point(Vector2Int cell)
    {
        return new Vector2(cell.x * GridStep, cell.y * GridStep);
    }
}

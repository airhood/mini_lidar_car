using System.Collections.Generic;
using UnityEngine;
using Random = System.Random;

internal sealed class OpenAvoidanceLayout
{
    public const int MissionsPerSeed = 6;
    private const float Wall = 0.20f;
    private static readonly Vector2[] Sides = { Vector2.right, Vector2.up, Vector2.left, Vector2.down };

    public Rect Area { get; private set; }
    public string Family { get; private set; }
    public Vector2 Start { get; private set; }
    public float Heading { get; private set; }
    public int PortalCount { get; private set; }
    public readonly List<Rect> Walls = new List<Rect>();
    public readonly List<Rect> Boxes = new List<Rect>();
    public readonly List<Vector2> Reference = new List<Vector2>();
    public readonly List<Vector2> Detour = new List<Vector2>();

    public static bool TryCreate(int seed, int mission, float clearance, out OpenAvoidanceLayout layout)
    {
        var random = new Random(unchecked(seed * 1087 + mission * 1237));
        for (int attempt = 0; attempt < 12; attempt++)
        {
            layout = new OpenAvoidanceLayout();
            int style = ((seed % 4) + mission) % 4;
            if (style == 3 ? layout.GenerateChain(random, clearance) : layout.Generate(random, style, clearance))
                return true;
        }
        layout = null;
        return false;
    }

    private bool Generate(Random random, int style, float clearance)
    {
        Family = style == 0 ? "hall-wings" : style == 1 ? "hall-partitions" : "hall-atrium";
        float width = Sample(random, 6.5f, 10.2f);
        float depth = Sample(random, 6.0f, 9.4f);
        var hall = new Rect(-width / 2, -depth / 2, width, depth);
        Area = Rect.MinMaxRect(hall.xMin - 4.5f, hall.yMin - 4.5f, hall.xMax + 4.5f, hall.yMax + 4.5f);
        int entranceSide = random.Next(4);
        int portalMask = ChoosePortals(random, entranceSide, style);
        var terminals = new List<Vector2>();
        var portals = new List<Vector2>();
        for (int side = 0; side < Sides.Length; side++)
        {
            Vector2 outward = Sides[side];
            if ((portalMask & (1 << side)) != 0)
                AddWing(random, hall, outward, terminals, portals);
            else
                AddEdge(hall, outward);
        }
        PortalCount = portals.Count;
        if (style != 0)
        {
            int wallCount = style == 1 ? random.Next(1, 3) : 1;
            for (int i = 0; i < wallCount; i++)
            {
                bool across = random.Next(2) == 0;
                float span = Sample(random, 1.45f, 3.2f);
                float x = Sample(random, -width * 0.30f, width * 0.30f);
                float z = Sample(random, -depth * 0.30f, depth * 0.30f);
                var divider = Solid(new Vector2(x, z), across ? Vector2.up : Vector2.right, span, Wall);
                if (!Walls.Exists(wall => wall.Overlaps(divider)))
                    Walls.Add(divider);
            }
        }

        int entry = random.Next(terminals.Count);
        Vector2 goal = style == 2 && random.Next(3) == 0
            ? new Vector2(Sample(random, -width * 0.37f, width * 0.37f), Sample(random, -depth * 0.37f, depth * 0.37f))
            : terminals[(entry + random.Next(1, terminals.Count)) % terminals.Count];
        return PlanCourse(random, style, new[] { hall }, portals, terminals[entry], goal, clearance);
    }

    private bool GenerateChain(Random random, float clearance)
    {
        Family = "hall-chain";
        PortalCount = 2;
        float reach = Sample(random, 1.0f, 1.65f);
        float leftWidth = Sample(random, 4.05f, 5.4f);
        float rightWidth = Sample(random, 4.15f, 5.65f);
        float leftOffset = Sample(random, -1.9f, -0.65f);
        float rightOffset = Sample(random, 0.65f, 1.9f);
        float leftHalf = Sample(random, 2.65f, 3.9f);
        float rightHalf = Sample(random, 2.65f, 3.9f);
        var left = new Rect(-reach - leftWidth, leftOffset - leftHalf, leftWidth, 2 * leftHalf);
        var right = new Rect(reach, rightOffset - rightHalf, rightWidth, 2 * rightHalf);
        float passageHalf = Sample(random, 0.74f, 1.04f);
        float lower = right.yMin + passageHalf + 0.24f;
        float upper = left.yMax - passageHalf - 0.24f;
        if (lower >= upper)
            return false;
        float passageZ = Sample(random, lower, upper);
        AddChainRoom(left, true, passageZ, passageHalf);
        AddChainRoom(right, false, passageZ, passageHalf);
        Walls.Add(Solid(new Vector2(0, passageZ + passageHalf), Vector2.up, 2f * reach + Wall, Wall));
        Walls.Add(Solid(new Vector2(0, passageZ - passageHalf), Vector2.up, 2f * reach + Wall, Wall));
        var portals = new List<Vector2> { new Vector2(-reach, passageZ), new Vector2(reach, passageZ) };
        var start = new Vector2(left.xMin + Sample(random, 0.75f, 1.35f), leftOffset + Sample(random, -1.45f, 0.85f));
        var goal = new Vector2(right.xMax - Sample(random, 0.75f, 1.35f), rightOffset + Sample(random, -0.85f, 1.45f));
        Area = Rect.MinMaxRect(left.xMin - Wall, Mathf.Min(left.yMin, right.yMin) - Wall,
                               right.xMax + Wall, Mathf.Max(left.yMax, right.yMax) + Wall);
        if (random.Next(2) == 0)
        {
            left = Transpose(left);
            right = Transpose(right);
            Area = Transpose(Area);
            for (int i = 0; i < Walls.Count; i++)
                Walls[i] = Transpose(Walls[i]);
            for (int i = 0; i < portals.Count; i++)
                portals[i] = new Vector2(portals[i].y, portals[i].x);
            start = new Vector2(start.y, start.x);
            goal = new Vector2(goal.y, goal.x);
        }
        if (random.Next(2) == 0)
            (start, goal) = (goal, start);
        return PlanCourse(random, 1, new[] { left, right }, portals, start, goal, clearance);
    }

    private void AddChainRoom(Rect room, bool left, float passageZ, float passageHalf)
    {
        float inner = left ? room.xMax : room.xMin;
        float outer = left ? room.xMin : room.xMax;
        Vector2 edge = new Vector2(inner, 0f);
        AddSpan(edge, Vector2.right, room.yMin, passageZ - passageHalf);
        AddSpan(edge, Vector2.right, passageZ + passageHalf, room.yMax);
        Walls.Add(Solid(new Vector2(outer, room.center.y), Vector2.right, room.height + Wall, Wall));
        Walls.Add(Solid(new Vector2(room.center.x, room.yMin), Vector2.up, room.width + Wall, Wall));
        Walls.Add(Solid(new Vector2(room.center.x, room.yMax), Vector2.up, room.width + Wall, Wall));
    }

    private bool PlanCourse(Random random, int style, Rect[] regions, List<Vector2> portals,
                            Vector2 start, Vector2 goal, float clearance)
    {
        Start = start;
        var walls = Walls.ConvertAll(AvoidancePlacement.ToBounds);
        if (!TeacherRoutePlanner.TryPlan(Start, goal, Area, clearance, walls, false, Reference) || Length(Reference) < 6f)
            return false;
        Vector2 direction = Reference[1] - Start;
        Heading = Mathf.Atan2(direction.x, direction.y) * Mathf.Rad2Deg + Sample(random, -10f, 10f);

        for (int attempt = 0; attempt < 15; attempt++)
        {
            ScatterBoxes(random, style, regions, portals);
            if (Boxes.Count < 5 || Boxes.FindAll(BlocksReference).Count < 2)
                continue;
            var occupied = new List<Bounds>(walls);
            occupied.AddRange(Boxes.ConvertAll(AvoidancePlacement.ToBounds));
            if (TeacherRoutePlanner.TryPlan(Start, goal, Area, clearance, occupied, false, Detour) &&
                Length(Detour) <= 15.5f && Length(Detour) >= Length(Reference) + 0.35f)
                return true;
        }
        return false;
    }

    private static int ChoosePortals(Random random, int entrance, int style)
    {
        int mask = (1 << entrance) | (1 << ((entrance + random.Next(1, 4)) % 4));
        if (style >= 1)
        {
            int extra = random.Next(4);
            while ((mask & (1 << extra)) != 0)
                extra = (extra + 1) % 4;
            mask |= 1 << extra;
        }
        return style == 2 ? 15 : mask;
    }

    private void AddWing(Random random, Rect hall, Vector2 outward, List<Vector2> terminals, List<Vector2> portals)
    {
        Vector2 along = new Vector2(-outward.y, outward.x);
        float half = outward.x == 0 ? hall.width / 2 : hall.height / 2;
        Vector2 edge = outward * (outward.x == 0 ? hall.height / 2 : hall.width / 2);
        float shift = Sample(random, -half * 0.38f, half * 0.38f);
        float opening = Sample(random, 1.35f, 2.10f);
        float wingWidth = Mathf.Min(Sample(random, 2.35f, 3.55f), 2f * (half - Mathf.Abs(shift) - 0.18f));
        float reach = Sample(random, 1.9f, 3.1f);
        Vector2 portal = edge + along * shift;
        AddSpan(edge, outward, -half, shift - opening / 2);
        AddSpan(edge, outward, shift + opening / 2, half);
        Walls.Add(Solid(portal + outward * reach, outward, wingWidth + Wall, Wall));
        foreach (float sign in new[] { -1f, 1f })
            Walls.Add(Solid(portal + outward * (reach / 2) + sign * along * (wingWidth / 2), along, reach + Wall, Wall));
        portals.Add(portal);
        terminals.Add(portal + outward * (reach - 0.70f) + along * Sample(random, -0.32f, 0.32f));
    }

    private void AddEdge(Rect hall, Vector2 outward)
    {
        float span = outward.x == 0 ? hall.width : hall.height;
        float radius = (outward.x == 0 ? hall.height : hall.width) / 2;
        Walls.Add(Solid(outward * radius, outward, span + Wall, Wall));
    }

    private void AddSpan(Vector2 edge, Vector2 outward, float from, float to)
    {
        Vector2 along = new Vector2(-outward.y, outward.x);
        Walls.Add(Solid(edge + along * ((from + to) / 2), outward, to - from + Wall / 2, Wall));
    }

    private void ScatterBoxes(Random random, int style, Rect[] regions, List<Vector2> portals)
    {
        Boxes.Clear();
        int target = random.Next(7, 14);
        Vector2 goal = Reference[Reference.Count - 1];
        for (int attempt = 0; attempt < 75 && Boxes.Count < target; attempt++)
        {
            Rect hall = regions[random.Next(regions.Length)];
            var center = new Vector2(Sample(random, hall.xMin + 0.65f, hall.xMax - 0.65f),
                                     Sample(random, hall.yMin + 0.65f, hall.yMax - 0.65f));
            bool longFurniture = style != 0 && random.Next(3) == 0;
            float width = Sample(random, 0.38f, longFurniture ? 2.5f : 1.25f);
            float depth = Sample(random, 0.35f, longFurniture ? 0.86f : 1.25f);
            Rect box = Solid(center, random.Next(2) == 0 ? Vector2.up : Vector2.right, width, depth);
            Rect gap = Rect.MinMaxRect(box.xMin - 0.18f, box.yMin - 0.18f, box.xMax + 0.18f, box.yMax + 0.18f);
            if (Vector2.Distance(center, Start) < 1.5f || Vector2.Distance(center, goal) < 1.5f ||
                portals.Exists(portal => Vector2.Distance(portal, box.center) < 1.30f) ||
                Walls.Exists(wall => wall.Overlaps(gap)) || Boxes.Exists(other => other.Overlaps(gap)))
                continue;
            Boxes.Add(box);
        }
    }

    private bool BlocksReference(Rect box)
    {
        for (int leg = 1; leg < Reference.Count; leg++)
        {
            Vector2 a = Reference[leg - 1], b = Reference[leg];
            int steps = Mathf.CeilToInt(Vector2.Distance(a, b) / 0.04f);
            for (int step = 0; step < steps; step++)
                if (box.Contains(Vector2.Lerp(a, b, (float)step / steps)))
                    return true;
        }
        return false;
    }

    private static float Length(List<Vector2> points)
    {
        float length = 0;
        for (int i = 1; i < points.Count; i++)
            length += Vector2.Distance(points[i - 1], points[i]);
        return length;
    }

    private static Rect Solid(Vector2 center, Vector2 normal, float span, float depth)
    {
        Vector2 size = normal.x == 0 ? new Vector2(span, depth) : new Vector2(depth, span);
        return new Rect(center - size / 2, size);
    }

    private static float Sample(Random random, float min, float max) =>
        min + (max - min) * (float)random.NextDouble();

    private static Rect Transpose(Rect rect) => new Rect(rect.y, rect.x, rect.height, rect.width);
}

using System;
using System.Collections.Generic;
using UnityEngine;
using Random = System.Random;

internal sealed class GraphFloorPlan
{
    private const float CellSize = 0.10f;
    private static readonly Vector2Int[] Directions =
        { Vector2Int.right, Vector2Int.up, Vector2Int.left, Vector2Int.down };

    internal readonly struct Wall
    {
        public readonly Rect Area;
        public Wall(Rect area) => Area = area;
    }

    internal readonly struct Task
    {
        public readonly Vector2 Start, Via, Goal;
        public readonly float Heading;
        public readonly NavigationIntent Intent;
        public readonly Rect? Obstacle;
        public readonly bool Blocked;

        public Task(Vector2 start, Vector2 via, Vector2 goal, float heading,
                    NavigationIntent intent, Rect? obstacle, bool blocked)
        {
            Start = start;
            Via = via;
            Goal = goal;
            Heading = heading;
            Intent = intent;
            Obstacle = obstacle;
            Blocked = blocked;
        }
    }

    private sealed class Room
    {
        public Vector2 Center;
        public Rect Area;
        public readonly List<Room> Neighbors = new List<Room>();
    }

    private readonly List<Room> rooms = new List<Room>();
    private readonly struct Sweep
    {
        public readonly Vector2 Start, End;
        public readonly float Radius;
        public readonly Room From, To;

        public Sweep(Vector2 start, Vector2 end, float radius, Room from, Room to)
        {
            Start = start;
            End = end;
            Radius = radius;
            From = from;
            To = to;
        }

        public bool Touches(Room room) => From == room || To == room;
    }

    private readonly List<Sweep> sweeps = new List<Sweep>();
    private readonly List<Rect> free = new List<Rect>();
    private readonly Dictionary<(Room, Room), (float width, Vector2 center, Vector2 tangent)> corridors =
        new Dictionary<(Room, Room), (float width, Vector2 center, Vector2 tangent)>();
    public List<float> DoorWidths { get; } = new List<float>();
    public List<float> CorridorWidths { get; } = new List<float>();
    public List<float> RoomAreas { get; } = new List<float>();
    private readonly List<(Room from, Room via, Room to, NavigationIntent intent)> turns =
        new List<(Room from, Room via, Room to, NavigationIntent intent)>();

    public Rect Area { get; }
    public bool IsCorridorNetwork { get; }
    public int RoomCount => rooms.Count;
    public int JunctionCount => rooms.FindAll(room => room.Neighbors.Count >= 3).Count;
    public int CycleCount => corridors.Count / 2 - rooms.Count + 1;
    public List<Wall> Walls { get; } = new List<Wall>();

    public GraphFloorPlan(int seed)
    {
        var random = new Random(seed);
        IsCorridorNetwork = seed % 4 != 0;
        int columns = IsCorridorNetwork ? random.Next(5, 7) : random.Next(3, 5);
        int rows = IsCorridorNetwork ? random.Next(3, 5) : random.Next(3, 5);
        float pitchX = IsCorridorNetwork ? Between(random, 5.40f, 6.50f) : Between(random, 3.15f, 4.25f);
        float pitchZ = IsCorridorNetwork ? Between(random, 5.40f, 6.50f) : Between(random, 3.15f, 4.25f);
        float margin = IsCorridorNetwork ? 1f : 0f;
        int mainRow = IsCorridorNetwork ? random.Next(1, rows - 1) : -1;
        int atriumColumn = IsCorridorNetwork ? random.Next(1, columns - 1) : -1;
        float mainWidth = IsCorridorNetwork ? Between(random, 2.35f, 3.10f) : 0f;
        var xOffsets = new float[columns];
        var zOffsets = new float[rows];
        if (IsCorridorNetwork)
        {
            for (int x = 0; x < columns; x++)
                xOffsets[x] = Between(random, -0.52f, 0.52f);
            for (int z = 0; z < rows; z++)
                zOffsets[z] = Between(random, -0.52f, 0.52f);
        }
        float width = Mathf.Ceil((columns * pitchX + 2f * margin) / CellSize) * CellSize;
        float length = Mathf.Ceil((rows * pitchZ + 2f * margin) / CellSize) * CellSize;
        Area = new Rect(-width / 2f, 0f, width, length);

        var occupied = new Dictionary<Vector2Int, Room>();
        var sites = new List<Vector2Int>();
        var edges = new List<(Vector2Int from, Vector2Int to)>();
        if (IsCorridorNetwork)
        {
            Room previous = AddRoom(new Vector2Int(0, mainRow));
            for (int x = 1; x < columns; x++)
            {
                Room next = AddRoom(new Vector2Int(x, mainRow));
                ConnectNetwork(previous, next, mainWidth, optional: false);
                previous = next;
            }

            var atriumSite = new Vector2Int(atriumColumn, mainRow);
            int wing = random.Next(2) == 0 ? 1 : -1;
            Connect(occupied[atriumSite], AddRoom(atriumSite + new Vector2Int(0, wing)), optional: false);
            if (random.NextDouble() < 0.70)
                Connect(occupied[atriumSite], AddRoom(atriumSite + new Vector2Int(0, -wing)), optional: false);
        }
        else
        {
            AddRoom(new Vector2Int(random.Next(columns), random.Next(rows)));
        }

        int roomCount = random.Next(IsCorridorNetwork ? columns + 4 : 6,
                                    IsCorridorNetwork ? columns * rows - 1 : columns * rows + 1);
        while (rooms.Count < roomCount)
        {
            edges.Clear();
            foreach (Vector2Int site in sites)
            foreach (Vector2Int direction in Directions)
            {
                Vector2Int next = site + direction;
                if (next.x >= 0 && next.x < columns && next.y >= 0 && next.y < rows &&
                    !occupied.ContainsKey(next))
                    edges.Add((site, next));
            }
            var connection = edges[random.Next(edges.Count)];
            Room from = occupied[connection.from];
            Room to = AddRoom(connection.to);
            // Cardinal tree edges use their reserved grid spans. Diagonal loop edges are optional.
            Connect(from, to, optional: false);
        }

        float extraConnectionProbability = IsCorridorNetwork ? Between(random, 0.24f, 0.58f) : Between(random, 0.08f, 0.72f);
        foreach (Vector2Int site in sites)
        for (int i = 0; i < 2; i++)
        {
            if (occupied.TryGetValue(site + Directions[i], out Room next) &&
                !occupied[site].Neighbors.Contains(next) &&
                random.NextDouble() < extraConnectionProbability)
                Connect(occupied[site], next, optional: false);
        }

        if (IsCorridorNetwork && random.NextDouble() < 0.45)
        {
            int start = random.Next(sites.Count);
            for (int i = 0; i < sites.Count; i++)
            {
                Vector2Int site = sites[(start + i) % sites.Count];
                if (!occupied.TryGetValue(site + new Vector2Int(1, 1), out Room across))
                    continue;
                Connect(occupied[site], across, optional: true);
                if (occupied[site].Neighbors.Contains(across))
                    break;
            }
        }

        foreach (Room via in rooms)
        foreach (Room from in via.Neighbors)
        foreach (Room to in via.Neighbors)
        {
            if (from == to)
                continue;
            Vector2 arrival = corridors[(from, via)].tangent;
            Vector2 exit = corridors[(via, to)].tangent;
            float side = arrival.y * exit.x - arrival.x * exit.y;
            float angle = Mathf.Atan2(side, Vector2.Dot(arrival, exit)) * Mathf.Rad2Deg;
            var intent = angle < -35f ? NavigationIntent.Left :
                         angle > 35f ? NavigationIntent.Right : NavigationIntent.Follow;
            turns.Add((from, via, to, intent));
        }

        BuildWalls();

        Room AddRoom(Vector2Int site)
        {
            Vector2 center = new Vector2((site.x - (columns - 1) * 0.5f) * pitchX,
                                         (site.y + 0.5f) * pitchZ + margin);
            bool atrium = IsCorridorNetwork && site.y == mainRow && site.x == atriumColumn;
            bool lounge = IsCorridorNetwork && (atrium || site.y != mainRow && random.NextDouble() < 0.18);
            center += new Vector2(xOffsets[site.x], zOffsets[site.y]);
            float roomWidth = IsCorridorNetwork
                ? Between(random, lounge ? 3.5f : 1.9f, lounge ? 5.0f : 3.2f)
                : Between(random, 1.80f, pitchX - 0.62f);
            float roomDepth = IsCorridorNetwork
                ? Between(random, lounge ? 3.5f : 1.9f, lounge ? 5.0f : 3.2f)
                : Between(random, 1.80f, pitchZ - 0.62f);
            if (atrium)
            {
                roomWidth = Between(random, 4.2f, 5.2f);
                roomDepth = Between(random, 4.3f, 5.3f);
            }
            else if (IsCorridorNetwork && site.y == mainRow)
            {
                roomWidth = mainWidth;
                roomDepth = mainWidth;
            }
            var room = new Room { Center = center, Area = Centered(center, roomWidth, roomDepth) };
            occupied.Add(site, room);
            sites.Add(site);
            rooms.Add(room);
            free.Add(room.Area);
            RoomAreas.Add(room.Area.width * room.Area.height);
            return room;
        }

        void Connect(Room a, Room b, bool optional)
        {
            if (IsCorridorNetwork)
            {
                ConnectNetwork(a, b, Between(random, 1.28f, 2.10f), optional);
                return;
            }

            a.Neighbors.Add(b);
            b.Neighbors.Add(a);
            bool horizontal = a.Center.y == b.Center.y;
            Vector2 direction = (b.Center - a.Center).normalized;
            Vector2 side = new Vector2(-direction.y, direction.x);
            float aDepth = (horizontal ? a.Area.width : a.Area.height) / 2f;
            float bDepth = (horizontal ? b.Area.width : b.Area.height) / 2f;
            Vector2 entrance = a.Center + direction * aDepth;
            Vector2 exit = b.Center - direction * bDepth;
            Vector2 midpoint = (entrance + exit) / 2f;
            float hallWidth = Between(random, 1.26f, 1.88f);
            CorridorWidths.Add(hallWidth);
            corridors.Add((a, b), (hallWidth, midpoint, direction));
            corridors.Add((b, a), (hallWidth, midpoint, -direction));

            const float jambDepth = 0.16f;
            Corridor(midpoint, Vector2.Distance(entrance, exit) - 2f * jambDepth, hallWidth);
            PlaceDoor(entrance + direction * (jambDepth / 2f));
            PlaceDoor(exit - direction * (jambDepth / 2f));

            void PlaceDoor(Vector2 center)
            {
                float doorWidth = Between(random, 0.98f, Mathf.Min(1.36f, hallWidth - 0.12f));
                float maxOffset = Mathf.Min(0.16f, (hallWidth - doorWidth) / 2f - 0.03f);
                Corridor(center + side * Between(random, -maxOffset, maxOffset),
                         jambDepth + 0.24f, doorWidth);
                DoorWidths.Add(doorWidth);
            }

            void Corridor(Vector2 center, float depth, float width)
            {
                free.Add(Centered(center, horizontal ? depth : width, horizontal ? width : depth));
            }
        }
    }

    private void ConnectNetwork(Room from, Room to, float width, bool optional)
    {
        Vector2 center = (from.Center + to.Center) / 2f;
        Vector2 change = to.Center - from.Center;
        Vector2 tangent = change.normalized;
        bool axisAligned = change.x == 0f || change.y == 0f;
        if (optional && !CanAddAngledBranch())
            return;

        from.Neighbors.Add(to);
        to.Neighbors.Add(from);
        corridors.Add((from, to), (width, center, tangent));
        corridors.Add((to, from), (width, center, -tangent));
        CorridorWidths.Add(width);

        sweeps.Add(new Sweep(from.Center, to.Center, width / 2f, from, to));
        if (axisAligned)
            free.Add(Centered(center, Mathf.Abs(change.x) + width, Mathf.Abs(change.y) + width));

        bool CanAddAngledBranch()
        {
            for (int i = 1; i < 24; i++)
            {
                Vector2 point = Vector2.Lerp(from.Center, to.Center, i / 24f);
                foreach (Room room in rooms)
                {
                    if (room == from || room == to)
                        continue;
                    Rect box = room.Area;
                    float dx = Mathf.Max(0f, Mathf.Max(box.xMin - point.x, point.x - box.xMax));
                    float dz = Mathf.Max(0f, Mathf.Max(box.yMin - point.y, point.y - box.yMax));
                    if (dx * dx + dz * dz < (width / 2f + 0.28f) * (width / 2f + 0.28f))
                        return false;
                }
                foreach (Sweep other in sweeps)
                {
                    if (other.Touches(from) || other.Touches(to))
                        continue;
                    float separation = other.Radius + width / 2f + 0.32f;
                    if (DistanceSquared(point, other.Start, other.End) < separation * separation)
                        return false;
                }
            }
            return true;
        }
    }

    private static float DistanceSquared(Vector2 point, Vector2 start, Vector2 end)
    {
        Vector2 delta = end - start;
        float fraction = delta.sqrMagnitude > 0f ? Vector2.Dot(point - start, delta) / delta.sqrMagnitude : 0f;
        return (point - Vector2.Lerp(start, end, fraction)).sqrMagnitude;
    }

    public Task CreateTask(int seed, int taskIndex)
    {
        // Adjacent tasks in each pair share their approach and obstacle.
        int pair = taskIndex / 2;
        var random = new Random(unchecked(seed * 3571 + pair));
        bool blocked = taskIndex == 9;
        var approach = ChooseApproach(random, taskIndex);
        int exitIndex = taskIndex % 2 == 0 ? 0 : approach.Count - 1;
        var move = approach[exitIndex];
        Vector2 heading = (move.via.Center - move.from.Center).normalized;
        var corridor = corridors[(move.from, move.via)];
        Vector2 perpendicular = new Vector2(corridor.tangent.y, -corridor.tangent.x);
        float width = corridor.width;
        Vector2 start = Vector2.Lerp(move.from.Center, move.via.Center, Between(random, 0.04f, 0.18f))
                        + perpendicular * Between(random, -0.23f, 0.23f);
        float yaw = Mathf.Atan2(heading.x, heading.y) * Mathf.Rad2Deg + Between(random, -18f, 18f);
        Vector2 barrier = corridor.center;
        Rect? obstacle = null;
        if (blocked)
        {
            Vector2 axis = corridor.tangent;
            const float depth = 0.24f;
            float barrierWidth = Mathf.Abs(axis.x) * depth + Mathf.Abs(axis.y) * (width + 0.24f);
            float barrierLength = Mathf.Abs(axis.y) * depth + Mathf.Abs(axis.x) * (width + 0.24f);
            obstacle = Centered(barrier, barrierWidth, barrierLength);
            float extent = (Mathf.Abs(axis.x) * barrierWidth + Mathf.Abs(axis.y) * barrierLength) / 2f;
            Vector2 stop = barrier - axis * (extent + 0.55f);
            return new Task(start, stop, stop, yaw, NavigationIntent.Follow, obstacle, true);
        }

        if (pair % 3 != 0)
        {
            float size = Between(random, 0.20f, 0.30f);
            float side = random.Next(2) == 0 ? -1f : 1f;
            obstacle = Centered(barrier + perpendicular * side * (width / 2f - size / 2f - 0.03f),
                                size, size);
        }
        Vector2 goal = Vector2.Lerp(move.via.Center, move.to.Center, Between(random, 0.72f, 0.94f)) +
                       new Vector2(Between(random, -0.12f, 0.12f), Between(random, -0.12f, 0.12f));
        return new Task(start, move.via.Center, goal, yaw, move.intent, obstacle, false);
    }

    private List<(Room from, Room via, Room to, NavigationIntent intent)> ChooseApproach(Random random, int taskIndex)
    {
        bool pairedDecision = taskIndex < 8;
        int start = random.Next(turns.Count);
        var sameApproach = new List<(Room from, Room via, Room to, NavigationIntent intent)>();
        for (int i = 0; i < turns.Count; i++)
        {
            var candidate = turns[(start + i) % turns.Count];
            sameApproach.Clear();
            foreach (var move in turns)
                if (move.from == candidate.from && move.via == candidate.via)
                    sameApproach.Add(move);
            if (IsCorridorNetwork)
            {
                var counts = new int[3];
                foreach (var move in sameApproach)
                    counts[(int)move.intent]++;
                sameApproach.RemoveAll(move => counts[(int)move.intent] > 1);
            }
            if (sameApproach.Count >= (pairedDecision ? 2 : 1))
                return sameApproach;
        }
        if (IsCorridorNetwork)
        {
            for (int i = 0; i < turns.Count; i++)
            {
                var candidate = turns[(start + i) % turns.Count];
                int matches = 0;
                foreach (var move in turns)
                    if (move.from == candidate.from && move.via == candidate.via && move.intent == candidate.intent)
                        matches++;
                if (matches == 1)
                    return new List<(Room from, Room via, Room to, NavigationIntent intent)> { candidate };
            }
            throw new InvalidOperationException("This floor plan has no uniquely described exit.");
        }
        return new List<(Room from, Room via, Room to, NavigationIntent intent)> { turns[start] };
    }

    private void BuildWalls()
    {
        int nx = Mathf.RoundToInt(Area.width / CellSize);
        int nz = Mathf.RoundToInt(Area.height / CellSize);
        var solid = new bool[nx, nz];
        for (int z = 0; z < nz; z++)
        for (int x = 0; x < nx; x++)
            solid[x, z] = true;

        foreach (Rect room in free)
            Carve(room, room.Contains);

        foreach (Sweep sweep in sweeps)
        {
            if (sweep.Start.x == sweep.End.x || sweep.Start.y == sweep.End.y)
                continue; // Axis-aligned links already use an exact rectangle.
            Rect bounds = Rect.MinMaxRect(Mathf.Min(sweep.Start.x, sweep.End.x) - sweep.Radius,
                                         Mathf.Min(sweep.Start.y, sweep.End.y) - sweep.Radius,
                                         Mathf.Max(sweep.Start.x, sweep.End.x) + sweep.Radius,
                                         Mathf.Max(sweep.Start.y, sweep.End.y) + sweep.Radius);
            Carve(bounds, point => DistanceSquared(point, sweep.Start, sweep.End) < sweep.Radius * sweep.Radius);
        }

        void Carve(Rect bounds, Predicate<Vector2> inside)
        {
            int xStart = Mathf.Max(0, Mathf.FloorToInt((bounds.xMin - Area.xMin) / CellSize));
            int xEnd = Mathf.Min(nx, Mathf.CeilToInt((bounds.xMax - Area.xMin) / CellSize));
            int zStart = Mathf.Max(0, Mathf.FloorToInt(bounds.yMin / CellSize));
            int zEnd = Mathf.Min(nz, Mathf.CeilToInt(bounds.yMax / CellSize));
            for (int z = zStart; z < zEnd; z++)
            for (int x = xStart; x < xEnd; x++)
            {
                Vector2 point = new Vector2(Area.xMin + (x + 0.5f) * CellSize, (z + 0.5f) * CellSize);
                if (inside(point))
                    solid[x, z] = false;
            }
        }

        for (int z = 0; z < nz; z++)
        for (int x = 0; x < nx; x++)
        {
            if (!solid[x, z])
                continue;
            int right = x + 1;
            while (right < nx && solid[right, z])
                right++;
            int top = z + 1;
            while (top < nz)
            {
                int column = x;
                while (column < right && solid[column, top])
                    column++;
                if (column != right)
                    break;
                top++;
            }
            for (int y = z; y < top; y++)
            for (int column = x; column < right; column++)
                solid[column, y] = false;
            Walls.Add(new Wall(new Rect(Area.xMin + x * CellSize, z * CellSize,
                                        (right - x) * CellSize, (top - z) * CellSize)));
        }
    }

    private static Rect Centered(Vector2 center, float width, float length)
        => new Rect(center.x - width / 2f, center.y - length / 2f, width, length);

    private static float Between(Random random, float min, float max)
        => min + (max - min) * (float)random.NextDouble();
}

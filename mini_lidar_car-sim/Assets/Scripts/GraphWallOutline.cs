using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

internal sealed class GraphWallOutline
{
    private const float HalfWidth = 0.0225f;
    private const float MillimetersPerMeter = 1000f;
    private readonly GameObject drawable;
    private readonly Mesh mesh;

    public GraphWallOutline(Material material)
    {
        drawable = new GameObject("Graph wall outlines", typeof(MeshFilter), typeof(MeshRenderer));
        mesh = new Mesh { name = "Graph wall outlines" };
        drawable.GetComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer renderer = drawable.GetComponent<MeshRenderer>();
        renderer.sharedMaterial = material;
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        var color = new MaterialPropertyBlock();
        color.SetColor("_BaseColor", new Color(0.20f, 0.20f, 0.20f));
        renderer.SetPropertyBlock(color);
    }

    public void Show(IReadOnlyList<Rect> walls, float height)
    {
        // Both the PDF and the scene draw interfaces between occupied and free cells.
        var edgesX = new SortedSet<int>();
        var edgesZ = new SortedSet<int>();
        foreach (Rect wall in walls)
        {
            edgesX.Add(Millimeters(wall.xMin));
            edgesX.Add(Millimeters(wall.xMax));
            edgesZ.Add(Millimeters(wall.yMin));
            edgesZ.Add(Millimeters(wall.yMax));
        }

        var x = new List<int>(edgesX);
        var z = new List<int>(edgesZ);
        var occupied = new bool[x.Count - 1, z.Count - 1];
        foreach (Rect wall in walls)
        {
            int x0 = x.BinarySearch(Millimeters(wall.xMin));
            int x1 = x.BinarySearch(Millimeters(wall.xMax));
            int z0 = z.BinarySearch(Millimeters(wall.yMin));
            int z1 = z.BinarySearch(Millimeters(wall.yMax));
            for (int ix = x0; ix < x1; ix++)
            for (int iz = z0; iz < z1; iz++)
                occupied[ix, iz] = true;
        }

        var vertices = new List<Vector3>();
        var triangles = new List<int>();
        // The exterior extent of the exported grid is outside the building.
        for (int ix = 1; ix < x.Count - 1; ix++)
        {
            int start = -1;
            for (int iz = 0; iz < z.Count; iz++)
            {
                bool boundary = iz < z.Count - 1 && occupied[ix - 1, iz] != occupied[ix, iz];
                if (boundary && start < 0)
                    start = iz;
                if (!boundary && start >= 0)
                {
                    AddEdge(x[ix], z[start], x[ix], z[iz]);
                    start = -1;
                }
            }
        }
        for (int iz = 1; iz < z.Count - 1; iz++)
        {
            int start = -1;
            for (int ix = 0; ix < x.Count; ix++)
            {
                bool boundary = ix < x.Count - 1 && occupied[ix, iz - 1] != occupied[ix, iz];
                if (boundary && start < 0)
                    start = ix;
                if (!boundary && start >= 0)
                {
                    AddEdge(x[start], z[iz], x[ix], z[iz]);
                    start = -1;
                }
            }
        }

        mesh.Clear();
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        drawable.SetActive(true);

        void AddEdge(int x0, int z0, int x1, int z1)
        {
            var start = new Vector3(x0 / MillimetersPerMeter, height, z0 / MillimetersPerMeter);
            var end = new Vector3(x1 / MillimetersPerMeter, height, z1 / MillimetersPerMeter);
            Vector3 cap = (end - start).normalized * HalfWidth;
            Vector3 side = new Vector3(-cap.z, 0f, cap.x);
            int first = vertices.Count;
            vertices.Add(start - cap - side);
            vertices.Add(start - cap + side);
            vertices.Add(end + cap + side);
            vertices.Add(end + cap - side);
            triangles.Add(first);
            triangles.Add(first + 1);
            triangles.Add(first + 2);
            triangles.Add(first);
            triangles.Add(first + 2);
            triangles.Add(first + 3);
        }
    }

    public void Hide() => drawable.SetActive(false);

    public void Destroy()
    {
        Object.Destroy(drawable);
        Object.Destroy(mesh);
    }

    private static int Millimeters(float meters) => Mathf.RoundToInt(meters * MillimetersPerMeter);
}

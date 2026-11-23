using System;
using System.Globalization;
using System.IO;

namespace UnityEngine
{
    internal struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
    }

    internal static class Mathf
    {
        public static float Min(float a, float b) => Math.Min(a, b);
        public static float Max(float a, float b) => Math.Max(a, b);
    }
}

internal enum NavigationIntent { Follow, Left, Right, Stop }

internal static class Runner
{
    private static void Main(string[] args)
    {
        var model = new LowerPolicyModel(File.ReadAllBytes(args[0]));
        var culture = CultureInfo.InvariantCulture;
        foreach (string row in File.ReadLines(args[1]))
        {
            string[] fields = row.Split(',');
            var ranges = new float[360];
            for (int i = 0; i < ranges.Length; i++)
                ranges[i] = float.Parse(fields[i + 3], culture);
            var context = new UnityEngine.Vector2(float.Parse(fields[1], culture), float.Parse(fields[2], culture));
            var output = model.Predict(ranges, (NavigationIntent)int.Parse(fields[0], culture), context);
            Console.WriteLine(output.x.ToString("R", culture) + "," + output.y.ToString("R", culture));
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

internal sealed class ReferenceAvoidanceLayout
{
    public int Seed { get; private set; }
    public int Mission { get; private set; }
    public Rect Area { get; private set; }
    public Rect Box { get; private set; }
    public NavigationIntent Intent { get; private set; }
    public readonly List<Rect> Walls = new List<Rect>();
    public readonly List<Vector2> Reference = new List<Vector2>();
    public readonly List<Vector2> Detour = new List<Vector2>();

    public static ReferenceAvoidanceLayout Load(int number)
    {
        var geometry = Resources.Load<TextAsset>("AvoidanceReferenceLayouts");
        if (geometry == null)
            throw new FileNotFoundException("Missing Resources/AvoidanceReferenceLayouts.csv");

        var layout = new ReferenceAvoidanceLayout();
        using var reader = new StringReader(geometry.text);
        int index = 0;
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            string[] entry = line.Split(',');
            if (entry[0] == "map")
            {
                if (++index > number)
                    break;
                if (index == number)
                {
                    layout.Seed = int.Parse(entry[1], CultureInfo.InvariantCulture);
                    layout.Mission = int.Parse(entry[2], CultureInfo.InvariantCulture);
                    layout.Intent = Enum.Parse<NavigationIntent>(entry[4]);
                    layout.Area = ReadRect(entry, 5);
                }
                continue;
            }
            if (index != number)
                continue;
            switch (entry[0])
            {
                case "wall": layout.Walls.Add(ReadRect(entry, 1)); break;
                case "box": layout.Box = ReadRect(entry, 1); break;
                case "reference": layout.Reference.Add(ReadPoint(entry)); break;
                case "detour": layout.Detour.Add(ReadPoint(entry)); break;
            }
        }
        if (layout.Walls.Count == 0 || layout.Box.width <= 0 || layout.Reference.Count < 2 || layout.Detour.Count < 2)
            throw new InvalidDataException($"PDF layout {number} is unavailable or incomplete.");
        return layout;
    }

    private static Rect ReadRect(string[] row, int offset) =>
        new Rect(Number(row[offset]), Number(row[offset + 1]), Number(row[offset + 2]), Number(row[offset + 3]));

    private static Vector2 ReadPoint(string[] row) => new Vector2(Number(row[1]), Number(row[2]));

    private static float Number(string value) => float.Parse(value, CultureInfo.InvariantCulture);
}

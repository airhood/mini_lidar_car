using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;

public static class RecoveryScheduleAudit
{
    public static void Run()
    {
        var args = Environment.GetCommandLineArgs();
        string source = args[Array.IndexOf(args, "-auditRecordingPath") + 1];
        string target = args[Array.IndexOf(args, "-auditScheduleOutput") + 1];
        var pattern = new Regex(@"^completed-teacher-v12-(network|rooms)-[^-]+-(clear|box-approach|box-exit)-task(\d+)-seed(\d+)-");
        using var output = new StreamWriter(target);
        output.WriteLine("file,family,condition,mission,seed,scheduled_seconds,direction");
        var records = Directory.GetFiles(source, "*.csv");
        Array.Sort(records, StringComparer.Ordinal);
        foreach (string file in records)
        {
            string name = Path.GetFileName(file);
            var match = pattern.Match(name);
            if (!match.Success)
                continue;
            int mission = int.Parse(match.Groups[3].Value, CultureInfo.InvariantCulture);
            int seed = int.Parse(match.Groups[4].Value, CultureInfo.InvariantCulture);
            var rng = new Random(unchecked(seed * 397 + mission / 2));
            string at = "";
            string direction = "";
            if (rng.NextDouble() < 0.6)
            {
                at = (0.7f + (float)rng.NextDouble() * 6f).ToString("R", CultureInfo.InvariantCulture);
                direction = (rng.Next(2) == 0 ? -1 : 1).ToString(CultureInfo.InvariantCulture);
            }
            output.WriteLine($"{name},{match.Groups[1]},{match.Groups[2]},{mission},{seed},{at},{direction}");
        }
    }
}

using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

public enum NavigationIntent
{
    Unspecified = -1,
    Follow = 0,
    Left = 1,
    Right = 2,
    Stop = 3
}

[RequireComponent(typeof(RearDrive), typeof(PlanarLidar), typeof(Rigidbody))]
public sealed class DriveRecorder : MonoBehaviour
{
    private static readonly CultureInfo CsvCulture = CultureInfo.InvariantCulture;

    private RearDrive drive;
    private PlanarLidar lidar;
    private Rigidbody body;
    private StreamWriter writer;
    private readonly StringBuilder row = new StringBuilder(4096);
    private int collisionCount;
    private int sampleCount;
    private string path;
    private bool automatic;
    private string progressPrefix;

    private Func<(Vector2 goal, Vector2 waypoint, bool hasCheckpoint, bool passed)> describePilot;
    public Vector2 PilotGoalContext { get; private set; }
    public Vector2 PilotWaypointContext { get; private set; }

    public NavigationIntent Intent { get; set; } = NavigationIntent.Unspecified;

    public int CollisionCount => collisionCount;
    public bool IsRecording => writer != null;

    private void Awake()
    {
        drive = GetComponent<RearDrive>();
        lidar = GetComponent<PlanarLidar>();
        body = GetComponent<Rigidbody>();
    }

    private void OnEnable()
    {
        lidar.ScanReady += RecordScan;
    }

    private void OnDisable()
    {
        lidar.ScanReady -= RecordScan;
        StopRecording(automatic ? "interrupted" : null);
    }

    private void Update()
    {
        if (automatic || Keyboard.current?.rKey.wasPressedThisFrame != true)
            return;

        if (writer == null)
            StartRecording();
        else
            StopRecording();
    }

    public void StartRecording(string label = "drive", bool automaticRun = false,
                               string outputDirectory = null, string progress = null,
                               Func<(Vector2 goal, Vector2 waypoint, bool hasCheckpoint, bool passed)> pilot = null)
    {
        if (writer != null)
            throw new InvalidOperationException("A drive recording is already open.");

        string directory = outputDirectory ?? DefaultOutputDirectory();
        Directory.CreateDirectory(directory);
        path = Path.Combine(directory, $"{label}-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.csv");
        writer = new StreamWriter(path);
        automatic = automaticRun;
        describePilot = pilot;
        progressPrefix = string.IsNullOrEmpty(progress) ? "" : progress + " ";
        collisionCount = 0;
        sampleCount = 0;

        row.Clear().Append("time_s,left_cmd,right_cmd,x_m,z_m,heading_deg,collision_count,intent_id");
        for (int i = 0; i < lidar.Ranges.Length; i++)
            row.Append(",range_").Append(i.ToString(CsvCulture)).Append("_m");
        if (describePilot != null)
            row.Append(",goal_x_m,goal_z_m,waypoint_x_m,waypoint_z_m,has_box_checkpoint,passed_box_checkpoint");
        writer.WriteLine(row.ToString());
        Debug.Log($"{progressPrefix}Recording drive: {DisplayPath(automaticRun)}", this);
    }

    private string DisplayPath(bool automaticRun) => automaticRun ? Path.GetFileName(path) : path;

    internal static string DefaultOutputDirectory()
    {
#if UNITY_EDITOR
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "data", "drive-recordings"));
#else
        return Path.Combine(Application.persistentDataPath, "drive-recordings");
#endif
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (writer != null)
            collisionCount++;
    }

    private void RecordScan()
    {
        if (writer == null)
            return;

        Vector2 command = drive.MotorCommands;
        Vector3 position = body.position;
        Vector3 forward = body.rotation * Vector3.forward;
        float heading = Mathf.Atan2(-forward.x, forward.z) * Mathf.Rad2Deg;

        row.Clear().Append(lidar.ScanTime.ToString("R", CsvCulture));
        Append(command.x);
        Append(command.y);
        Append(position.x);
        Append(position.z);
        Append(heading);
        row.Append(',').Append(collisionCount.ToString(CsvCulture));
        row.Append(',').Append(((int)Intent).ToString(CsvCulture));
        foreach (float range in lidar.Ranges)
        {
            if (float.IsPositiveInfinity(range))
                row.Append(",inf");
            else
                Append(range);
        }

        if (describePilot != null)
        {
            var context = describePilot();
            Vector2 planarPosition = new Vector2(position.x, position.z);
            float radians = heading * Mathf.Deg2Rad;
            Vector2 forwardAxis = new Vector2(-Mathf.Sin(radians), Mathf.Cos(radians));
            Vector2 rightAxis = new Vector2(forwardAxis.y, -forwardAxis.x);
            PilotGoalContext = LocalTarget(context.goal, 15f);
            PilotWaypointContext = LocalTarget(context.waypoint, 1f);

            Vector2 LocalTarget(Vector2 target, float scale)
            {
                Vector2 offset = target - planarPosition;
                return new Vector2(Mathf.Clamp(Vector2.Dot(offset, forwardAxis) / scale, -1f, 1f),
                                   Mathf.Clamp(Vector2.Dot(offset, rightAxis) / scale, -1f, 1f));
            }
            Append(context.goal.x);
            Append(context.goal.y);
            Append(context.waypoint.x);
            Append(context.waypoint.y);
            row.Append(context.hasCheckpoint ? ",1" : ",0").Append(context.passed ? ",1" : ",0");
        }

        writer.WriteLine(row.ToString());
        sampleCount++;
    }

    private void Append(float value)
    {
        row.Append(',').Append(value.ToString("R", CsvCulture));
    }

    public void StopRecording(string outcome = null)
    {
        if (writer == null)
            return;

        writer.Dispose();
        writer = null;
        describePilot = null;
        bool wasAutomatic = automatic;
        automatic = false;
        Intent = NavigationIntent.Unspecified;
        if (outcome != null)
        {
            string resultPath = Path.Combine(Path.GetDirectoryName(path),
                $"{outcome}-{Path.GetFileName(path)}");
            File.Move(path, resultPath);
            path = resultPath;
        }
        Debug.Log($"{progressPrefix}Saved {sampleCount} scans, {collisionCount} collision entries: {DisplayPath(wasAutomatic)}", this);
    }
}

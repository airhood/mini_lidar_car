using System;
using System.IO;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

internal sealed class PolicyFrameCapture : IDisposable
{
    private readonly int width;
    private readonly int height;
    private readonly Renderer vehicle;
    private readonly Rigidbody followBody;
    private readonly LineRenderer requestedDirection;
    private readonly LineRenderer motorCommand;
    private const int TurnBinCount = 41;
    private const float TurnBinWidth = 0.05f;
    private const float TurnAngleScale = 100f;
    private readonly LineRenderer[] noiseBand;
    private readonly int[] turnCounts;
    private readonly int noiseSampleCount;
    private StreamWriter noiseSamples;
    private Vector3 requestedHeading;
    private Vector2 lastCommands;
    private readonly Vector3 originalPosition;
    private readonly Quaternion originalRotation;
    private readonly bool originalOrthographic;
    private readonly float originalFieldOfView;
    private Vector3 followPosition;
    private Vector3 followHeading;
    private readonly MaterialPropertyBlock vehicleAppearance;
    private const int FramesPerSecond = 25;

    private readonly Camera camera;
    private readonly RenderTexture target;
    private readonly Texture2D pixels;
    private readonly RenderPipeline.StandardRequest request;
    private readonly string directory;
    private readonly bool cameraWasEnabled;
    private string episodePath;
    private double episodeStart;
    private int frameIndex;

    public PolicyFrameCapture(Camera camera, string directory, Renderer vehicle = null,
                              Rigidbody followBody = null, bool showSteering = false,
                              int noiseSampleCount = 0)
    {
        this.camera = camera;
        this.directory = directory;
        this.vehicle = vehicle;
        this.followBody = followBody;
        this.noiseSampleCount = noiseSampleCount;
        width = followBody ? 1280 : (vehicle ? 768 : 600);
        height = followBody ? 720 : (vehicle ? 768 : 900);
        originalPosition = camera.transform.position;
        originalRotation = camera.transform.rotation;
        originalOrthographic = camera.orthographic;
        originalFieldOfView = camera.fieldOfView;
        if (followBody)
        {
            camera.orthographic = false;
            camera.fieldOfView = 57f;
        }
        if (showSteering)
        {
            requestedDirection = CreateArrow("Policy direction input", new Color32(218, 136, 22, 255), 0.040f, 0.20f);
            motorCommand = CreateArrow("Policy motor command", new Color32(0, 118, 115, 255), 0.058f, 0.18f);
        }
        if (noiseSampleCount > 0)
        {
            noiseBand = new LineRenderer[TurnBinCount];
            turnCounts = new int[TurnBinCount];
            for (int i = 0; i < noiseBand.Length; i++)
            {
                noiseBand[i] = CreateFloorLine("LiDAR-noise steering bin");
                noiseBand[i].positionCount = 4;
                noiseBand[i].widthMultiplier = 0.26f;
            }
        }
        if (vehicle)
        {
            vehicleAppearance = new MaterialPropertyBlock();
            vehicle.GetPropertyBlock(vehicleAppearance);
            var highlight = new MaterialPropertyBlock();
            vehicle.GetPropertyBlock(highlight);
            highlight.SetColor("_BaseColor", new Color(0.07f, 0.34f, 0.78f));
            vehicle.SetPropertyBlock(highlight);
        }
        cameraWasEnabled = camera.enabled;
        camera.enabled = false;
        camera.aspect = (float)width / height;
        target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        pixels = new Texture2D(width, height, TextureFormat.RGB24, false);
        request = new RenderPipeline.StandardRequest { destination = target };
        Directory.CreateDirectory(directory);
    }

    public void Begin(string label)
    {
        if (!RenderPipeline.SupportsRenderRequest(camera, request))
            throw new InvalidOperationException("The active render pipeline does not support frame capture.");
        episodePath = Path.Combine(directory, label);
        Directory.CreateDirectory(episodePath);
        episodeStart = Time.fixedTimeAsDouble;
        frameIndex = 0;
        ClearSteering();
        if (noiseBand != null)
        {
            noiseSamples = new StreamWriter(Path.Combine(episodePath, "steering-samples.csv"));
            noiseSamples.Write("time_s,nominal_turn");
            for (int i = 0; i < noiseSampleCount; i++)
                noiseSamples.Write($",sample_{i:D2}");
            noiseSamples.WriteLine();
        }
    }

    public void Sample()
    {
        if (frameIndex == 0 || Time.fixedTimeAsDouble + 1e-6 >= episodeStart + (double)frameIndex / FramesPerSecond)
            WriteFrame();
    }

    public void Finish(string outcome)
    {
        WriteFrame();
        noiseSamples?.Dispose();
        noiseSamples = null;
        File.WriteAllText(Path.Combine(episodePath, "outcome.txt"), outcome);
    }

    private void UpdateFollowCamera()
    {
        Vector3 heading = Vector3.ProjectOnPlane(followBody.rotation * Vector3.forward, Vector3.up).normalized;
        if (frameIndex == 0)
        {
            followPosition = followBody.position;
            followHeading = heading;
        }
        else
        {
            followPosition = Vector3.Lerp(followPosition, followBody.position, 0.34f);
            followHeading = Vector3.Slerp(followHeading, heading, 0.18f).normalized;
        }

        Vector3 position = followPosition + Vector3.up * 1.65f - followHeading * 2.10f;
        Vector3 lookAt = followPosition + followHeading * 0.68f;
        camera.transform.SetPositionAndRotation(position, Quaternion.LookRotation(lookAt - position, Vector3.up));
    }

    public void SetSteering(Vector2 context, Vector2 commands)
    {
        if (requestedDirection == null)
            return;

        Vector3 local = new Vector3(context.y, 0f, context.x);
        requestedHeading = Vector3.ProjectOnPlane(followBody.rotation * local, Vector3.up).normalized;
        lastCommands = commands;
        requestedDirection.enabled = requestedHeading.sqrMagnitude > 0.001f;
        motorCommand.enabled = commands.sqrMagnitude > 0.0001f;
    }

    public void SetSteeringSamples(double scanTime, float[] sampledTurns)
    {
        Array.Clear(turnCounts, 0, turnCounts.Length);
        noiseSamples.Write(scanTime.ToString("R", CultureInfo.InvariantCulture));
        noiseSamples.Write(',');
        noiseSamples.Write(((lastCommands.x - lastCommands.y) * 0.5f).ToString("R", CultureInfo.InvariantCulture));
        foreach (float turn in sampledTurns)
        {
            int bin = Mathf.Clamp(Mathf.RoundToInt(turn / TurnBinWidth) + TurnBinCount / 2, 0, TurnBinCount - 1);
            turnCounts[bin]++;
            noiseSamples.Write(',');
            noiseSamples.Write(turn.ToString("R", CultureInfo.InvariantCulture));
        }
        noiseSamples.WriteLine();

        for (int i = 0; i < noiseBand.Length; i++)
        {
            float frequency = (float)turnCounts[i] / noiseSampleCount;
            noiseBand[i].enabled = turnCounts[i] > 0;
            noiseBand[i].sharedMaterial.color = Color.Lerp(new Color32(214, 201, 239, 255),
                                                          new Color32(76, 32, 143, 255), frequency);
        }
    }

    public void ClearSteering()
    {
        if (requestedDirection == null)
            return;

        requestedDirection.enabled = false;
        motorCommand.enabled = false;
        if (noiseBand != null)
            foreach (LineRenderer bin in noiseBand)
                bin.enabled = false;
    }

    private static LineRenderer CreateArrow(string name, Color color, float width, float tipStart)
    {
        LineRenderer arrow = CreateFloorLine(name);
        arrow.sharedMaterial.color = color;
        arrow.positionCount = 21;
        arrow.widthMultiplier = width;
        arrow.widthCurve = new AnimationCurve(
            new Keyframe(0f, 1f), new Keyframe(1f - tipStart - 0.01f, 1f),
            new Keyframe(1f - tipStart, 3f), new Keyframe(1f, 0f));
        return arrow;
    }

    private static LineRenderer CreateFloorLine(string name)
    {
        var line = new GameObject(name).AddComponent<LineRenderer>();
        line.sharedMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        line.shadowCastingMode = ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.alignment = LineAlignment.TransformZ;
        line.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
        line.enabled = false;
        return line;
    }

    private void UpdateNoiseBand(Vector3 origin, Vector3 heading)
    {
        for (int i = 0; i < noiseBand.Length; i++)
        {
            LineRenderer bin = noiseBand[i];
            if (!bin.enabled)
                continue;

            float center = (i - TurnBinCount / 2) * TurnBinWidth * TurnAngleScale;
            float halfWidth = 0.48f * TurnBinWidth * TurnAngleScale;
            for (int point = 0; point < bin.positionCount; point++)
            {
                float angle = center + Mathf.Lerp(-halfWidth, halfWidth, (float)point / (bin.positionCount - 1));
                Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * heading;
                bin.SetPosition(point, origin + 1.28f * direction);
            }
        }
    }

    private void UpdateSteeringArrows()
    {
        Vector3 position = followBody.position;
        Vector3 origin = new Vector3(position.x, 0.045f, position.z);
        if (requestedDirection.enabled)
        {
            for (int i = 0; i < requestedDirection.positionCount; i++)
            {
                float fraction = (float)i / (requestedDirection.positionCount - 1);
                requestedDirection.SetPosition(i, origin + requestedHeading * Mathf.Lerp(0.27f, 1.58f, fraction));
            }
        }

        Vector3 heading = Vector3.ProjectOnPlane(followBody.rotation * Vector3.forward, Vector3.up).normalized;
        if (noiseBand != null)
            UpdateNoiseBand(origin - 0.02f * Vector3.up, heading);
        if (motorCommand.enabled)
        {
            float forward = (lastCommands.x + lastCommands.y) * 0.5f;
            float yaw = (lastCommands.x - lastCommands.y) * 0.5f;
            float direction = forward < -0.02f ? -1f : 1f;
            float length = 0.28f + 0.67f * Mathf.Abs(forward) + 0.23f * Mathf.Abs(yaw);
            Vector3 point = origin + 0.045f * Vector3.up + 0.23f * direction * heading;
            motorCommand.SetPosition(0, point);
            for (int i = 1; i < motorCommand.positionCount; i++)
            {
                float fraction = (i - 0.5f) / (motorCommand.positionCount - 1);
                Vector3 tangent = Quaternion.AngleAxis(135f * yaw * fraction, Vector3.up) * heading;
                point += tangent * (direction * length / (motorCommand.positionCount - 1));
                motorCommand.SetPosition(i, point);
            }
        }
    }

    private static void DestroyFloorLine(LineRenderer arrow)
    {
        UnityEngine.Object.Destroy(arrow.sharedMaterial);
        UnityEngine.Object.Destroy(arrow.gameObject);
    }

    private void WriteFrame()
    {
        if (followBody)
            UpdateFollowCamera();
        if (requestedDirection != null)
            UpdateSteeringArrows();
        var previousTarget = RenderTexture.active;
        try
        {
            RenderPipeline.SubmitRenderRequest(camera, request);
            RenderTexture.active = target;
            pixels.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            File.WriteAllBytes(Path.Combine(episodePath, $"frame-{frameIndex:D05}.jpg"), pixels.EncodeToJPG(93));
            frameIndex++;
        }
        finally
        {
            RenderTexture.active = previousTarget;
        }
    }

    public void Dispose()
    {
        if (vehicle)
            vehicle.SetPropertyBlock(vehicleAppearance);
        if (followBody)
        {
            camera.transform.SetPositionAndRotation(originalPosition, originalRotation);
            camera.orthographic = originalOrthographic;
            camera.fieldOfView = originalFieldOfView;
        }
        if (requestedDirection != null)
        {
            DestroyFloorLine(requestedDirection);
            DestroyFloorLine(motorCommand);
        }
        if (noiseBand != null)
        {
            noiseSamples?.Dispose();
            foreach (LineRenderer bin in noiseBand)
                DestroyFloorLine(bin);
        }
        camera.ResetAspect();
        camera.enabled = cameraWasEnabled;
        target.Release();
        UnityEngine.Object.Destroy(target);
        UnityEngine.Object.Destroy(pixels);
    }
}

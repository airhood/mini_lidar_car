using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

[DefaultExecutionOrder(-100)]
[RequireComponent(typeof(RearDrive), typeof(ManualDrive), typeof(DriveRecorder))]
public sealed class ObstacleTeacher : MonoBehaviour
{
    private const float Clearance = 0.40f;
    private const float ArrivalDistance = 0.07f;
    private const float LookAhead = 0.32f;
    private const float FinishZ = 5.2f;
    private const float SettleTime = 0.45f;
    private const float Timeout = 65f;
    private const int PreviewCount = 13;
    private const int OpenMissionsPerMap = 9;
    private const int BlockedMission = OpenMissionsPerMap * 2;
    private const int MissionsPerMap = BlockedMission + 1;
    private const int HallScenarios = 2 * OpenAvoidanceLayout.MissionsPerSeed;

    private static readonly string[] Labels =
    {
        "empty", "left-only", "right-only", "both-left", "both-right", "blocked",
        "staggered", "offset-doorway", "clutter",
        "junction-clear-left", "junction-clear-right", "junction-box-left", "junction-box-right"
    };

    [SerializeField, Min(0)] private int firstSeed = 1121;
    [SerializeField, Range(1, 500)] private int seedCount = 3;
    [SerializeField] private bool evaluationScene;
    [SerializeField] private TextAsset evaluationWeights;
    [SerializeField, Range(1, 100)] private int pdfCase = 3;
    [SerializeField, Range(1f, 12f)] private float collectionTimeScale = 8f;
    [SerializeField] private BoxCollider obstacle = null;
    [SerializeField] private BoxCollider leftWall = null;
    [SerializeField] private BoxCollider rightWall = null;
    [SerializeField] private BoxCollider backWall = null;
    [SerializeField] private BoxCollider endWall = null;
    [SerializeField] private Transform floor = null;
    [SerializeField] private Camera overview = null;

    private readonly List<Vector2> route = new List<Vector2>();
    private readonly List<Bounds> obstacles = new List<Bounds>();
    private readonly BoxCollider[] layoutBoxes = new BoxCollider[8];
    private readonly List<BoxCollider> graphBoxes = new List<BoxCollider>();
    private GraphFloorPlan currentMap;
    private Rect pairedObstacle;
    private List<Vector2> pairedReference;
    private List<Vector2> pairedDetour;
    private bool pairedAtExit;
    private OpenAvoidanceLayout pairedHall;
    private ReferenceAvoidanceLayout pdfPreview;
    private bool hallsOnly;
    private bool waypointPilot;
    private bool waypointInput;
    private bool sideSwap;
    private bool directionChoice;
    private bool directionChoiceApproach;
    private Vector2 choiceWaypoint;
    private float choiceSwitchZ;
    private bool directionTransfer;
    private bool staggeredOnly;
    private bool directionStress;
    private bool largeDirection;
    private bool doorwayArena;
    private int doorwayCaptureTask = -1;
    private bool doorBearing;
    private bool exitBearing;
    private bool postDoorTrial;
    private DirectionTransferLayout activeDirectionLayout;
    private bool stressRecoveryFocus;
    private bool wallSideOnly;
    private List<RecoveryStart> recoveryStarts;
    private PolicyFrameCapture frameCapture;

    private struct RecoveryStart
    {
        public int Seed;
        public int Task;
        public string Source;
        public Vector2 Position;
        public float Heading;
    }
    private bool UsesDirectionPilot => sideSwap || directionTransfer || directionChoice;
    private float transferExitHalfWidth;
    private float swapBoxBack;
    private float swapExit;
    private float[] blindRanges;
    private int waypointSegment;
    private Vector2? obstacleCheckpoint;
    private bool passedObstacle;
    private readonly List<Rect> sceneWalls = new List<Rect>();
    private readonly List<Rect> sceneBoxes = new List<Rect>();
    private MaterialPropertyBlock obstacleAppearance;
    private GraphWallOutline wallOutline;
    private LowerPolicyModel evaluatedPolicy;
    private int noiseSampleCount;
    private LowerPolicyModel[] noisePolicies;
    private PlanarLidar.NoiseRealization[] noiseRealizations;
    private float[] noiseRanges;
    private float[] noiseTurns;
    private string evaluationModelId = "v8";
    private double firstPolicyScan;
    private double lastPolicyScan;
    private int mapSeed = -1;
    private Vector3 floorPosition, floorScale, cameraPosition;
    private float cameraSize;
    private bool[] originalWallsActive;
    private Rigidbody body;
    private RearDrive drive;
    private ManualDrive manual;
    private DriveRecorder recorder;
    private PlanarLidar lidar;
    private bool procedural;
    private bool batchMode;
    private string outputDirectory;
    private int episodeCount;
    private int completedCount;
    private int skippedCount;
    private int failedCount;
    private Vector3 startPosition;
    private Quaternion startRotation;
    private bool originalObstacleActive;
    private Vector3 leftWallPosition;
    private Vector3 rightWallPosition;
    private float originalTimeScale;
    private int episode = -1;
    private int segment;
    private float elapsed;
    private float settled;
    private float forwardCommand;
    private float turnCommand;
    private bool blocked;
    private bool finishing;
    private float disturbanceTime;
    private float disturbanceDirection;

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
        drive = GetComponent<RearDrive>();
        manual = GetComponent<ManualDrive>();
        recorder = GetComponent<DriveRecorder>();
        lidar = GetComponent<PlanarLidar>();
        startPosition = body.position;
        startRotation = body.rotation;
    }

    private void Start()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        batchMode = Application.isBatchMode && System.Array.IndexOf(args, "-teacherCollect") >= 0;
        if (!batchMode)
        {
            if (evaluationScene)
                StartPolicyPreview();
            return;
        }

        try
        {
            firstSeed = ReadInteger(args, "-teacherFirstSeed", firstSeed, 0, int.MaxValue - 500);
            seedCount = ReadInteger(args, "-teacherSeedCount", seedCount, 1, 500);
            outputDirectory = ReadOption(args, "-teacherOutput");
            if (outputDirectory != null)
                System.IO.Directory.CreateDirectory(outputDirectory);
            int auditCount = ReadInteger(args, "-teacherAuditLayouts", 0, 1, 1000);
            if (auditCount > 0)
            {
                ExportLayoutAudit(auditCount, System.Array.IndexOf(args, "-teacherAuditMixed") >= 0);
                return;
            }
            if (System.Array.IndexOf(args, "-teacherHallGoals") >= 0)
            {
                ExportHallGoals();
                return;
            }
            noiseSampleCount = ReadInteger(args, "-teacherCaptureNoiseSamples", 0, 2, 128);
            if (System.Array.IndexOf(args, "-teacherInference") >= 0)
                LoadPolicy(ReadOption(args, "-teacherPolicyWeights"), ReadOption(args, "-teacherPolicyName"));
            hallsOnly = System.Array.IndexOf(args, "-teacherHallsOnly") >= 0;
            waypointPilot = System.Array.IndexOf(args, "-teacherWaypointPilot") >= 0;
            waypointInput = System.Array.IndexOf(args, "-teacherWaypointInput") >= 0;
            sideSwap = System.Array.IndexOf(args, "-teacherSideSwap") >= 0;
            directionChoice = System.Array.IndexOf(args, "-teacherDirectionChoice") >= 0;
            directionChoiceApproach = System.Array.IndexOf(args, "-teacherDirectionChoiceApproach") >= 0;
            if (directionChoiceApproach && (!directionChoice || evaluatedPolicy != null))
                throw new System.ArgumentException("Direction choice approach starts require teacher collection.");
            directionTransfer = System.Array.IndexOf(args, "-teacherDirectionTransfer") >= 0;
            staggeredOnly = System.Array.IndexOf(args, "-teacherStaggeredOnly") >= 0;
            directionStress = System.Array.IndexOf(args, "-teacherDirectionStress") >= 0;
            largeDirection = System.Array.IndexOf(args, "-teacherLargeDirection") >= 0;
            doorwayArena = System.Array.IndexOf(args, "-teacherDoorwayArena") >= 0;
            doorBearing = System.Array.IndexOf(args, "-teacherDoorBearing") >= 0;
            exitBearing = System.Array.IndexOf(args, "-teacherExitBearing") >= 0;
            if (exitBearing && !doorBearing)
                throw new System.ArgumentException("Exit bearings require door bearings.");
            if (doorBearing && !doorwayArena)
                throw new System.ArgumentException("Door bearings require a doorway arena.");
            if (doorwayArena && !largeDirection)
                throw new System.ArgumentException("Doorway arenas require large-map mode.");
            if (largeDirection && (!directionTransfer || directionStress || staggeredOnly))
                throw new System.ArgumentException("Large maps require an independent full direction evaluation.");
            stressRecoveryFocus = System.Array.IndexOf(args, "-teacherStressRecoveryFocus") >= 0;
            wallSideOnly = System.Array.IndexOf(args, "-teacherWallSideOnly") >= 0;
            if (wallSideOnly && (!directionStress || stressRecoveryFocus))
                throw new System.ArgumentException("-teacherWallSideOnly requires a wall-side stress run.");
            if (stressRecoveryFocus && !directionStress)
                throw new System.ArgumentException("Stress recovery focus requires -teacherDirectionStress.");
            if (directionStress && (!directionTransfer || staggeredOnly))
                throw new System.ArgumentException("Direction stress requires the complete direction-transfer pilot.");
            if (staggeredOnly && !directionTransfer)
                throw new System.ArgumentException("-teacherStaggeredOnly requires -teacherDirectionTransfer.");
            string recoveryFile = ReadOption(args, "-teacherRecoveryStarts");
            if (recoveryFile != null)
            {
                if (!directionTransfer || evaluatedPolicy != null || staggeredOnly || largeDirection)
                    throw new System.ArgumentException("Recovery starts require a teacher direction-transfer run.");
                recoveryStarts = ReadRecoveryStarts(recoveryFile, directionStress ? 5 : 3);
            }
            string postDoorFile = ReadOption(args, "-teacherPostDoorStarts");
            if (postDoorFile != null)
            {
                if (!doorwayArena || recoveryStarts != null || (evaluatedPolicy != null && !doorBearing))
                    throw new System.ArgumentException("Post-door starts require a matching doorway run.");
                postDoorTrial = true;
                recoveryStarts = ReadRecoveryStarts(postDoorFile, maximumTask: 2, minimumTask: 0);
            }
            if (directionChoice && (sideSwap || directionTransfer))
                throw new System.ArgumentException("Direction choice requires its own layout mode.");
            if (UsesDirectionPilot && (waypointPilot || hallsOnly || outputDirectory == null || (sideSwap && directionTransfer)))
                throw new System.ArgumentException("Direction pilots require their own output and cannot be combined with each other or hall mode.");
            if (UsesDirectionPilot && evaluatedPolicy != null)
                waypointInput = true;
            if (waypointPilot && (!hallsOnly || outputDirectory == null))
                throw new System.ArgumentException("-teacherWaypointPilot requires -teacherHallsOnly and -teacherOutput.");
            if (waypointInput && ((!waypointPilot && !UsesDirectionPilot) || evaluatedPolicy == null || !evaluatedPolicy.UsesRelativeGoal))
                throw new System.ArgumentException("-teacherWaypointInput requires a goal-conditioned policy and waypoint pilot.");
            if (System.Array.IndexOf(args, "-teacherWaypointBlind") >= 0)
            {
                if (!waypointInput)
                    throw new System.ArgumentException("-teacherWaypointBlind requires -teacherWaypointInput.");
                blindRanges = new float[lidar.Ranges.Length];
                System.Array.Fill(blindRanges, float.PositiveInfinity);
            }
            int referenceCase = ReadInteger(args, "-teacherReferenceCase", 0, 1, 100);
            if (UsesDirectionPilot && referenceCase > 0)
                throw new System.ArgumentException("Direction pilots cannot be used with a reference layout.");
            if (referenceCase > 0)
                pdfPreview = ReferenceAvoidanceLayout.Load(referenceCase);
            string captureDirectory = ReadOption(args, "-teacherCaptureFrames");
            bool captureFollow = System.Array.IndexOf(args, "-teacherCaptureFollow") >= 0;
            bool captureSteering = System.Array.IndexOf(args, "-teacherCaptureSteering") >= 0;
            if (captureSteering && (!captureFollow || !waypointInput))
                throw new System.ArgumentException("Steering overlay requires follow capture and directional input.");
            if (noiseSampleCount > 0 && (!captureSteering || !doorwayArena || blindRanges != null))
                throw new System.ArgumentException("LiDAR resampling requires unblinded doorway steering capture.");
            if (captureFollow && captureDirectory == null)
                throw new System.ArgumentException("Follow camera requires a frame capture directory.");
            doorwayCaptureTask = ReadInteger(args, "-teacherDoorwayCaptureTask", -1, 0,
                                             DirectionTransferLayout.LargeScenarioCount - 1);
            if (doorwayCaptureTask >= 0 && (!doorwayArena || captureDirectory == null || seedCount != 1 ||
                                           recoveryStarts != null))
                throw new System.ArgumentException("Select one seed and enable doorway frame capture.");
            if (captureDirectory != null)
            {
                if (evaluatedPolicy == null || !directionTransfer)
                    throw new System.ArgumentException("Frame capture requires a direction policy evaluation.");
                frameCapture = new PolicyFrameCapture(overview, captureDirectory,
                    doorwayCaptureTask >= 0 || captureFollow ? body.GetComponent<Renderer>() : null,
                    captureFollow ? body : null, captureSteering, noiseSampleCount);
            }
            StartCollection(true);
        }
        catch (System.Exception error)
        {
            Debug.LogException(error, this);
            ExitBatch(2);
        }
    }

    private void ExportHallGoals()
    {
        if (outputDirectory == null)
            throw new System.ArgumentException("-teacherHallGoals requires -teacherOutput.");
        using var output = new StreamWriter(Path.Combine(outputDirectory, "goals.csv"));
        output.WriteLine("family,task,seed,goal_x_m,goal_z_m");
        int count = 0;
        for (int seed = firstSeed; seed < firstSeed + seedCount; seed++)
        for (int mission = 0; mission < OpenAvoidanceLayout.MissionsPerSeed; mission++)
        {
            if (!OpenAvoidanceLayout.TryCreate(seed, mission, Clearance, out OpenAvoidanceLayout hall))
                continue;
            Vector2 goal = hall.Reference[hall.Reference.Count - 1];
            output.WriteLine(System.FormattableString.Invariant($"{hall.Family},{mission},{seed},{goal.x:R},{goal.y:R}"));
            count++;
        }
        Debug.Log($"Exported {count} Unity hall targets.", this);
        ExitBatch(count == 0 ? 1 : 0);
    }

    private void ExportLayoutAudit(int count, bool mixed)
    {
        if (outputDirectory == null)
            throw new System.ArgumentException("-teacherAuditLayouts requires -teacherOutput.");
        using var audit = new StreamWriter(Path.Combine(outputDirectory, "layouts.txt"), append: false);
        using var graphs = mixed ? new StreamWriter(Path.Combine(outputDirectory, "graph-audit.csv")) : null;
        graphs?.WriteLine("seed,family,feasible_pairs,attempted_pairs,shown_mission");
        int layoutsPerSeed = OpenAvoidanceLayout.MissionsPerSeed + (mixed ? 1 : 0);
        int exported = 0;
        for (int index = 0; index < count; index++)
        {
            int seed = firstSeed + index / layoutsPerSeed;
            int mission = index % layoutsPerSeed;
            if (mixed && mission == OpenAvoidanceLayout.MissionsPerSeed)
            {
                ExportGraphAudit(seed, audit, graphs);
                exported++;
                continue;
            }
            if (!OpenAvoidanceLayout.TryCreate(seed, mission, Clearance, out OpenAvoidanceLayout hall))
            {
                audit.WriteLine($"{seed}/{mission} skipped");
                continue;
            }

            string walls = string.Join(";", hall.Walls.ConvertAll(RectCoordinates));
            string boxes = string.Join(";", hall.Boxes.ConvertAll(RectCoordinates));
            string direct = string.Join(";", hall.Reference.ConvertAll(PointCoordinates));
            string detour = string.Join(";", hall.Detour.ConvertAll(PointCoordinates));
            audit.WriteLine($"{seed}/{mission} {hall.Family} {hall.PortalCount} | {walls} | {boxes} | {direct} | {detour}");
            exported++;
        }
        Debug.Log($"Layout audit finished: {exported} accepted, {count - exported} skipped.", this);
        ExitBatch(exported == count ? 0 : 1);
    }

    private void ExportGraphAudit(int seed, StreamWriter audit, StreamWriter graphs)
    {
        currentMap = new GraphFloorPlan(seed);
        string family = currentMap.IsCorridorNetwork ? "network" : "rooms";
        int feasible = 0;
        int shownMission = -1;
        Rect shownBox = default;
        var reference = new List<Vector2>();
        var detour = new List<Vector2>();
        for (int mission = 0; mission < OpenMissionsPerMap; mission++)
        {
            GraphFloorPlan.Task task;
            try
            {
                task = currentMap.CreateTask(seed, mission);
            }
            catch (System.InvalidOperationException)
            {
                continue;
            }
            bool nearExit = PreferExit(task, seed, mission);
            if (!FindPairedRoutes(task, seed, mission, ref nearExit))
                continue;
            feasible++;
            if (shownMission >= 0)
                continue;
            shownMission = mission;
            shownBox = pairedObstacle;
            reference.AddRange(pairedReference);
            detour.AddRange(pairedDetour);
        }

        string walls = string.Join(";", currentMap.Walls.ConvertAll(wall => RectCoordinates(wall.Area)));
        string box = feasible > 0 ? RectCoordinates(shownBox) : "";
        string direct = string.Join(";", reference.ConvertAll(PointCoordinates));
        string avoided = string.Join(";", detour.ConvertAll(PointCoordinates));
        audit.WriteLine($"{seed}/graph {family} {currentMap.DoorWidths.Count} | {walls} | {box} | {direct} | {avoided}");
        graphs.WriteLine($"{seed},{family},{feasible},{OpenMissionsPerMap},{shownMission}");
    }

    private static List<RecoveryStart> ReadRecoveryStarts(string path, int maximumTask, int minimumTask = 2)
    {
        var starts = new List<RecoveryStart>();
        var keys = new HashSet<(int, int, string)>();
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        using var reader = new StreamReader(path);
        if (reader.ReadLine() != "seed,task,source,x_m,z_m,heading_deg")
            throw new System.ArgumentException("Unexpected recovery-start header.");
        for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
        {
            string[] fields = line.Split(',');
            if (fields.Length != 6)
                throw new System.ArgumentException("Expected six fields for each recovery start.");
            int seed = int.Parse(fields[0], culture);
            int task = int.Parse(fields[1], culture);
            string source = fields[2];
            float x = float.Parse(fields[3], culture);
            float z = float.Parse(fields[4], culture);
            float heading = float.Parse(fields[5], culture);
            if (seed < 0 || (task < minimumTask || task > maximumTask) || (source != "normal" && source != "policy") ||
                !float.IsFinite(x) || !float.IsFinite(z) || !float.IsFinite(heading) ||
                !keys.Add((seed, task, source)))
                throw new System.ArgumentException($"Invalid or duplicate recovery start for {seed}/{task}/{source}.");
            starts.Add(new RecoveryStart { Seed = seed, Task = task, Source = source,
                                           Position = new Vector2(x, z), Heading = heading });
        }
        if (starts.Count == 0)
            throw new System.ArgumentException("No recovery starts were supplied.");
        return starts;
    }

    private static string ReadOption(string[] args, string name)
    {
        int index = System.Array.IndexOf(args, name);
        if (index < 0)
            return null;
        if (index + 1 >= args.Length || args[index + 1].StartsWith("-"))
            throw new System.ArgumentException($"Missing value for {name}.");
        return args[index + 1];
    }

    private static int ReadInteger(string[] args, string name, int fallback, int min, int max)
    {
        string text = ReadOption(args, name);
        if (text == null)
            return fallback;
        if (!int.TryParse(text, out int value) || value < min || value > max)
            throw new System.ArgumentException($"{name} must be between {min} and {max}.");
        return value;
    }

    private void Update()
    {
        Keyboard keyboard = Keyboard.current;
        if (batchMode || episode >= 0 || recorder.IsRecording || keyboard == null)
            return;
        if (evaluationScene)
        {
            if (keyboard.pKey.wasPressedThisFrame)
                StartPolicyPreview();
            return;
        }
        if (keyboard.gKey.wasPressedThisFrame)
            StartCollection(true);
        else if (keyboard.tKey.wasPressedThisFrame)
            StartCollection(false);
        else if (keyboard.pKey.wasPressedThisFrame)
            StartPolicyPreview();
        else if (keyboard.vKey.wasPressedThisFrame)
        {
            pdfPreview = ReferenceAvoidanceLayout.Load(pdfCase);
            StartCollection(true);
        }
    }

    private void StartPolicyPreview()
    {
        try
        {
            LoadPolicy();
            if (evaluationScene)
            {
                string dataFolder = Path.GetDirectoryName(DriveRecorder.DefaultOutputDirectory());
                outputDirectory = Path.Combine(dataFolder, "policy-evaluations",
                    $"{evaluationModelId}-seed{firstSeed:D5}-{System.DateTime.UtcNow:yyyyMMdd-HHmmss}");
            }
            StartCollection(true);
        }
        catch (System.Exception error)
        {
            evaluatedPolicy = null;
            Debug.LogException(error, this);
        }
    }

    private void LoadPolicy(string weightsPath = null, string modelName = null)
    {
        TextAsset asset = evaluationWeights ? evaluationWeights : Resources.Load<TextAsset>("LowerPolicyV8");
        if (weightsPath == null && asset == null)
            throw new FileNotFoundException("Missing Resources/LowerPolicyV8.bytes");

        string name = modelName ?? (weightsPath != null ? Path.GetFileNameWithoutExtension(weightsPath) :
            evaluationWeights != null ? evaluationWeights.name : "v8");
        evaluationModelId = System.Text.RegularExpressions.Regex.Replace(name.ToLowerInvariant(), "[^a-z0-9-]", "-");
        byte[] weights = weightsPath != null ? File.ReadAllBytes(weightsPath) : asset.bytes;
        evaluatedPolicy = new LowerPolicyModel(weights);
        if (noiseSampleCount > 0)
        {
            noisePolicies = new LowerPolicyModel[noiseSampleCount];
            noiseRealizations = new PlanarLidar.NoiseRealization[noiseSampleCount];
            noiseRanges = new float[lidar.Ranges.Length];
            noiseTurns = new float[noiseSampleCount];
            for (int i = 0; i < noiseSampleCount; i++)
                noisePolicies[i] = new LowerPolicyModel(weights);
        }
        Debug.Log($"Policy checkpoint: {evaluationModelId} ({weightsPath ?? asset.name})", this);
    }

    private int ScenariosPerSeed
    {
        get
        {
            if (directionChoice)
                return 4;
            if (largeDirection)
                return doorwayCaptureTask >= 0 ? 1 : DirectionTransferLayout.LargeScenarioCount;
            if (wallSideOnly)
                return 2;
            if (directionStress)
                return DirectionTransferLayout.StressScenarioCount - (stressRecoveryFocus ? 2 : 0);
            if (directionTransfer)
                return staggeredOnly ? 2 : DirectionTransferLayout.ScenarioCount;
            if (sideSwap)
                return 2;
            if (hallsOnly)
                return HallScenarios;
            return evaluatedPolicy != null && !evaluationScene ? MissionsPerMap : MissionsPerMap + HallScenarios;
        }
    }

    private void StartCollection(bool useProcedural)
    {
        procedural = useProcedural;
        firstSeed = Mathf.Clamp(firstSeed, 0, int.MaxValue - 500);
        episodeCount = pdfPreview != null ? 1 : evaluatedPolicy != null && !batchMode && !evaluationScene ? MissionsPerMap :
            procedural ? ScenariosPerSeed * Mathf.Clamp(seedCount, 1, 500) : PreviewCount;
        if (waypointPilot)
            episodeCount *= 2;
        if (recoveryStarts != null)
            episodeCount = recoveryStarts.Count;
        completedCount = 0;
        skippedCount = 0;
        failedCount = 0;

        for (int i = 0; i < layoutBoxes.Length; i++)
        {
            if (layoutBoxes[i] != null)
                continue;
            layoutBoxes[i] = Instantiate(obstacle);
            layoutBoxes[i].name = $"Layout obstacle {i + 1}";
        }

        originalObstacleActive = obstacle.gameObject.activeSelf;
        if (procedural)
        {
            originalWallsActive = new[] { leftWall.gameObject.activeSelf, rightWall.gameObject.activeSelf,
                                          backWall.gameObject.activeSelf, endWall.gameObject.activeSelf };
            floorPosition = floor.position;
            floorScale = floor.localScale;
            cameraPosition = overview.transform.position;
            cameraSize = overview.orthographicSize;
            leftWall.gameObject.SetActive(false);
            rightWall.gameObject.SetActive(false);
            backWall.gameObject.SetActive(false);
            endWall.gameObject.SetActive(false);
            foreach (BoxCollider box in layoutBoxes)
                box.gameObject.SetActive(false);
            mapSeed = -1;
        }
        leftWallPosition = leftWall.transform.position;
        rightWallPosition = rightWall.transform.position;
        originalTimeScale = Time.timeScale;
        obstacle.gameObject.SetActive(false);
        manual.enabled = false;
        Time.timeScale = (evaluatedPolicy != null || pdfPreview != null) && !batchMode ? 2f :
            procedural ? Mathf.Clamp(collectionTimeScale, 1f, 12f) : 2f;
        Debug.Log($"Starting {episodeCount} teacher episodes. CSV directory: {outputDirectory ?? DriveRecorder.DefaultOutputDirectory()}", this);
        AdvanceEpisode(0);
    }

    private void AdvanceEpisode(int first)
    {
        for (int index = first; index < episodeCount; index++)
        {
            if (TryBeginEpisode(index))
                return;
            skippedCount++;
        }

        bool evaluatedFailures = evaluatedPolicy != null && failedCount > 0;
        RestoreManualScene();
        Debug.Log($"Teacher batch finished: {completedCount} saved, {skippedCount} skipped, {failedCount} failed.", this);
        if (batchMode)
            ExitBatch(completedCount == 0 && !evaluatedFailures ? 1 : 0);
    }

    private static void ExitBatch(int code)
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.update += ExitEditor;

        void ExitEditor()
        {
            UnityEditor.EditorApplication.update -= ExitEditor;
            UnityEditor.EditorApplication.Exit(code);
        }
#else
        Application.Quit(code);
#endif
    }

    private bool TryBeginEpisode(int episodeIndex)
    {
        episode = episodeIndex;
        segment = 0;
        elapsed = 0f;
        settled = 0f;
        forwardCommand = 0f;
        turnCommand = 0f;
        finishing = false;
        disturbanceTime = float.PositiveInfinity;
        ResetVehicle();
        if (!procedural)
            return BeginPreviewEpisode(episodeIndex);
        if (pdfPreview != null)
            return BeginPdfEpisode();

        if (recoveryStarts != null)
        {
            RecoveryStart restart = recoveryStarts[episodeIndex];
            return BeginDirectionTransferEpisode(restart.Seed, restart.Task, restart);
        }

        int seed = evaluatedPolicy != null && !batchMode && !evaluationScene ? 1009 : firstSeed + (episodeIndex / ScenariosPerSeed) % seedCount;
        int scenario = episodeIndex % ScenariosPerSeed;
        if (directionTransfer)
        {
            int taskOffset = wallSideOnly ? 4 : (staggeredOnly || stressRecoveryFocus ? 2 : 0);
            return BeginDirectionTransferEpisode(seed,
                doorwayCaptureTask >= 0 ? doorwayCaptureTask : scenario + taskOffset);
        }
        if (directionChoice)
            return BeginDirectionChoiceEpisode(seed, fork: scenario >= 2, chooseLeft: scenario % 2 == 0);
        if (sideSwap)
            return BeginSideSwapEpisode(seed, boxLeft: scenario == 0);
        return hallsOnly || scenario >= MissionsPerMap
            ? BeginHallEpisode(seed, hallsOnly ? scenario : scenario - MissionsPerMap)
            : BeginGraphEpisode(seed, scenario);
    }

    private bool BeginGraphEpisode(int seed, int scenario)
    {
        int mission = scenario == BlockedMission ? OpenMissionsPerMap : scenario / 2;
        bool useBox = scenario != BlockedMission && scenario % 2 == 1;
        if (seed != mapSeed)
        {
            currentMap = new GraphFloorPlan(seed);
            mapSeed = seed;
            FitOverview(currentMap.Area);
            sceneWalls.Clear();
            foreach (GraphFloorPlan.Wall wall in currentMap.Walls)
                sceneWalls.Add(wall.Area);
            ShowWallOutline(sceneWalls);
        }

        GraphFloorPlan.Task task;
        try
        {
            task = currentMap.CreateTask(seed, mission);
        }
        catch (System.InvalidOperationException error)
        {
            Debug.LogWarning($"{EpisodeProgress} Skipped map {seed}, mission {mission}: {error.Message}", this);
            return false;
        }
        bool nearExit = PreferExit(task, seed, mission);
        Rect? sceneObstacle = task.Blocked ? task.Obstacle : null;
        string placement = task.Blocked ? "blocked" : "clear";

        if (!task.Blocked)
        {
            if (!useBox && !FindPairedRoutes(task, seed, mission, ref nearExit))
            {
                Debug.Log($"{PairProgress} Skipped paired runs for map {seed}, mission {mission}: no feasible detour.", this);
                return false;
            }
            if (pairedReference == null)
                return false;

            sceneObstacle = useBox ? pairedObstacle : (Rect?)null;
            if (useBox)
                placement = pairedAtExit ? "box-exit" : "box-approach";
            route.Clear();
            route.AddRange(useBox ? pairedDetour : pairedReference);
        }

        body.position = new Vector3(task.Start.x, startPosition.y, task.Start.y);
        body.rotation = Quaternion.Euler(0f, task.Heading, 0f);
        sceneBoxes.Clear();
        if (sceneObstacle.HasValue)
            sceneBoxes.Add(sceneObstacle.Value);
        SetProceduralColliders(sceneWalls, sceneBoxes, drawWalls: false);
        if (task.Blocked && !AvoidancePlacement.Plan(currentMap, task, Clearance, obstacles, route, out _))
        {
            Debug.Log($"{EpisodeProgress} Skipped blocked run for map {seed}: stopping point is unavailable.", this);
            return false;
        }

        blocked = task.Blocked;
        recorder.Intent = task.Intent;
        lidar.SetNoiseSeed(unchecked(seed * 733 + mission / 2));

        var random = new System.Random(unchecked(seed * 397 + mission / 2));
        if (!blocked && random.NextDouble() < 0.6)
        {
            disturbanceTime = 0.7f + (float)random.NextDouble() * 6f;
            disturbanceDirection = random.Next(2) == 0 ? -1f : 1f;
        }

        string family = currentMap.IsCorridorNetwork ? "network" : "rooms";
        string intent = task.Intent.ToString().ToLowerInvariant();
        string label = $"{family}-{intent}-{placement}-task{mission:D2}-seed{seed:D5}";
        StartProceduralRecording(label);
        return true;
    }

    private bool BeginSideSwapEpisode(int seed, bool boxLeft)
    {
        var random = new System.Random(seed);
        float halfWidth = Sample(random, 1.36f, 1.90f);
        float nose = Sample(random, 1.35f, 2.00f);
        float length = Sample(random, 0.50f, 0.83f);
        float intrusion = Sample(random, 0.12f, 0.27f);
        float startX = Sample(random, -0.12f, 0.12f);
        swapBoxBack = nose + length;
        swapExit = swapBoxBack + Sample(random, 1.65f, 2.05f);
        float end = swapExit + 0.72f;
        var area = Rect.MinMaxRect(-halfWidth, -0.65f, halfWidth, end);
        sceneWalls.Clear();
        sceneWalls.Add(new Rect(-halfWidth - 0.15f, -0.8f, 0.15f, end + 0.95f));
        sceneWalls.Add(new Rect(halfWidth, -0.8f, 0.15f, end + 0.95f));
        sceneWalls.Add(new Rect(-halfWidth, -0.8f, 2f * halfWidth, 0.15f));
        sceneWalls.Add(new Rect(-halfWidth, end, 2f * halfWidth, 0.15f));
        Rect box = boxLeft
            ? Rect.MinMaxRect(-halfWidth, nose, intrusion, swapBoxBack)
            : Rect.MinMaxRect(-intrusion, nose, halfWidth, swapBoxBack);

        sceneBoxes.Clear();
        sceneBoxes.Add(box);
        FitOverview(area);
        SetProceduralColliders(sceneWalls, sceneBoxes);
        var start = new Vector2(startX, 0f);
        var goal = new Vector2(0f, swapExit + 0.18f);
        var driveArea = Rect.MinMaxRect(-halfWidth + Clearance, -0.2f, halfWidth - Clearance, end - Clearance);
        obstacles.Clear();
        obstacles.Add(new Bounds(new Vector3(box.center.x, 0.2f, box.center.y),
                                 new Vector3(box.width, 0.4f, box.height)));
        if (!TeacherRoutePlanner.TryPlan(start, goal, driveArea, Clearance, obstacles, !boxLeft, route))
        {
            Debug.LogWarning($"{EpisodeProgress} No safe side-swap teacher route for {seed}.");
            return false;
        }

        body.position = new Vector3(startX, startPosition.y, 0f);
        body.rotation = Quaternion.identity;
        blocked = false;
        passedObstacle = false;
        recorder.Intent = NavigationIntent.Follow;
        lidar.SetNoiseSeed(unchecked(seed * 1999));
        lidar.ScanOnNextFixedUpdate();
        string side = boxLeft ? "left" : "right";
        Debug.Log(System.FormattableString.Invariant(
            $"Side swap {seed}/{side}: {RectCoordinates(area)} | {RectCoordinates(box)} | exit_z={swapExit:R}"));
        StartProceduralRecording($"sideswap-follow-swap{side}-task00-seed{seed:D5}");
        return true;
    }

    private bool BeginDirectionChoiceEpisode(int seed, bool fork, bool chooseLeft)
    {
        var random = new System.Random(unchecked(seed * 149 + (fork ? 31 : 0)));
        float halfWidth = Sample(random, 2.45f, 3.15f);
        float startX = Sample(random, -0.18f, 0.18f);
        float yaw = Sample(random, -6f, 6f);
        float front = Sample(random, 1.80f, 2.30f);
        float islandCenter = Sample(random, -0.25f, 0.25f);
        float islandHalfWidth = Sample(random, 0.37f, 0.57f);
        float back = front + (fork ? Sample(random, 1.50f, 2.20f) : Sample(random, 0.60f, 0.95f));
        Rect island = Rect.MinMaxRect(islandCenter - islandHalfWidth, front,
                                       islandCenter + islandHalfWidth, back);
        swapExit = back + Sample(random, 1.15f, 1.65f);
        transferExitHalfWidth = halfWidth - 0.22f;
        swapBoxBack = back;
        float end = swapExit + 0.80f;
        var area = Rect.MinMaxRect(-halfWidth, -0.65f, halfWidth, end);
        sceneWalls.Clear();
        sceneWalls.Add(new Rect(-halfWidth - 0.15f, -0.80f, 0.15f, end + 0.95f));
        sceneWalls.Add(new Rect(halfWidth, -0.80f, 0.15f, end + 0.95f));
        sceneWalls.Add(new Rect(-halfWidth, -0.80f, 2f * halfWidth, 0.15f));
        sceneWalls.Add(new Rect(-halfWidth, end, 2f * halfWidth, 0.15f));
        float approachHalfWidth = Sample(random, 0.95f, 1.20f);
        float branchStart = front - Sample(random, 1.25f, 1.45f);
        if (fork)
        {
            sceneWalls.Add(Rect.MinMaxRect(-halfWidth, -0.65f, -approachHalfWidth, branchStart));
            sceneWalls.Add(Rect.MinMaxRect(approachHalfWidth, -0.65f, halfWidth, branchStart));
            sceneWalls.Add(island);
        }
        sceneBoxes.Clear();
        if (!fork)
            sceneBoxes.Add(island);
        FitOverview(area);
        SetProceduralColliders(sceneWalls, sceneBoxes);

        float sideX = chooseLeft ? (island.xMin - halfWidth) * 0.5f : (island.xMax + halfWidth) * 0.5f;
        choiceWaypoint = new Vector2(sideX, back + 0.65f);
        choiceSwitchZ = back + 0.30f;
        float startZ = 0f;
        if (directionChoiceApproach)
        {
            startX = islandCenter + Sample(random, -0.16f, 0.16f);
            startZ = front - Sample(random, 0.67f, 0.99f);
            yaw = Sample(random, -15f, 15f);
        }
        route.Clear();
        route.Add(new Vector2(startX, startZ));
        if (evaluatedPolicy == null)
        {
            if (fork && !directionChoiceApproach)
                route.Add(new Vector2(Mathf.Sign(sideX) * (approachHalfWidth - 0.36f), branchStart + 0.22f));
            route.Add(new Vector2(sideX, front - 0.45f));
            route.Add(choiceWaypoint);
        }
        route.Add(new Vector2(sideX, swapExit + 0.18f));
        body.position = new Vector3(startX, startPosition.y, startZ);
        body.rotation = Quaternion.Euler(0f, yaw, 0f);
        blocked = false;
        passedObstacle = false;
        recorder.Intent = NavigationIntent.Follow;
        lidar.SetNoiseSeed(unchecked(seed * 1999 + (fork ? 17 : 0)));
        lidar.ScanOnNextFixedUpdate();
        string family = fork ? "fork" : "island";
        string side = chooseLeft ? "left" : "right";
        Debug.Log(System.FormattableString.Invariant(
            $"Direction choice {seed}/{family}/{side}: {RectCoordinates(area)} | {RectCoordinates(island)} | exit={swapExit:R}"));
        StartProceduralRecording($"direction-choice-{family}-{side}-seed{seed:D5}");
        return true;
    }

    private static float Sample(System.Random random, float min, float max) =>
        Mathf.Lerp(min, max, (float)random.NextDouble());

    private bool BeginDirectionTransferEpisode(int seed, int scenario, RecoveryStart? restart = null)
    {
        var layout = new DirectionTransferLayout(seed, scenario, directionStress, largeDirection, doorwayArena);
        activeDirectionLayout = layout;
        FitOverview(layout.Area);
        sceneBoxes.Clear();
        sceneBoxes.AddRange(layout.Boxes);
        SetProceduralColliders(layout.Walls, sceneBoxes);
        swapExit = layout.ExitZ;
        transferExitHalfWidth = layout.ExitHalfWidth;
        swapBoxBack = sceneBoxes.Count > 0 ? sceneBoxes[sceneBoxes.Count - 1].yMax : float.PositiveInfinity;
        Vector2 start = restart?.Position ?? layout.Start;
        if (postDoorTrial && (start.y <= layout.Walls[6].yMax + 0.28f || start.y >= layout.ExitZ))
            throw new System.ArgumentException($"Post-door start outside the last room for {seed}/{scenario}.");
        var goal = new Vector2(0f, swapExit + 0.18f);
        if (evaluatedPolicy == null)
        {
            Rect area = layout.Area;
            var driveArea = Rect.MinMaxRect(area.xMin + Clearance, -0.2f, area.xMax - Clearance,
                                            area.yMax - Clearance);
            if (!TeacherRoutePlanner.TryPlan(start, goal, driveArea, Clearance, obstacles,
                                             scenario % 2 != 0, route))
            {
                Debug.LogWarning($"{EpisodeProgress} No safe route for transfer layout {seed}/{scenario}.");
                return false;
            }
        }
        else
        {
            route.Clear();
            route.Add(start);
            route.Add(goal);
        }

        body.position = new Vector3(start.x, startPosition.y, start.y);
        body.rotation = Quaternion.Euler(0f, restart.HasValue ? -restart.Value.Heading : layout.StartHeading, 0f);
        blocked = false;
        passedObstacle = false;
        recorder.Intent = NavigationIntent.Follow;
        int noiseSeed = unchecked(seed * 1999 + scenario / 2);
        lidar.SetNoiseSeed(noiseSeed);
        if (noisePolicies != null)
        {
            for (int i = 0; i < noisePolicies.Length; i++)
            {
                noisePolicies[i].Reset();
                noiseRealizations[i] = new PlanarLidar.NoiseRealization(unchecked(noiseSeed + (i + 1) * 104729));
            }
        }
        lidar.ScanOnNextFixedUpdate();
        Debug.Log(System.FormattableString.Invariant(
            $"Direction transfer {seed}/{scenario} {layout.Family}: {RectCoordinates(layout.Area)} | {string.Join(";", layout.Walls.ConvertAll(RectCoordinates))} | {string.Join(";", sceneBoxes.ConvertAll(RectCoordinates))} | exit={swapExit:R},{transferExitHalfWidth:R}"));
        string prefix = restart.HasValue && !postDoorTrial ? $"recovery-{restart.Value.Source}-" : "direction-";
        StartProceduralRecording($"{prefix}{layout.Family}-follow-task{scenario:D2}-seed{seed:D5}");
        return true;
    }

    private bool BeginHallEpisode(int seed, int scenario)
    {
        int mission = scenario / 2;
        bool cluttered = scenario % 2 == 1;
        if (!cluttered && !OpenAvoidanceLayout.TryCreate(seed, mission, Clearance, out pairedHall))
        {
            Debug.LogWarning($"{PairProgress} Skipped hall pair for seed {seed}, mission {mission}: no feasible crossing.", this);
            return false;
        }
        if (pairedHall == null)
            return false;

        OpenAvoidanceLayout hall = pairedHall;
        FitOverview(hall.Area);
        wallOutline?.Hide();
        mapSeed = -1;
        sceneBoxes.Clear();
        if (cluttered)
            sceneBoxes.AddRange(hall.Boxes);
        SetProceduralColliders(hall.Walls, sceneBoxes);
        body.position = new Vector3(hall.Start.x, startPosition.y, hall.Start.y);
        body.rotation = Quaternion.Euler(0f, hall.Heading, 0f);
        route.Clear();
        route.AddRange(cluttered ? hall.Detour : hall.Reference);
        blocked = false;
        recorder.Intent = NavigationIntent.Follow;
        lidar.SetNoiseSeed(unchecked(seed * 1999 + mission));
        bool yawDisturbed = waypointPilot && episode >= episodeCount / 2;
        if (yawDisturbed)
        {
            float yaw = (seed + mission) % 2 == 0 ? 32f : -32f;
            body.rotation = Quaternion.Euler(0f, hall.Heading + yaw, 0f);
        }
        if (waypointPilot)
        {
            waypointSegment = 0;
            passedObstacle = false;
            obstacleCheckpoint = FirstObstacleCheckpoint();
        }
        string placement = cluttered ? $"clutter{hall.Boxes.Count:D2}" : "clear";
        if (cluttered && batchMode)
        {
            var boxes = hall.Boxes.ConvertAll(RectCoordinates);
            var walls = hall.Walls.ConvertAll(RectCoordinates);
            Debug.Log($"Hall layout {seed}/{mission} {hall.Family} doors={hall.PortalCount}: {RectCoordinates(hall.Area)} | {string.Join(";", boxes)} | {string.Join(";", walls)}", this);
        }
        StartProceduralRecording($"{hall.Family}-follow-{placement}-task{mission:D2}-seed{seed:D5}" +
                                 (waypointPilot ? (yawDisturbed ? "-yaw" : "-nominal") : ""));
        return true;
    }

    private Vector2? FirstObstacleCheckpoint()
    {
        // Mark the end of the first contiguous obstacle encounter along the planned crossing.
        if (sceneBoxes.Count == 0)
            return null;

        bool nearBox = false;
        for (int i = 0; i + 1 < route.Count; i++)
        {
            Vector2 start = route[i], end = route[i + 1];
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(start, end) / 0.05f));
            for (int step = 0; step <= steps; step++)
            {
                Vector2 point = Vector2.Lerp(start, end, (float)step / steps);
                bool near = sceneBoxes.Exists(box =>
                {
                    Vector2 closest = new Vector2(Mathf.Clamp(point.x, box.xMin, box.xMax),
                                                  Mathf.Clamp(point.y, box.yMin, box.yMax));
                    return (point - closest).sqrMagnitude <= 0.70f * 0.70f;
                });
                if (nearBox && !near)
                    return point;
                nearBox |= near;
            }
        }
        return null;
    }

    private Vector2 Waypoint(Vector2 position) =>
        TeacherPathFollower.LookAheadPoint(route, position, LookAhead, ref waypointSegment);

    private (Vector2 goal, Vector2 waypoint, bool hasCheckpoint, bool passed) DescribePilotScan()
    {
        Vector2 position = new Vector2(body.position.x, body.position.z);
        if (UsesDirectionPilot)
        {
            passedObstacle |= position.y >= swapBoxBack + 0.28f;
            // The teacher's obstacle-aware route supplies motor labels only.
            Vector2 direction = Vector2.up;
            if (directionChoice && position.y < choiceSwitchZ)
                direction = (choiceWaypoint - position).normalized;
            else if (doorBearing)
                direction = activeDirectionLayout.DoorBearing(position, exitBearing);
            return (route[route.Count - 1], position + direction, sceneBoxes.Count > 0, passedObstacle);
        }
        if (obstacleCheckpoint.HasValue && Vector2.Distance(position, obstacleCheckpoint.Value) <= 0.30f)
            passedObstacle = true;
        return (route[route.Count - 1], Waypoint(position), obstacleCheckpoint.HasValue, passedObstacle);
    }

    private static string RectCoordinates(Rect rect) =>
        System.FormattableString.Invariant($"{rect.x:F3},{rect.y:F3},{rect.width:F3},{rect.height:F3}");

    private static string PointCoordinates(Vector2 point) =>
        System.FormattableString.Invariant($"{point.x:F3},{point.y:F3}");

    private bool BeginPdfEpisode()
    {
        FitOverview(pdfPreview.Area);
        sceneBoxes.Clear();
        sceneBoxes.Add(pdfPreview.Box);
        ShowWallOutline(pdfPreview.Walls);
        SetProceduralColliders(pdfPreview.Walls, sceneBoxes, drawWalls: false);
        Vector2 start = pdfPreview.Reference[0];
        Vector2 direction = pdfPreview.Reference[1] - start;
        body.position = new Vector3(start.x, startPosition.y, start.y);
        body.rotation = Quaternion.Euler(0f, Mathf.Atan2(direction.x, direction.y) * Mathf.Rad2Deg, 0f);
        route.Clear();
        route.AddRange(pdfPreview.Detour);
        blocked = false;
        recorder.Intent = pdfPreview.Intent;
        lidar.SetNoiseSeed(pdfPreview.Seed);
        Debug.Log($"{EpisodeProgress} PDF layout seed {pdfPreview.Seed}, task {pdfPreview.Mission}, {pdfPreview.Walls.Count} wall colliders.", this);
        StartEpisodeRecording($"reference-pdf-task{pdfPreview.Mission:D2}-seed{pdfPreview.Seed:D5}");
        return true;
    }

    private string EpisodeProgress => $"[{episode + 1}/{episodeCount}]";
    private string PairProgress => $"[{episode + 1}-{episode + 2}/{episodeCount}]";

    private void StartProceduralRecording(string label)
    {
        evaluatedPolicy?.Reset();
        firstPolicyScan = Time.fixedTimeAsDouble;
        lastPolicyScan = double.NegativeInfinity;
        string source = evaluatedPolicy == null ? "teacher-v12-" : $"policy-{evaluationModelId}-";
        StartEpisodeRecording(source + label);
    }

    private void StartEpisodeRecording(string label)
    {
        frameCapture?.Begin(label);
        recorder.StartRecording(label, automaticRun: true, outputDirectory: outputDirectory, progress: EpisodeProgress, pilot: waypointPilot || UsesDirectionPilot ? DescribePilotScan : null);
    }

    private static bool PreferExit(GraphFloorPlan.Task task, int seed, int mission) =>
        task.Intent != NavigationIntent.Follow && (seed + mission) % 2 == 0;

    private bool FindPairedRoutes(GraphFloorPlan.Task task, int seed, int mission, ref bool nearExit)
    {
        int placementSeed = unchecked(seed * 397 + mission);
        if (!AvoidancePlacement.TryPlace(currentMap, task, placementSeed, nearExit, Clearance,
                                        out pairedObstacle, out pairedReference, out pairedDetour))
        {
            nearExit = !nearExit;
            if (!AvoidancePlacement.TryPlace(currentMap, task, placementSeed, nearExit, Clearance,
                                             out pairedObstacle, out pairedReference, out pairedDetour))
            {
                pairedReference = null;
                return false;
            }
        }

        pairedAtExit = nearExit;
        return true;
    }

    private void FitOverview(Rect area)
    {
        floor.position = new Vector3(area.center.x, floorPosition.y, area.center.y);
        floor.localScale = new Vector3(area.width, floorScale.y, area.height);
        overview.transform.position = new Vector3(area.center.x, cameraPosition.y, area.center.y);
        overview.orthographicSize = Mathf.Max(area.height / 2f, area.width / (2f * overview.aspect)) + 0.4f;
    }

    private void ShowWallOutline(List<Rect> walls)
    {
        Renderer floorRenderer = floor.GetComponent<Renderer>();
        wallOutline ??= new GraphWallOutline(floorRenderer.sharedMaterial);
        wallOutline.Show(walls, floorRenderer.bounds.max.y + 0.006f);
    }

    private void SetProceduralColliders(List<Rect> walls, List<Rect> boxes, bool drawWalls = true)
    {
        int count = walls.Count + boxes.Count;
        if (obstacleAppearance == null)
        {
            obstacleAppearance = new MaterialPropertyBlock();
            obstacleAppearance.SetColor("_BaseColor", new Color(0.33f, 0.27f, 0.21f));
        }
        while (graphBoxes.Count < count)
            graphBoxes.Add(Instantiate(obstacle));
        obstacles.Clear();
        for (int i = 0; i < graphBoxes.Count; i++)
        {
            BoxCollider box = graphBoxes[i];
            box.gameObject.SetActive(i < count);
            if (i >= count)
                continue;
            bool movable = i >= walls.Count;
            Rect area = movable ? boxes[i - walls.Count] : walls[i];
            Renderer meshRenderer = box.GetComponent<Renderer>();
            meshRenderer.enabled = movable || drawWalls;
            meshRenderer.SetPropertyBlock(movable ? obstacleAppearance : null);
            box.transform.SetPositionAndRotation(new Vector3(area.center.x, 0.2f, area.center.y),
                                                 Quaternion.identity);
            box.transform.localScale = new Vector3(area.width, 0.4f, area.height);
            obstacles.Add(new Bounds(box.transform.position, box.transform.localScale));
        }
        Physics.SyncTransforms();
    }

    private bool BeginPreviewEpisode(int index)
    {
        blocked = index == 5;
        bool junction = index >= 9;
        bool turnLeft = index == 9 || index == 11;
        Vector3 extraWidth = junction ? Vector3.right * 0.95f : Vector3.zero;
        leftWall.transform.position = leftWallPosition - extraWidth;
        rightWall.transform.position = rightWallPosition + extraWidth;
        foreach (BoxCollider box in layoutBoxes)
            box.gameObject.SetActive(false);
        PlacePreviewLayout(index);
        if (junction)
        {
            PlaceBox(6, 0f, 6.05f, 4.3f, 0.1f);
            PlaceBox(7, 0f, -0.05f, 4.3f, 0.1f);
        }
        Physics.SyncTransforms();
        obstacles.Clear();
        foreach (BoxCollider box in layoutBoxes)
            if (box.gameObject.activeSelf)
                obstacles.Add(box.bounds);
        lidar.SetNoiseSeed(null);
        recorder.Intent = junction ? (turnLeft ? NavigationIntent.Left : NavigationIntent.Right)
                                   : NavigationIntent.Follow;
        Vector2 start = new Vector2(body.position.x, body.position.z);
        var area = Rect.MinMaxRect(leftWall.bounds.max.x + Clearance, Clearance,
                                   rightWall.bounds.min.x - Clearance, 6f - Clearance);
        Vector2 goal = blocked ? new Vector2(0f, obstacles[0].min.z - Clearance - 0.15f) :
                       junction ? new Vector2(turnLeft ? -1.62f : 1.62f, 4.95f) :
                                  new Vector2(0f, FinishZ);
        if (!TeacherRoutePlanner.TryPlan(start, goal, area, Clearance, obstacles, index == 3, route))
        {
            Debug.LogWarning($"{EpisodeProgress} No teacher path for preview {Labels[index]}.", this);
            return false;
        }
        StartEpisodeRecording("teacher-v3-" + Labels[index]);
        return true;
    }

    private void PlacePreviewLayout(int index)
    {
        if (index >= 1 && index <= 5)
        {
            float x = index == 1 ? 0.43f : index == 2 ? -0.43f : 0f;
            float width = index == 5 ? 2.4f : 0.65f;
            PlaceBox(0, x, 3.2f, width, 0.55f);
        }
        if (index == 6)
        {
            PlaceBox(0, -0.49f, 2.25f, 1.42f, 0.5f);
            PlaceBox(1, 0.49f, 4.05f, 1.42f, 0.5f);
        }
        if (index == 7)
        {
            PlaceBox(0, -0.56f, 3.25f, 1.28f, 0.20f);
            PlaceBox(1, 1.14f, 3.25f, 0.12f, 0.20f);
        }
        if (index == 8)
        {
            PlaceBox(0, -0.44f, 2.25f, 0.48f, 0.5f);
            PlaceBox(1, 0.78f, 3.55f, 0.53f, 0.56f);
            PlaceBox(2, -0.73f, 4.45f, 0.44f, 0.45f);
        }

        if (index >= 9)
        {
            const float stemLength = 3.8f;
            PlaceBox(0, -1.47f, stemLength / 2f, 1.36f, stemLength);
            PlaceBox(1, 1.47f, stemLength / 2f, 1.36f, stemLength);
            if (index >= 11)
                PlaceBox(4, 0.24f, 2.45f, 0.28f, 0.42f);
        }
    }

    private void PlaceBox(int index, float x, float z, float width, float length)
    {
        BoxCollider box = layoutBoxes[index];
        box.transform.SetPositionAndRotation(new Vector3(x, 0.2f, z), Quaternion.identity);
        box.transform.localScale = new Vector3(width, 0.4f, length);
        box.gameObject.SetActive(true);
    }

    private void FixedUpdate()
    {
        if (episode < 0)
            return;

        frameCapture?.Sample();
        float dt = Time.fixedDeltaTime;
        elapsed += dt;
        if (elapsed >= disturbanceTime)
        {
            // Perturb the plant without changing the command the teacher is asked to imitate.
            disturbanceTime = float.PositiveInfinity;
            Vector3 right = body.rotation * Vector3.right;
            body.AddForce(right * (0.16f * disturbanceDirection), ForceMode.VelocityChange);
            body.AddTorque(Vector3.up * (0.50f * disturbanceDirection), ForceMode.VelocityChange);
        }

        if (recorder.CollisionCount > 0 || elapsed > (largeDirection && !postDoorTrial ? 115f : Timeout))
        {
            FinishEpisode(recorder.CollisionCount > 0 ? "collision" : "timeout");
            return;
        }

        if (evaluatedPolicy != null && !finishing)
        {
            UpdatePolicy(dt);
            return;
        }

        Vector2 position = new Vector2(body.position.x, body.position.z);
        Vector2 goal = route[route.Count - 1];
        float goalDistance = Vector2.Distance(position, goal);
        finishing |= UsesDirectionPilot ? PassedExit(position) : goalDistance < ArrivalDistance;
        if (finishing)
        {
            if (!blocked)
                recorder.Intent = NavigationIntent.Stop;
            ApplyCommand(0f, 0f, dt);
            bool stationary = body.linearVelocity.sqrMagnitude <= 0.01f * 0.01f &&
                              body.angularVelocity.sqrMagnitude <= 0.03f * 0.03f &&
                              forwardCommand == 0f && turnCommand == 0f;
            bool holdStop = procedural && blocked;
            settled = !holdStop || stationary ? settled + dt : 0f;
            if (settled >= (holdStop ? 0.7f : SettleTime) && forwardCommand == 0f && turnCommand == 0f)
                FinishEpisode(blocked ? "stopped" : "completed");
            return;
        }

        Vector2 delta = TeacherPathFollower.LookAheadPoint(route, position, LookAhead, ref segment) - position;
        Vector3 bodyForward = body.rotation * Vector3.forward;
        float heading = Mathf.Atan2(bodyForward.x, bodyForward.z) * Mathf.Rad2Deg;
        float target = Mathf.Atan2(delta.x, delta.y) * Mathf.Rad2Deg;
        float error = Mathf.DeltaAngle(heading, target);
        float turn = Mathf.Clamp(error / 110f, -0.21f, 0.21f);
        float alignment = Mathf.Clamp01(Mathf.Cos(error * Mathf.Deg2Rad));
        float speed = Mathf.Min(0.66f, goalDistance * 2.6f) * alignment * alignment;
        speed *= 1f - 1.6f * Mathf.Abs(turn);
        ApplyCommand(speed, turn, dt);
    }

    private void UpdatePolicy(float dt)
    {
        Vector2 position = new Vector2(body.position.x, body.position.z);
        float distance = Vector2.Distance(position, route[route.Count - 1]);
        if (!blocked && (UsesDirectionPilot ? PassedExit(position) : distance <= ArrivalDistance * 2f))
        {
            finishing = true;
            frameCapture?.ClearSteering();
            drive.SetMotorCommands(0f, 0f);
            recorder.Intent = NavigationIntent.Stop;
            return;
        }

        if (lidar.ScanTime >= firstPolicyScan && lidar.ScanTime > lastPolicyScan)
        {
            Vector2 context = drive.MotorCommands;
            if (evaluatedPolicy.UsesRelativeGoal && (waypointPilot || UsesDirectionPilot))
                context = waypointInput ? recorder.PilotWaypointContext : recorder.PilotGoalContext;
            else if (evaluatedPolicy.UsesRelativeGoal)
            {
                Vector2 goal = route[route.Count - 1];
                Vector3 local = Quaternion.Inverse(body.rotation) *
                    new Vector3(goal.x - position.x, 0f, goal.y - position.y);
                const float goalScale = 15f;
                context = new Vector2(Mathf.Clamp(local.z / goalScale, -1f, 1f),
                                      Mathf.Clamp(local.x / goalScale, -1f, 1f));
            }
            Vector2 commands = evaluatedPolicy.Predict(blindRanges == null ? lidar.Ranges : blindRanges,
                                                        recorder.Intent, context);
            drive.SetMotorCommands(commands.x, commands.y);
            frameCapture?.SetSteering(context, commands);
            if (noisePolicies != null)
            {
                for (int i = 0; i < noisePolicies.Length; i++)
                {
                    lidar.SampleNoise(noiseRealizations[i], noiseRanges);
                    Vector2 sampled = noisePolicies[i].Predict(noiseRanges, recorder.Intent, context);
                    noiseTurns[i] = (sampled.x - sampled.y) * 0.5f;
                }
                frameCapture.SetSteeringSamples(lidar.ScanTime, noiseTurns);
            }
            lastPolicyScan = lidar.ScanTime;
        }
        else if (Time.fixedTimeAsDouble - lastPolicyScan > 0.25)
        {
            frameCapture?.ClearSteering();
            drive.SetMotorCommands(0f, 0f);
        }

        // Judge the stop independently of the teacher route follower.
        bool stoppedNearBarrier = blocked && distance <= 0.30f && elapsed >= 1f &&
                                  body.linearVelocity.sqrMagnitude <= 0.015f * 0.015f &&
                                  body.angularVelocity.sqrMagnitude <= 0.03f * 0.03f;
        settled = stoppedNearBarrier ? settled + dt : 0f;
        if (settled >= 0.7f)
            FinishEpisode("stopped");
    }

    private bool PassedExit(Vector2 position) => position.y >= swapExit &&
        Mathf.Abs(position.x) <= (directionTransfer || directionChoice ? transferExitHalfWidth : 0.50f);

    private void ApplyCommand(float forward, float turn, float dt)
    {
        forwardCommand = Mathf.MoveTowards(forwardCommand, forward, 0.9f * dt);
        turnCommand = Mathf.MoveTowards(turnCommand, turn, 0.8f * dt);
        drive.SetMotorCommands(forwardCommand + turnCommand, forwardCommand - turnCommand);
    }

    private void FinishEpisode(string outcome)
    {
        frameCapture?.Finish(outcome);
        recorder.StopRecording(outcome);
        if (outcome == "completed" || outcome == "stopped")
            completedCount++;
        else
            failedCount++;
        AdvanceEpisode(episode + 1);
    }

    private void OnDisable()
    {
        if (episode < 0)
            return;

        recorder.StopRecording("interrupted");
        RestoreManualScene();
    }

    private void OnDestroy()
    {
        frameCapture?.Dispose();
        wallOutline?.Destroy();
    }

    private void RestoreManualScene()
    {
        episode = -1;
        Time.timeScale = originalTimeScale;
        lidar.SetNoiseSeed(null);
        ResetVehicle();
        foreach (BoxCollider box in layoutBoxes)
            box.gameObject.SetActive(false);
        if (procedural)
        {
            wallOutline?.Hide();
            foreach (BoxCollider box in graphBoxes)
                box.gameObject.SetActive(false);
            leftWall.gameObject.SetActive(originalWallsActive[0]);
            rightWall.gameObject.SetActive(originalWallsActive[1]);
            backWall.gameObject.SetActive(originalWallsActive[2]);
            endWall.gameObject.SetActive(originalWallsActive[3]);
            floor.position = floorPosition;
            floor.localScale = floorScale;
            overview.transform.position = cameraPosition;
            overview.orthographicSize = cameraSize;
        }
        obstacle.gameObject.SetActive(originalObstacleActive);
        leftWall.transform.position = leftWallPosition;
        rightWall.transform.position = rightWallPosition;
        manual.enabled = true;
        evaluatedPolicy = null;
        pdfPreview = null;
    }

    private void ResetVehicle()
    {
        drive.SetMotorCommands(0f, 0f);
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.position = startPosition;
        body.rotation = startRotation;
    }
}

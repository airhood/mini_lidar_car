using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public sealed class RearDrive : MonoBehaviour
{
    [SerializeField, Min(0.01f)] private float track = 0.24f;
    [SerializeField] private float rearAxleZ = -0.12f;
    [SerializeField, Min(0.01f)] private float topSpeed = 0.45f;
    [SerializeField, Min(0.01f)] private float responseTime = 0.20f;
    [SerializeField, Min(0.01f)] private float maxWheelForce = 1.5f;

    [SerializeField, Min(0.01f)] private float stopResponseTime = 0.065f;
    [SerializeField, Min(0.01f)] private float maxBrakeForce = 3f;
    [SerializeField, Min(0.01f)] private float lateralResponseTime = 0.075f;
    [SerializeField, Min(0f)] private float yawBrake = 12f;

    private Rigidbody body;
    private float leftCommand;
    private float rightCommand;

    public Vector2 MotorCommands => new Vector2(leftCommand, rightCommand);

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
    }

    public void SetMotorCommands(float left, float right)
    {
        leftCommand = Mathf.Clamp(left, -1f, 1f);
        rightCommand = Mathf.Clamp(right, -1f, 1f);
    }

    private void FixedUpdate()
    {
        Vector3 forward = body.rotation * Vector3.forward;
        Vector3 right = body.rotation * Vector3.right;
        Vector3 axle = body.position + forward * rearAxleZ;
        DriveWheel(axle - right * (track * 0.5f), forward, leftCommand);
        DriveWheel(axle + right * (track * 0.5f), forward, rightCommand);

        // Approximate the rear axle's lateral grip. The front caster is passive.
        float sidewaysSpeed = Vector3.Dot(body.GetPointVelocity(axle), right);
        float sideForce = Mathf.Clamp(-sidewaysSpeed * body.mass / lateralResponseTime,
                                      -2f * maxWheelForce, 2f * maxWheelForce);
        body.AddForceAtPosition(right * sideForce, axle);

        if (Mathf.Abs(leftCommand - rightCommand) < 0.001f)
            body.AddTorque(-Vector3.up * (body.angularVelocity.y * yawBrake),
                           ForceMode.Acceleration);
    }

    private void DriveWheel(Vector3 point, Vector3 forward, float command)
    {
        float speed = Vector3.Dot(body.GetPointVelocity(point), forward);
        bool stopping = Mathf.Abs(leftCommand) < 0.001f && Mathf.Abs(rightCommand) < 0.001f;
        float time = stopping ? stopResponseTime : responseTime;
        float limit = stopping ? maxBrakeForce : maxWheelForce;
        float force = (command * topSpeed - speed) * body.mass / (2f * time);
        body.AddForceAtPosition(forward * Mathf.Clamp(force, -limit, limit), point);
    }
}

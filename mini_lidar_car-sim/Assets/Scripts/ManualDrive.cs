using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(RearDrive))]
public sealed class ManualDrive : MonoBehaviour
{
    private RearDrive drive;

    private void Awake()
    {
        drive = GetComponent<RearDrive>();
    }

    private void Update()
    {
        Keyboard keyboard = Keyboard.current;
        if (keyboard == null)
        {
            drive.SetMotorCommands(0f, 0f);
            return;
        }

        float forward = (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed ? 1 : 0)
                      - (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed ? 1 : 0);
        float turn = (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed ? 1 : 0)
                   - (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed ? 1 : 0);
        float left = forward + turn;
        float right = forward - turn;
        float scale = Mathf.Max(1f, Mathf.Abs(left), Mathf.Abs(right));
        drive.SetMotorCommands(left / scale, right / scale);
    }

    private void OnDisable()
    {
        drive.SetMotorCommands(0f, 0f);
    }
}

using System;
using UnityEditor;
using UnityEditor.SceneManagement;

public static class TeacherBatch
{
    public static void Run() => RunScene("ManualDrive");
    public static void RunEvaluation() => RunScene("PolicyEvaluation");

    private static void RunScene(string scene)
    {
        if (!Array.Exists(Environment.GetCommandLineArgs(), arg => arg == "-teacherCollect"))
            throw new ArgumentException("TeacherBatch.Run requires -teacherCollect.");

        EditorSceneManager.OpenScene($"Assets/Scenes/{scene}.unity");
        EditorApplication.EnterPlaymode();
    }
}

using System.Collections.Generic;
using UnityEngine;

internal static class TeacherPathFollower
{
    public static Vector2 LookAheadPoint(IReadOnlyList<Vector2> route, Vector2 position,
                                         float lookAhead, ref int segment)
    {
        while (segment + 2 < route.Count)
        {
            Vector2 a = route[segment];
            Vector2 b = route[segment + 1];
            Vector2 c = route[segment + 2];
            float currentProgress = Projection(position, a, b);
            float nextProgress = Projection(position, b, c);
            Vector2 currentPoint = Vector2.Lerp(a, b, currentProgress);
            Vector2 nextPoint = Vector2.Lerp(b, c, nextProgress);
            // A rounded corner can enter the next segment without reaching the old endpoint.
            bool enteredNext = nextProgress > 0f &&
                (position - nextPoint).sqrMagnitude + 0.0025f < (position - currentPoint).sqrMagnitude;
            if (currentProgress < 0.98f && !enteredNext)
                break;
            segment++;
        }

        Vector2 cursor = Vector2.Lerp(route[segment], route[segment + 1],
                                      Projection(position, route[segment], route[segment + 1]));
        for (int i = segment + 1; i < route.Count; i++)
        {
            Vector2 end = route[i];
            float length = Vector2.Distance(cursor, end);
            if (length > lookAhead)
                return Vector2.Lerp(cursor, end, lookAhead / length);
            lookAhead -= length;
            cursor = end;
        }
        return cursor;
    }

    private static float Projection(Vector2 point, Vector2 start, Vector2 end)
    {
        Vector2 delta = end - start;
        return delta.sqrMagnitude > 0f
            ? Vector2.Dot(point - start, delta) / delta.sqrMagnitude
            : 1f;
    }
}

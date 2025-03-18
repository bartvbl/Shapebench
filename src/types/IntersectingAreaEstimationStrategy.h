#pragma once

namespace ShapeBench {
    enum class IntersectingAreaEstimationStrategy {
        // If your method uses a cylindrical or spherical support volume, you should use these
        FAST_CYLINDRICAL,
        FAST_SPHERICAL,

        // If your support volume has a different shape, but the intersecting area can still be computed reasonably quickly,
        // you should pick this option and implement the computeIntersectingAreaCustom() method
        CUSTOM,

        // If none of the above are applicable, you can estimate area by counting the number of points intersecting
        // the support volume around your descriptor. This is EXTREMELY slow, so avoid this option at all costs.
        // Requires that you implement the isPointInSupportVolume() method
        SLOW_MONTE_CARLO_ESTIMATION
    };
}

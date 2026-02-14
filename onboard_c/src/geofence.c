/**
 * @file geofence.c
 * @brief Geofence safety boundary checks.
 */
#include "geofence.h"
#include "tile_math.h"

bool vps_geofence_contains(const vps_geofence_t *fence, vps_geopoint_t point) {
    if (fence->type == VPS_FENCE_CIRCLE) {
        double dist = vps_haversine_km(fence->center, point);
        return dist <= (fence->radius_km - fence->margin_km);
    } else {
        /* Rectangle */
        double dlat = vps_haversine_km(fence->center,
            (vps_geopoint_t){point.lat, fence->center.lon});
        double dlon = vps_haversine_km(fence->center,
            (vps_geopoint_t){fence->center.lat, point.lon});

        if (point.lat < fence->center.lat) dlat = -dlat;
        if (point.lon < fence->center.lon) dlon = -dlon;

        return (dlat >= -(fence->half_lat_km - fence->margin_km)) &&
               (dlat <=  (fence->half_lat_km - fence->margin_km)) &&
               (dlon >= -(fence->half_lon_km - fence->margin_km)) &&
               (dlon <=  (fence->half_lon_km - fence->margin_km));
    }
}

double vps_geofence_distance_km(const vps_geofence_t *fence, vps_geopoint_t point) {
    if (fence->type == VPS_FENCE_CIRCLE) {
        double dist = vps_haversine_km(fence->center, point);
        return fence->radius_km - dist;
    }
    /* Rect: distance to nearest edge */
    double dlat = vps_haversine_km(fence->center,
        (vps_geopoint_t){point.lat, fence->center.lon});
    double dlon = vps_haversine_km(fence->center,
        (vps_geopoint_t){fence->center.lat, point.lon});

    double margin_lat = fence->half_lat_km - dlat;
    double margin_lon = fence->half_lon_km - dlon;
    return (margin_lat < margin_lon) ? margin_lat : margin_lon;
}

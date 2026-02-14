/**
 * @file geofence.h
 * @brief Geofence safety boundary checks.
 */
#ifndef GEOFENCE_H
#define GEOFENCE_H

#include "vps_types.h"

typedef enum {
    VPS_FENCE_CIRCLE,
    VPS_FENCE_RECT,
} vps_fence_type_t;

typedef struct {
    vps_fence_type_t type;
    vps_geopoint_t center;
    double radius_km;        /* for circle */
    double margin_km;
    /* For rect: center ± half_lat_km, center ± half_lon_km */
    double half_lat_km;
    double half_lon_km;
} vps_geofence_t;

/** Check if point is inside geofence. */
bool vps_geofence_contains(const vps_geofence_t *fence, vps_geopoint_t point);

/** Distance to nearest fence boundary in km (negative = outside). */
double vps_geofence_distance_km(const vps_geofence_t *fence, vps_geopoint_t point);

#endif /* GEOFENCE_H */

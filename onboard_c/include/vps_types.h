/**
 * @file vps_types.h
 * @brief Common types for the VPS onboard system.
 */
#ifndef VPS_TYPES_H
#define VPS_TYPES_H

#include <stdbool.h>
#include <stdint.h>

/** GPS coordinate. */
typedef struct {
    double lat;
    double lon;
} vps_geopoint_t;

/** Slippy map tile coordinate. */
typedef struct {
    int z;
    int x;
    int y;
} vps_tile_coord_t;

/** Pixel position within a tile (0..255). */
typedef struct {
    double x;
    double y;
} vps_pixel_t;

/** 2D velocity in m/s (north, east). */
typedef struct {
    double vn;
    double ve;
} vps_velocity_t;

/** Position fix source. */
typedef enum {
    VPS_SOURCE_NONE = 0,
    VPS_SOURCE_VISUAL = 1,
    VPS_SOURCE_EKF_PREDICT = 2,
    VPS_SOURCE_DEAD_RECKONING = 3,
} vps_source_t;

/** Position fix quality (NMEA-style). */
typedef enum {
    VPS_FIX_NONE = 0,
    VPS_FIX_VISUAL = 1,
    VPS_FIX_EKF = 2,
    VPS_FIX_DR = 3,
} vps_fix_quality_t;

#define VPS_TILE_SIZE 256
#define VPS_EARTH_CIRCUMFERENCE_M 40075016.686
#define VPS_MAX_MERCATOR_LAT 85.0511287798

#endif /* VPS_TYPES_H */

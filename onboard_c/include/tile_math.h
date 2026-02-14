/**
 * @file tile_math.h
 * @brief GPS ↔ tile ↔ pixel coordinate conversions.
 */
#ifndef TILE_MATH_H
#define TILE_MATH_H

#include "vps_types.h"

/** Convert GPS to slippy map tile coordinates. */
vps_tile_coord_t vps_gps_to_tile(vps_geopoint_t point, int zoom);

/** Get GPS coordinate of tile center. */
vps_geopoint_t vps_tile_center(vps_tile_coord_t tile);

/** Haversine distance in km between two points. */
double vps_haversine_km(vps_geopoint_t a, vps_geopoint_t b);

/** Ground resolution in meters per pixel. */
double vps_meters_per_pixel(double lat, int zoom);

/** Number of tiles covering a radius around a center point. */
int vps_tiles_in_radius(vps_geopoint_t center, double radius_km, int zoom,
                        vps_tile_coord_t *out, int max_out);

#endif /* TILE_MATH_H */

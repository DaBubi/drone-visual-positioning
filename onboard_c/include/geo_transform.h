/**
 * @file geo_transform.h
 * @brief Pixel ↔ tile ↔ GPS coordinate transformations.
 */
#ifndef GEO_TRANSFORM_H
#define GEO_TRANSFORM_H

#include "vps_types.h"

/** Convert a pixel within a tile to GPS. */
vps_geopoint_t vps_tile_pixel_to_gps(vps_tile_coord_t tile, vps_pixel_t pixel);

/** Convert GPS to tile + pixel within tile. */
void vps_gps_to_tile_pixel(vps_geopoint_t point, int zoom,
                           vps_tile_coord_t *tile_out, vps_pixel_t *pixel_out);

/**
 * Extract GPS from a 3x3 homography (drone→tile).
 * H is row-major: H[0..8] = [h00,h01,h02, h10,h11,h12, h20,h21,h22]
 */
vps_geopoint_t vps_homography_to_gps(const double H[9],
                                     vps_tile_coord_t tile,
                                     double cx, double cy);

/** Convert pixel displacement to meters. */
double vps_pixel_distance_to_meters(double dx, double dy, double lat, int zoom);

#endif /* GEO_TRANSFORM_H */

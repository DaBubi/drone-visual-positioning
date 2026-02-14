/**
 * @file geo_transform.c
 * @brief Pixel ↔ tile ↔ GPS coordinate transformations.
 */
#include "geo_transform.h"
#include "tile_math.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

vps_geopoint_t vps_tile_pixel_to_gps(vps_tile_coord_t tile, vps_pixel_t pixel) {
    double n = pow(2.0, tile.z);
    double global_x = tile.x + pixel.x / VPS_TILE_SIZE;
    double global_y = tile.y + pixel.y / VPS_TILE_SIZE;

    vps_geopoint_t p;
    p.lon = global_x / n * 360.0 - 180.0;
    p.lat = atan(sinh(M_PI * (1.0 - 2.0 * global_y / n))) * 180.0 / M_PI;
    return p;
}

void vps_gps_to_tile_pixel(vps_geopoint_t point, int zoom,
                           vps_tile_coord_t *tile_out, vps_pixel_t *pixel_out) {
    double n = pow(2.0, zoom);
    double lat_rad = point.lat * M_PI / 180.0;

    double x_global = (point.lon + 180.0) / 360.0 * n;
    double y_global = (1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / M_PI) / 2.0 * n;

    tile_out->z = zoom;
    tile_out->x = (int)x_global;
    tile_out->y = (int)y_global;

    pixel_out->x = (x_global - tile_out->x) * VPS_TILE_SIZE;
    pixel_out->y = (y_global - tile_out->y) * VPS_TILE_SIZE;
}

vps_geopoint_t vps_homography_to_gps(const double H[9],
                                     vps_tile_coord_t tile,
                                     double cx, double cy) {
    /* Project center through homography: dst = H * [cx, cy, 1] */
    double dx = H[0] * cx + H[1] * cy + H[2];
    double dy = H[3] * cx + H[4] * cy + H[5];
    double dw = H[6] * cx + H[7] * cy + H[8];

    if (fabs(dw) < 1e-10) {
        vps_geopoint_t zero = {0.0, 0.0};
        return zero;
    }

    vps_pixel_t px = {dx / dw, dy / dw};
    return vps_tile_pixel_to_gps(tile, px);
}

double vps_pixel_distance_to_meters(double dx, double dy, double lat, int zoom) {
    double mpp = vps_meters_per_pixel(lat, zoom);
    return sqrt(dx * dx + dy * dy) * mpp;
}

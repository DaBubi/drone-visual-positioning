/**
 * @file tile_math.c
 * @brief GPS ↔ tile ↔ pixel coordinate conversions.
 */
#include "tile_math.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

vps_tile_coord_t vps_gps_to_tile(vps_geopoint_t point, int zoom) {
    double n = pow(2.0, zoom);
    double lat_rad = point.lat * M_PI / 180.0;

    vps_tile_coord_t t;
    t.z = zoom;
    t.x = (int)((point.lon + 180.0) / 360.0 * n);
    t.y = (int)((1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / M_PI) / 2.0 * n);

    /* Clamp to valid range */
    int max_tile = (int)n - 1;
    if (t.x < 0) t.x = 0;
    if (t.x > max_tile) t.x = max_tile;
    if (t.y < 0) t.y = 0;
    if (t.y > max_tile) t.y = max_tile;

    return t;
}

vps_geopoint_t vps_tile_center(vps_tile_coord_t tile) {
    double n = pow(2.0, tile.z);
    double lon = (tile.x + 0.5) / n * 360.0 - 180.0;
    double lat_rad = atan(sinh(M_PI * (1.0 - 2.0 * (tile.y + 0.5) / n)));
    vps_geopoint_t p;
    p.lat = lat_rad * 180.0 / M_PI;
    p.lon = lon;
    return p;
}

double vps_haversine_km(vps_geopoint_t a, vps_geopoint_t b) {
    double dlat = (b.lat - a.lat) * M_PI / 180.0;
    double dlon = (b.lon - a.lon) * M_PI / 180.0;
    double lat1 = a.lat * M_PI / 180.0;
    double lat2 = b.lat * M_PI / 180.0;

    double s = sin(dlat / 2.0);
    double c = sin(dlon / 2.0);
    double h = s * s + cos(lat1) * cos(lat2) * c * c;
    return 6371.0 * 2.0 * asin(sqrt(h));
}

double vps_meters_per_pixel(double lat, int zoom) {
    return (VPS_EARTH_CIRCUMFERENCE_M * cos(lat * M_PI / 180.0))
           / (VPS_TILE_SIZE * pow(2.0, zoom));
}

int vps_tiles_in_radius(vps_geopoint_t center, double radius_km, int zoom,
                        vps_tile_coord_t *out, int max_out) {
    /* Compute bounding box in tile coordinates */
    double dlat = radius_km / 111.32;
    double dlon = radius_km / (111.32 * cos(center.lat * M_PI / 180.0));

    vps_geopoint_t nw = {center.lat + dlat, center.lon - dlon};
    vps_geopoint_t se = {center.lat - dlat, center.lon + dlon};

    vps_tile_coord_t t_nw = vps_gps_to_tile(nw, zoom);
    vps_tile_coord_t t_se = vps_gps_to_tile(se, zoom);

    int count = 0;
    for (int x = t_nw.x; x <= t_se.x && count < max_out; x++) {
        for (int y = t_nw.y; y <= t_se.y && count < max_out; y++) {
            out[count].z = zoom;
            out[count].x = x;
            out[count].y = y;
            count++;
        }
    }
    return count;
}

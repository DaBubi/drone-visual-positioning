/**
 * @file dead_reckoning.c
 * @brief Constant-velocity dead reckoning.
 */
#include "dead_reckoning.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void vps_dr_init(vps_dr_state_t *dr, double max_extrap_s, double hdop_growth_rate) {
    dr->has_reference = false;
    dr->ref_t = 0;
    dr->max_extrap_s = max_extrap_s;
    dr->hdop_growth_rate = hdop_growth_rate;
}

void vps_dr_update_ref(vps_dr_state_t *dr, vps_geopoint_t pos,
                       double vn, double ve, double hdop, double t) {
    dr->ref_pos = pos;
    dr->vn_mps = vn;
    dr->ve_mps = ve;
    dr->ref_hdop = hdop;
    dr->ref_t = t;
    dr->has_reference = true;
}

bool vps_dr_extrapolate(const vps_dr_state_t *dr, double t,
                        vps_geopoint_t *pos_out, double *hdop_out) {
    if (!dr->has_reference) return false;

    double dt = t - dr->ref_t;
    if (dt < 0 || dt > dr->max_extrap_s) return false;

    /* Convert m/s to degrees/s */
    double dlat = dr->vn_mps / 111320.0;
    double dlon = dr->ve_mps / (111320.0 * cos(dr->ref_pos.lat * M_PI / 180.0));

    pos_out->lat = dr->ref_pos.lat + dlat * dt;
    pos_out->lon = dr->ref_pos.lon + dlon * dt;
    *hdop_out = dr->ref_hdop + dr->hdop_growth_rate * dt;

    return true;
}

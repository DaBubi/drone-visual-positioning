/**
 * @file fusion.c
 * @brief Position fusion engine.
 */
#include "fusion.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void vps_fusion_init(vps_fusion_t *f, const vps_ekf_config_t *ekf_cfg,
                     double max_dr_s, vps_geofence_t *fence) {
    if (ekf_cfg)
        f->ekf_cfg = *ekf_cfg;
    else
        f->ekf_cfg = vps_ekf_default_config();

    vps_ekf_init(&f->ekf);
    vps_dr_init(&f->dr, max_dr_s, 2.0);
    f->fence = fence;
}

vps_fusion_output_t vps_fusion_update(vps_fusion_t *f,
                                      const vps_geopoint_t *visual,
                                      double hdop, double t) {
    vps_fusion_output_t out;
    out.has_position = false;
    out.position = (vps_geopoint_t){0, 0};
    out.hdop = 99.0;
    out.speed_mps = 0.0;
    out.heading_deg = 0.0;
    out.fix_quality = VPS_FIX_NONE;
    out.source = VPS_SOURCE_NONE;
    out.geofence_ok = true;
    out.ekf_accepted = false;

    if (visual) {
        /* Case 1: Visual fix */
        out.ekf_accepted = vps_ekf_update(&f->ekf, &f->ekf_cfg, *visual, hdop, t);
        if (f->ekf.initialized) {
            out.position = vps_ekf_position(&f->ekf);
            out.hdop = hdop;
            out.source = VPS_SOURCE_VISUAL;
            out.fix_quality = VPS_FIX_VISUAL;
            out.has_position = true;

            /* Update dead reckoning reference */
            vps_velocity_t vel = vps_ekf_velocity(&f->ekf);
            vps_dr_update_ref(&f->dr, out.position, vel.vn, vel.ve, hdop, t);
        }
    } else if (f->ekf.initialized) {
        /* Case 2: EKF prediction */
        vps_geopoint_t pred = vps_ekf_predict(&f->ekf, t);
        if (pred.lat != 0.0 || pred.lon != 0.0) {
            out.position = pred;
            out.hdop = 3.0;
            out.source = VPS_SOURCE_EKF_PREDICT;
            out.fix_quality = VPS_FIX_EKF;
            out.has_position = true;
        }
    }

    if (!out.has_position) {
        /* Case 3: Dead reckoning */
        vps_geopoint_t dr_pos;
        double dr_hdop;
        if (vps_dr_extrapolate(&f->dr, t, &dr_pos, &dr_hdop)) {
            out.position = dr_pos;
            out.hdop = dr_hdop;
            out.source = VPS_SOURCE_DEAD_RECKONING;
            out.fix_quality = VPS_FIX_DR;
            out.has_position = true;
        }
    }

    /* Geofence check */
    if (out.has_position && f->fence) {
        out.geofence_ok = vps_geofence_contains(f->fence, out.position);
        if (!out.geofence_ok) {
            out.has_position = false;
            out.fix_quality = VPS_FIX_NONE;
            out.source = VPS_SOURCE_NONE;
        }
    }

    /* Speed and heading */
    if (f->ekf.initialized) {
        out.speed_mps = vps_ekf_speed(&f->ekf);
        if (out.speed_mps > 0.5) {
            vps_velocity_t vel = vps_ekf_velocity(&f->ekf);
            double vn_ms = vel.vn * 111320.0;
            double ve_ms = vel.ve * 111320.0 * cos(f->ekf.x[0] * M_PI / 180.0);
            out.heading_deg = fmod(atan2(ve_ms, vn_ms) * 180.0 / M_PI + 360.0, 360.0);
        }
    }

    return out;
}

void vps_fusion_reset(vps_fusion_t *f) {
    vps_ekf_reset(&f->ekf);
    vps_dr_init(&f->dr, f->dr.max_extrap_s, f->dr.hdop_growth_rate);
}

/**
 * @file fusion.h
 * @brief Position fusion: visual + EKF + dead reckoning + geofence.
 */
#ifndef FUSION_H
#define FUSION_H

#include "vps_types.h"
#include "ekf.h"
#include "dead_reckoning.h"
#include "geofence.h"

/** Fusion output for one frame. */
typedef struct {
    vps_geopoint_t position;
    double hdop;
    double speed_mps;
    double heading_deg;
    vps_fix_quality_t fix_quality;
    vps_source_t source;
    bool geofence_ok;
    bool ekf_accepted;
    bool has_position;
} vps_fusion_output_t;

/** Fusion engine state. */
typedef struct {
    vps_ekf_state_t ekf;
    vps_ekf_config_t ekf_cfg;
    vps_dr_state_t dr;
    vps_geofence_t *fence;  /* NULL if no geofence */
} vps_fusion_t;

/** Initialize fusion engine. */
void vps_fusion_init(vps_fusion_t *f, const vps_ekf_config_t *ekf_cfg,
                     double max_dr_s, vps_geofence_t *fence);

/** Process one frame. visual may be NULL if no match. */
vps_fusion_output_t vps_fusion_update(vps_fusion_t *f,
                                      const vps_geopoint_t *visual,
                                      double hdop, double t);

/** Reset all state. */
void vps_fusion_reset(vps_fusion_t *f);

#endif /* FUSION_H */

/**
 * @file msp.c
 * @brief MSP protocol GPS injection.
 */
#include "msp.h"
#include <string.h>

vps_msp_gps_t vps_msp_from_position(vps_geopoint_t pos, double speed_mps,
                                    double heading_deg, double hdop,
                                    bool has_fix) {
    vps_msp_gps_t g;
    g.fix_type = has_fix ? 2 : 0;
    g.num_sat = has_fix ? 12 : 0;
    g.lat = (int32_t)(pos.lat * 1e7);
    g.lon = (int32_t)(pos.lon * 1e7);
    g.altitude_m = 0;
    g.speed_cms = (uint16_t)(speed_mps * 100.0);
    g.heading_deg10 = (uint16_t)(heading_deg * 10.0);
    g.hdop = (uint16_t)(hdop * 100.0);
    return g;
}

uint8_t vps_msp_checksum(const uint8_t *data, size_t len) {
    uint8_t cs = 0;
    for (size_t i = 0; i < len; i++) {
        cs ^= data[i];
    }
    return cs;
}

int vps_msp_encode(uint8_t *out, const vps_msp_gps_t *gps) {
    /* Header: $M< */
    out[0] = '$';
    out[1] = 'M';
    out[2] = '<';
    out[3] = MSP_GPS_PAYLOAD;
    out[4] = MSP_CMD_SET_RAW_GPS;

    /* Payload (little-endian) */
    uint8_t *p = &out[5];
    p[0] = gps->fix_type;
    p[1] = gps->num_sat;

    /* lat (int32 LE) */
    p[2] = (gps->lat >>  0) & 0xFF;
    p[3] = (gps->lat >>  8) & 0xFF;
    p[4] = (gps->lat >> 16) & 0xFF;
    p[5] = (gps->lat >> 24) & 0xFF;

    /* lon (int32 LE) */
    p[6] = (gps->lon >>  0) & 0xFF;
    p[7] = (gps->lon >>  8) & 0xFF;
    p[8] = (gps->lon >> 16) & 0xFF;
    p[9] = (gps->lon >> 24) & 0xFF;

    /* altitude (int16 LE) */
    p[10] = (gps->altitude_m >>  0) & 0xFF;
    p[11] = (gps->altitude_m >>  8) & 0xFF;

    /* speed (uint16 LE) */
    p[12] = (gps->speed_cms >>  0) & 0xFF;
    p[13] = (gps->speed_cms >>  8) & 0xFF;

    /* heading (uint16 LE) */
    p[14] = (gps->heading_deg10 >>  0) & 0xFF;
    p[15] = (gps->heading_deg10 >>  8) & 0xFF;

    /* hdop (uint16 LE) */
    p[16] = (gps->hdop >>  0) & 0xFF;
    p[17] = (gps->hdop >>  8) & 0xFF;

    /* Checksum: XOR of [len, cmd, payload...] */
    out[MSP_GPS_FRAME_SIZE - 1] = vps_msp_checksum(&out[3], MSP_GPS_PAYLOAD + 2);

    return MSP_GPS_FRAME_SIZE;
}

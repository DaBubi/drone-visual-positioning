/**
 * @file nmea.c
 * @brief NMEA sentence generation.
 */
#include "nmea.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

uint8_t vps_nmea_checksum(const char *sentence) {
    uint8_t cs = 0;
    const char *p = sentence;

    /* Skip leading $ */
    if (*p == '$') p++;

    /* XOR until * or end */
    while (*p && *p != '*') {
        cs ^= (uint8_t)*p;
        p++;
    }
    return cs;
}

/** Convert decimal degrees to NMEA ddmm.mmmm format. */
static void deg_to_nmea(double deg, int is_lon, char *buf, size_t buflen, char *dir) {
    double abs_deg = fabs(deg);
    int d = (int)abs_deg;
    double m = (abs_deg - d) * 60.0;

    if (is_lon) {
        snprintf(buf, buflen, "%03d%08.5f", d, m);
        *dir = (deg >= 0) ? 'E' : 'W';
    } else {
        snprintf(buf, buflen, "%02d%08.5f", d, m);
        *dir = (deg >= 0) ? 'N' : 'S';
    }
}

int vps_format_gga(char *buf, size_t buflen,
                   vps_geopoint_t pos, int fix_quality,
                   double hdop, double altitude) {
    time_t now = time(NULL);
    struct tm *utc = gmtime(&now);

    char lat_str[20], lon_str[20];
    char lat_dir, lon_dir;
    deg_to_nmea(pos.lat, 0, lat_str, sizeof(lat_str), &lat_dir);
    deg_to_nmea(pos.lon, 1, lon_str, sizeof(lon_str), &lon_dir);

    char body[128];
    snprintf(body, sizeof(body),
             "GPGGA,%02d%02d%02d.00,%s,%c,%s,%c,%d,08,%.1f,%.1f,M,0.0,M,,",
             utc->tm_hour, utc->tm_min, utc->tm_sec,
             lat_str, lat_dir, lon_str, lon_dir,
             fix_quality, hdop, altitude);

    uint8_t cs = 0;
    for (const char *p = body; *p; p++) cs ^= (uint8_t)*p;

    return snprintf(buf, buflen, "$%s*%02X\r\n", body, cs);
}

int vps_format_rmc(char *buf, size_t buflen,
                   vps_geopoint_t pos, bool active,
                   double speed_knots, double heading_deg) {
    time_t now = time(NULL);
    struct tm *utc = gmtime(&now);

    char lat_str[20], lon_str[20];
    char lat_dir, lon_dir;
    deg_to_nmea(pos.lat, 0, lat_str, sizeof(lat_str), &lat_dir);
    deg_to_nmea(pos.lon, 1, lon_str, sizeof(lon_str), &lon_dir);

    char body[128];
    snprintf(body, sizeof(body),
             "GPRMC,%02d%02d%02d.00,%c,%s,%c,%s,%c,%.1f,%.1f,%02d%02d%02d,,,A",
             utc->tm_hour, utc->tm_min, utc->tm_sec,
             active ? 'A' : 'V',
             lat_str, lat_dir, lon_str, lon_dir,
             speed_knots, heading_deg,
             utc->tm_mday, utc->tm_mon + 1, utc->tm_year % 100);

    uint8_t cs = 0;
    for (const char *p = body; *p; p++) cs ^= (uint8_t)*p;

    return snprintf(buf, buflen, "$%s*%02X\r\n", body, cs);
}

#include "ForbiddenPointFinder.h"

extern "C" {

__declspec(dllexport) int kg_is_forbidden(const int8_t* board15x15, int x, int y) {
    if (x < 0 || x >= 15 || y < 0 || y >= 15) return 0;
    if (board15x15[y * 15 + x] != C_EMPTY) return 0;
    CForbiddenPointFinder finder(15);
    for (int py = 0; py < 15; ++py) {
        for (int px = 0; px < 15; ++px) {
            int8_t v = board15x15[py * 15 + px];
            if (v == C_BLACK || v == C_WHITE) {
                finder.SetStone(px, py, v);
            }
        }
    }
    return finder.isForbidden(x, y) ? 1 : 0;
}

__declspec(dllexport) void kg_get_forbidden_map(const int8_t* board15x15, uint8_t* out_mask15x15) {
    CForbiddenPointFinder finder(15);
    for (int py = 0; py < 15; ++py) {
        for (int px = 0; px < 15; ++px) {
            int8_t v = board15x15[py * 15 + px];
            if (v == C_BLACK || v == C_WHITE) {
                finder.SetStone(px, py, v);
            }
        }
    }
    for (int py = 0; py < 15; ++py) {
        for (int px = 0; px < 15; ++px) {
            if (board15x15[py * 15 + px] == C_EMPTY) {
                out_mask15x15[py * 15 + px] = finder.isForbidden(px, py) ? 1 : 0;
            } else {
                out_mask15x15[py * 15 + px] = 0;
            }
        }
    }
}

}

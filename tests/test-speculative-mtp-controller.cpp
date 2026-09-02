#include "speculative-mtp-controller.h"

#include <cassert>

int main() {
    common_speculative_mtp_controller_config cfg;
    cfg.mode = common_speculative_mtp_controller_mode::BATCH;

    const auto d1 = common_speculative_mtp_controller_select(cfg, 1, 4);
    const auto d2 = common_speculative_mtp_controller_select(cfg, 2, 4);
    const auto d3 = common_speculative_mtp_controller_select(cfg, 3, 4);
    const auto d4 = common_speculative_mtp_controller_select(cfg, 4, 4);
    const auto d7 = common_speculative_mtp_controller_select(cfg, 7, 4);
    const auto d8 = common_speculative_mtp_controller_select(cfg, 8, 4);
    assert(d1.depth == 4 && !d1.limited_by_batch);
    assert(d2.depth == 3 && d2.limited_by_batch);
    assert(d3.depth == 3 && d3.limited_by_batch);
    assert(d4.depth == 2 && d4.limited_by_batch);
    assert(d7.depth == 4 && !d7.limited_by_batch);
    assert(d8.depth == 4 && !d8.limited_by_batch);

    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 1, 4) == 4);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 2, 4) == 3);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 4, 4) == 2);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 8, 4) == 4);

    cfg.mode = common_speculative_mtp_controller_mode::TRACE;
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 8, 4) == 4);
    assert(common_speculative_mtp_controller_select(cfg, 8, 4).depth == 4);

    cfg.mode = common_speculative_mtp_controller_mode::OFF;
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 8, 4) == 4);

    cfg.mode = common_speculative_mtp_controller_mode::BATCH;
    cfg.max_depth = 3;
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 1, 4) == 3);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 2, 4) == 3);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 4, 4) == 2);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 8, 4) == 3);
    assert(common_speculative_mtp_controller_pre_draft_cap(cfg, 2, 0) == 0);

    return 0;
}

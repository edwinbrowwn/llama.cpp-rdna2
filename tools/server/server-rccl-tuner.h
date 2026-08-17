#pragma once

#include "common.h"

// Select the validated per-size RCCL tuner before any backend communicator or
// HIP graph is initialized.  Failure or ineligibility is always a no-op and
// leaves RCCL Auto in control.
void server_rccl_tuner_prepare(const common_params & params, const char * argv0);

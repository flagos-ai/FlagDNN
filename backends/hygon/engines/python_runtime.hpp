/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#pragma once

namespace flagdnn::hygon::detail {

// Call under the Hygon JIT initialization lock. A caller-owned interpreter is
// borrowed; only an interpreter created here receives Hygon exit handling.
void initialize_embedded_python();

// Register after extension imports, including a partially failed import. Exit
// handlers run in reverse registration order, ahead of those extensions' C++
// static destructors. A failed initialization may retry and register again.
void guard_python_shutdown_after_imports();

} // namespace flagdnn::hygon::detail

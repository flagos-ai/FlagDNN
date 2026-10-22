if(NOT DEFINED REFERENCE_SOURCE OR NOT EXISTS "${REFERENCE_SOURCE}")
  message(FATAL_ERROR
    "Convolution validation contract requires REFERENCE_SOURCE")
endif()
if(NOT DEFINED BENCHMARK_SOURCE OR NOT EXISTS "${BENCHMARK_SOURCE}")
  message(FATAL_ERROR
    "Convolution validation contract requires BENCHMARK_SOURCE")
endif()
if(NOT DEFINED RUNNER_SOURCE OR NOT EXISTS "${RUNNER_SOURCE}")
  message(FATAL_ERROR
    "Convolution validation contract requires RUNNER_SOURCE")
endif()

file(READ "${REFERENCE_SOURCE}" _reference)
file(READ "${BENCHMARK_SOURCE}" _benchmark)
file(READ "${RUNNER_SOURCE}" _runner)
string(REGEX REPLACE "[ \t\r\n]" "" _reference "${_reference}")
string(REGEX REPLACE "[ \t\r\n]" "" _benchmark "${_benchmark}")
string(REGEX REPLACE "[ \t\r\n]" "" _runner "${_runner}")

function(_require_contains content needle description)
  string(FIND "${content}" "${needle}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "${description}")
  endif()
endfunction()

_require_contains(
  "${_reference}"
  "check_hipdnn_status(status,stage)"
  "Private convolution status checks must preserve HipdnnStatusError")
_require_contains(
  "${_benchmark}"
  "result.data_type=FLAGDNN_DATA_FLOAT32"
  "FP16 convolution correctness specs must upcast to FP32")
_require_contains(
  "${_benchmark}"
  "source.binding_byte_offset%kSourceElementBytes!=0"
  "FP16 convolution oracle offsets must be element-aligned")
_require_contains(
  "${_benchmark}"
  "result.binding_byte_offset=offset_elements*kOracleElementBytes"
  "FP16 convolution oracle offsets must be scaled by element index")
_require_contains(
  "${_runner}"
  "HipdnnReferencePolicy::kCorrectnessOracle"
  "Benchmark runner lost the independent correctness policy")
_require_contains(
  "${_runner}"
  "HipdnnReferencePolicy::kPerformance"
  "Benchmark runner lost the independent performance policy")
_require_contains(
  "${_runner}"
  "candidate_buffers=performance_buffers.get()"
  "Performance timing must use its independent original-dtype buffers")
_require_contains(
  "${_runner}"
  "timed_reference_buffers=candidate_buffers"
  "Performance timing lost the accuracy-gated candidate buffers")
_require_contains(
  "${_runner}"
  "HIPDNN_PERFORMANCE_ACCURACY_EXHAUSTED"
  "FP16 numerical candidate exhaustion must retain its diagnosed cause")
_require_contains(
  "${_runner}"
  "if(fp32_correctness_oracle&&rejected_numerical_candidate)"
  "Only numerically exhausted FP16-upcast cases may use this fallback")
_require_contains(
  "${_runner}"
  "returnunavailable(\"provider=hipdnnphase=probe\"\"status=HIPDNN_PERFORMANCE_ACCURACY_EXHAUSTED"
  "FP16 numerical exhaustion must try the exact equivalent baseline before skipping")
_require_contains(
  "${_runner}"
  "sameprimitivecorrectnessoraclesucceeded"
  "Non-FP16 performance candidate exhaustion must remain a hard failure")

string(FIND "${_runner}" "HIPDNN_PERFORMANCE_ACCURACY_EXHAUSTED"
  _exhaustion_position)
string(FIND "${_runner}" "warmup(*flagdnn" _warmup_position)
if(_exhaustion_position EQUAL -1 OR _warmup_position EQUAL -1 OR
   NOT _exhaustion_position LESS _warmup_position)
  message(FATAL_ERROR
    "Numerical candidate exhaustion must return before warmup and sampling")
endif()

message(STATUS "Hygon convolution validation contract passed")

#include "ilogger.h"
#include <fstream>
#include <NvInfer.h>

namespace ILogger {
	const char* severityString(nvinfer1::ILogger::Severity severity) {
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			return "INTERNAL_ERROR";
		case nvinfer1::ILogger::Severity::kERROR:
			return "ERROR";
		case nvinfer1::ILogger::Severity::kWARNING:
			return "WARNING";
		case nvinfer1::ILogger::Severity::kINFO:
			return "INFO";
		case nvinfer1::ILogger::Severity::kVERBOSE:
			return "VERBOSE";
		default:
			return "UNKNOWN";
		}
	}

	void TrtLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
		if (severity <= Severity::kINFO) {
			if (severity == Severity::kWARNING) {
				printf("\033[33m%s: %s\033[0m\n", severityString(severity), msg);
			}
			else if (severity <= Severity::kERROR) {
				printf("\033[31m%s: %s\033[0m\n", severityString(severity), msg);
			}
			else {
				printf("%s: %s\n", severityString(severity), msg);
			}
		}
	}
}
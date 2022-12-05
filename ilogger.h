#ifndef ILOGGER_HPP
#include <NvInfer.h>

namespace ILogger {
	const char* severityString(nvinfer1::ILogger::Severity severity);

	class TrtLogger : public nvinfer1::ILogger {
	public:
		virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
	};
}

#endif // !ILOGGER_HPP


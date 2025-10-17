################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
./src/spdlog/fmt/bundled/format.cc \
./src/spdlog/fmt/bundled/ostream.cc \
./src/spdlog/fmt/bundled/posix.cc \
./src/spdlog/fmt/bundled/printf.cc 

CC_DEPS += \
./src/spdlog/fmt/bundled/format.d \
./src/spdlog/fmt/bundled/ostream.d \
./src/spdlog/fmt/bundled/posix.d \
./src/spdlog/fmt/bundled/printf.d 

OBJS += \
./src/spdlog/fmt/bundled/format.o \
./src/spdlog/fmt/bundled/ostream.o \
./src/spdlog/fmt/bundled/posix.o \
./src/spdlog/fmt/bundled/printf.o 


# Each subdirectory must supply rules for building sources it contributes
src/spdlog/fmt/bundled/%.o: ./src/spdlog/fmt/bundled/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++14 -DBOOST_DISABLE_ASSERTS -O3 -march=native -Wall -c -fmessage-length=0 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



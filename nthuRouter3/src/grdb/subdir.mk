################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
./src/grdb/RoutingComponent.cpp \
./src/grdb/RoutingRegion.cpp \
./src/grdb/parser.cpp 

OBJS += \
./src/grdb/RoutingComponent.o \
./src/grdb/RoutingRegion.o \
./src/grdb/parser.o 

CPP_DEPS += \
./src/grdb/RoutingComponent.d \
./src/grdb/RoutingRegion.d \
./src/grdb/parser.d 


# Each subdirectory must supply rules for building sources it contributes
src/grdb/%.o: ./src/grdb/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++14 -DBOOST_DISABLE_ASSERTS -O3 -march=native -Wall -c -fmessage-length=0 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
./src/router/Congestion.cpp \
./src/router/Construct_2d_tree.cpp \
./src/router/DataDef.cpp \
./src/router/Layerassignment.cpp \
./src/router/MM_mazeroute.cpp \
./src/router/MonotonicRouting.cpp \
./src/router/OutputGeneration.cpp \
./src/router/Post_processing.cpp \
./src/router/Range_router.cpp \
./src/router/Route.cpp \
./src/router/Route_2pinnets.cpp \
./src/router/flute4nthuroute.cpp \
./src/router/parameter.cpp 

OBJS += \
./src/router/Congestion.o \
./src/router/Construct_2d_tree.o \
./src/router/DataDef.o \
./src/router/Layerassignment.o \
./src/router/MM_mazeroute.o \
./src/router/MonotonicRouting.o \
./src/router/OutputGeneration.o \
./src/router/Post_processing.o \
./src/router/Range_router.o \
./src/router/Route.o \
./src/router/Route_2pinnets.o \
./src/router/flute4nthuroute.o \
./src/router/parameter.o 

CPP_DEPS += \
./src/router/Congestion.d \
./src/router/Construct_2d_tree.d \
./src/router/DataDef.d \
./src/router/Layerassignment.d \
./src/router/MM_mazeroute.d \
./src/router/MonotonicRouting.d \
./src/router/OutputGeneration.d \
./src/router/Post_processing.d \
./src/router/Range_router.d \
./src/router/Route.d \
./src/router/Route_2pinnets.d \
./src/router/flute4nthuroute.d \
./src/router/parameter.d 


# Each subdirectory must supply rules for building sources it contributes
src/router/%.o: ./src/router/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++14 -DBOOST_DISABLE_ASSERTS -O3 -march=native -Wall -c -fmessage-length=0 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



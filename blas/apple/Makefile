# Get Xcode developer path
DEVELOPER_PATH := $(shell xcode-select -p)
SDK_PATH := $(shell xcrun --show-sdk-path)

# Compiler settings
SWIFTC = xcrun -sdk macosx swiftc
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Directories
BUILD_DIR = build
METAL_DIR = $(BUILD_DIR)/metal

# Source files
SWIFT_SOURCES = matmul.swift example.swift
METAL_SOURCES = Shaders.metal

# Output files
EXECUTABLE = $(BUILD_DIR)/matmul_example
METALLIB_FILE = $(METAL_DIR)/default.metallib
METAL_AIR = $(METAL_DIR)/Shaders.air

# Flags
SWIFT_FLAGS = -O -framework Metal -framework MetalKit
METAL_FLAGS = -c -std=macos-metal2.4

# Check if xcrun exists
$(if $(shell which xcrun),,$(error "xcrun not found. Please install Xcode and Xcode Command Line Tools"))

# Default target
all: directories $(EXECUTABLE)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(METAL_DIR)

# Compile Metal shader to .metallib
$(METALLIB_FILE): $(METAL_SOURCES)
	$(METAL) $(METAL_FLAGS) $(METAL_SOURCES) -o $(METAL_AIR)
	$(METALLIB) $(METAL_AIR) -o $(METALLIB_FILE)

# Compile Swift code and link with Metal library
$(EXECUTABLE): $(METALLIB_FILE) $(SWIFT_SOURCES)
	$(SWIFTC) $(SWIFT_FLAGS) $(SWIFT_SOURCES) -o $(EXECUTABLE)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Run the example
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Print debug information
debug-info:
	@echo "Developer Path: $(DEVELOPER_PATH)"
	@echo "SDK Path: $(SDK_PATH)"
	@echo "Metal Command: $(METAL)"
	@echo "Metal Library Tool: $(METALLIB)"
	@which xcrun
	@xcrun -sdk macosx --find metal
	@xcrun -sdk macosx --find metallib

.PHONY: all clean run directories debug-info

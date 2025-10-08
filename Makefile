# Compiler and flags
MY_OPT := 
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17  -ggdb -O3 $(MY_OPT)

# Target executable name
TARGET := mm

# Source files (all .cpp in current directory and matrix subdirectory)
SRCS := $(wildcard *.cpp) $(wildcard dgemm/*.cpp) $(wildcard matrix/*.cpp) $(wildcard utils/*.cpp) $(wildcard cse260_hw1/*.cpp)

# Object files
OBJS := $(SRCS:.cpp=.o)

LD := g++

LDFLAGS := -lopenblas

# Default rule
all: $(TARGET)


# Explicit link step
$(TARGET): $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS) -lpthread -lm


# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET) matrix/*.o utils/*.o dgemm/*.o

# Phony targets
.PHONY: all clean

# Customize
SSH_HOST ?= amazonaws
DEST_DIR ?= ~/hw1

# rsync options: -a archive, -z compress, -v verbose, --delete remove remote files not present locally
RSYNC_OPTS ?= -azv --delete --exclude=.git --exclude='*.o' --exclude='$(TARGET)'

# Use ssh (which will read ~/.ssh/config). Do NOT set -i/-p here.
RSYNC_SSH ?= ssh

.PHONY: upload upload-dryrun

# Build then upload
# upload: all
# 	@echo "Creating remote directory $(SSH_HOST):$(DEST_DIR)"
# 	@ssh $(SSH_HOST) 'mkdir -p $(DEST_DIR)'
# 	@echo "Syncing current directory -> $(SSH_HOST):$(DEST_DIR)"
# 	rsync $(RSYNC_OPTS) -e '$(RSYNC_SSH)' ./ $(SSH_HOST):$(DEST_DIR)/

# Upload without building
upload:
	@echo "Creating remote directory $(SSH_HOST):$(DEST_DIR)"
	@ssh $(SSH_HOST) 'mkdir -p $(DEST_DIR)'
	@echo "Syncing current directory -> $(SSH_HOST):$(DEST_DIR)"
	rsync $(RSYNC_OPTS) -e '$(RSYNC_SSH)' ./ $(SSH_HOST):$(DEST_DIR)/

# Dry-run to preview changes
upload-dryrun:
	@echo "DRY RUN: rsync (no changes will be made)"
	rsync --dry-run $(RSYNC_OPTS) -e '$(RSYNC_SSH)' ./ $(SSH_HOST):$(DEST_DIR)/


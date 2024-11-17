#====================================================
# C2S2 Training Build Framework v2
#====================================================
# A build framework that supports multi-file
# applications written in C

# Dependencies: GCC

#----------------------------------------------------
# Setup

### Set compilation and linker flags
CFLAGS  := -Wall -O3
LDFLAGS := -lm

### Set commands correct based on operating system

ifeq ($(OS),Windows_NT)  # Windows
    RM_BUILD   := if exist build ( rmdir /s /q build )

intro :
	@echo ########################################
	@echo ## C2S2 Build System
	@echo ########################################

endif

ifneq ($(OS),Windows_NT) # Linux
    RM_BUILD   := rm -rf build

intro :
	@echo "########################################"
	@echo "## C2S2 Build System"
	@echo "########################################"

endif

TARGETS := $(basename $(notdir $(wildcard app/*.c)))
SOURCES := $(subst .c,.o,$(subst src,build,$(wildcard src/*.c)))

#----------------------------------------------------
# Compilation

.PHONY: intro build

all: $(TARGETS)
	@echo  - All executables built in build directory

build :
	@echo  - Making Clean Build Directory
	@${RM_BUILD}
	@mkdir build

test:
	echo $(SOURCES)

$(SOURCES) :
	$(eval source_file=$(subst .o,.c,$(subst build,src,$@)))
	$(eval file_name=$(notdir $@))

	@echo  - Building Object File: $(file_name)
	@gcc $(CFLAGS) -c -Isrc -o $@ $(source_file)

$(TARGETS) : intro build $(SOURCES)
	$(eval source_file=$(addprefix app/, $(addsuffix .c,$@)))
	$(eval target_file=$(basename $(subst app,build,$(source_file))))
	$(eval eval_file=$(notdir $(target_file)))

	@echo  - Building Executable: $(eval_file)
	@gcc $(CFLAGS) $(LDFLAGS) -Isrc -o $(target_file) $(source_file) $(SOURCES)

ifneq "$(MAKECMDGOALS)" "all"
ifneq "$(MAKECMDGOALS)" ""
	@echo  - Executable built in build directory
endif
endif

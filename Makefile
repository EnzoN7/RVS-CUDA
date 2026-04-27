BUILD_DIR ?= Build/Release
JOBS      ?= $(shell nproc)

MAKEFLAGS += --no-print-directory
export MAKEFLAGS

.PHONY: format format-check build configure run clean help

help:
	@echo "Cibles disponibles :"
	@echo "  make format        — formate les sources (.cpp/.hpp/.cu/.cuh) via clang-format"
	@echo "  make format-check  — vérifie le formatage sans modifier (CI)"
	@echo "  make configure     — (re)configure CMake dans $(BUILD_DIR)"
	@echo "  make build         — compile le projet"
	@echo "  make run CONFIG=…  — lance RVS avec la config (défaut : Config/RVS-A01.json)"
	@echo "  make clean         — supprime $(BUILD_DIR)"

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

$(BUILD_DIR)/Makefile: CMakeLists.txt
	$(MAKE) configure

format: $(BUILD_DIR)/Makefile
	@cmake --build $(BUILD_DIR) --target format

format-check: $(BUILD_DIR)/Makefile
	@cmake --build $(BUILD_DIR) --target format-check

build: $(BUILD_DIR)/Makefile
	@cmake --build $(BUILD_DIR) -j$(JOBS)

CONFIG ?= Config/RVS-A01.json
run: build
	cd Build && ./Release/RVS ../$(CONFIG)

clean:
	rm -rf $(BUILD_DIR)
